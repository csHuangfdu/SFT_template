"""
使用deepspeed进行SFT的详细模板
单机多卡分布式训练
"""
# DeepSpeed Team
import argparse
import json
import os
import math
import random
import sys
import numpy as np
import wandb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
    get_scheduler,
    set_seed,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import DataCollatorForSeq2Seq


#这里设置一下路径，免得找不到一些包
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from utils.parameter.parser_utils import parse_get_args
from utils.parameter.deepspeed_utils import get_train_ds_config
from utils.parameter.optimizer_utils import get_optimizer_grouped_parameters

from utils.universal_utils.utils import set_random_seed,print_rank_0,to_device,get_all_reduce_mean

from utils.model.tokenizer_utils import load_hf_tokenizer,create_hf_model
from utils.model.save_utils import save_hf_format,save_zero_three_model

from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import MyDataCollator

from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible


def main():
    # 获取从.sh那边传过来的参数（就是控制deepspeed训练的那些参数）
    # .sh就是启动整个项目的文件，可以在控制台输入【xxx.sh --xxx xxx】这样子启动deepspeed对模型进行训练
    args = parse_get_args()

    # 初始化wandb
    wdb = wandb.init(project="MyCode", resume="allow")
    wdb.config.update(
        dict(
            epoch=args.num_train_epochs, 
            lr=args.learning_rate, 
            batch_size=args.per_device_train_batch_size,
            dataset=args.data_path
            )
        )
                      


    # 初始化分布式训练的设置，以确保在多个 GPU 或节点上的模型训练可以正确协同工作
    if args.local_rank == -1:
        # local_rank 参数通常用于指定当前节点的 GPU 设备编号。
        # 如果 local_rank 等于 -1，表示当前不是在分布式训练模式下，而是在单机模式下
        device = torch.device("cuda")
    else:
        # 设置当前节点的 GPU 设备编号为 args.local_rank。为了确保模型运行在正确的 GPU 上
        torch.cuda.set_device(args.local_rank)

        # 将当前设备设置为 CUDA，并传递了 args.local_rank 以指定 GPU 设备编号
        device = torch.device("cuda", args.local_rank)

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')

        #初始化 DeepSpeed 库的分布式设置
        deepspeed.init_distributed()

    # 获取当前节点在全局分布式环境中的全局排名（global rank），并将其存储在 args.global_rank 变量中
    # 全局排名是一个唯一的标识，用于在分布式训练中识别每个节点的位置
    args.global_rank = torch.distributed.get_rank()


    # 获取 deepspeed 训练配置
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="v2_sft")


    # 设置 batch size
    # 将每个 GPU 设备上的微批处理大小设置为 args.per_device_train_batch_size。微批处理大小是模型每次在单个 GPU 上处理的样本数量。
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    # 设置总的训练批处理大小
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps


    # 设置随机种子
    set_random_seed(args.seed)


    # 分布式同步操作: 实现节点间的同步，确保不同节点上的操作在同一时间点执行
    torch.distributed.barrier()


    # 加载 tokenizer （包含设置padding）
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)


    # 创建model
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout)

    # 线性层转换为LoRA（通过lora_dim控制是否使用lora）
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        # 仅优化LoRA参数
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)


    # 准备数据（具有prompt的数据集）
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_output_path,
        args.seed)


    # 创建DataLoader
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    #此处data_collator可以按照需求自行修改
    data_collator = MyDataCollator(
        tokenizer,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=False
    )

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)


    # 设置权重衰减分组（一组有权重衰减，另一组没有）
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)
    

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
        

    # 设置学习率调度器（LR scheduler）
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )


    # 使用 DeepSpeed（一个深度学习加速库）来初始化模型、优化器和学习率调度器
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=args,
    config=ds_config,
    lr_scheduler=lr_scheduler,
    dist_init_required=True)


    # 启用梯度检查点（gradient checkpointing）,用于降低内存占用并减少计算成本
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    # 定义评估函数（评估一个深度学习模型在给定数据集上的性能，并计算模型的平均损失和困惑度）
    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity


    # 训练！！
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)
    wdb.log({'epoch': 0,'Step': 0,'evel_ppl': perplexity})



    # 初始化一个全局进度条
    total_steps = args.num_train_epochs * len(train_dataloader)
    progress_bar = tqdm(total=total_steps, leave=True, disable=(args.global_rank != 0))

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()

        for step, batch in enumerate(train_dataloader):
            del batch['sources']
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            if args.global_rank == 0:
                # Update the progress bar
                progress_bar.update(1)
                description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                progress_bar.set_description(description, refresh=False)
            
            if args.print_loss:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )

            # wandb输出loss
            wdb.log({'epoch': epoch,'Step': step,'Rank': torch.distributed.get_rank(),'train_loss': loss})

            # 每1000step输出一次evel_ppl
            if step % 1000 == 0:
                perplexity = evaluation(model, eval_dataloader)
                print_rank_0(f"ppl: {perplexity}", args.global_rank)
                wdb.log({'epoch': epoch,'Step': step,'evel_ppl': perplexity})


            model.backward(loss)
            model.step()
        
        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)
        print_rank_0(f'Sucessful saving the final model to {args.output_dir}', args.global_rank)

    

if __name__ == "__main__":
    main()


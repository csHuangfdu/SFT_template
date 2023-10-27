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

# 传入parser，设置需要获取的参数（可以设置默认值）
# .sh中传入的参数必须是parser中设置的参数，才能够接收
# 如需设置别的参数，在这个函数中新增参数项即可
# def parser_add_argument(parser):
#     parser.add_argument('--data_path',
#                         nargs='*',
#                         default=['Dahoas/rm-static'],
#                         help='Path to the training dataset. Accepted format:'
#                         '1) a single data path, 2) multiple datasets in the'
#                         'form: dataset1-path dataset2-path ...')
#     parser.add_argument('--data_output_path',
#         type=str,
#         default='/tmp/data_files/',
#         help=
#         'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
#     )
#     parser.add_argument("--model_name_or_path",
#         type=str,
#         help=
#         "Path to pretrained model or model identifier from huggingface.co/models.",
#         required=True,
#     )
#     parser.add_argument("--per_device_train_batch_size",
#         type=int,
#         default=16,
#         help="Batch size (per device) for the training dataloader.",
#     )
#     parser.add_argument("--per_device_eval_batch_size",
#         type=int,
#         default=16,
#         help="Batch size (per device) for the evaluation dataloader.",
#     )
#     parser.add_argument("--max_prompt_len",
#         type=int,
#         default=512,
#         help="The maximum sequence length.",
#     )
#     parser.add_argument("--max_ans_len",
#         type=int,
#         default=512,
#         help="The maximum sequence length.",
#     )

#     parser.add_argument("--learning_rate",
#         type=float,
#         default=1e-5,
#         help=
#         "Initial learning rate (after the potential warmup period) to use.",
#     )
#     parser.add_argument("--weight_decay",
#                         type=float,
#                         default=0.,
#                         help="Weight decay to use.")
#     parser.add_argument("--num_train_epochs",
#                         type=int,
#                         default=1,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--gradient_accumulation_steps",
#         type=int,
#         default=1,
#         help=
#         "Number of updates steps to accumulate before performing a backward/update pass.",
#     )
#     parser.add_argument("--lr_scheduler_type",
#         type=SchedulerType,
#         default="cosine",
#         help="The scheduler type to use.",
#         choices=[
#             "linear", "cosine", "cosine_with_restarts", "polynomial",
#             "constant", "constant_with_warmup"
#         ],
#     )
#     parser.add_argument("--num_warmup_steps",
#         type=int,
#         default=0,
#         help="Number of steps for the warmup in the lr scheduler.")
#     parser.add_argument("--output_dir",
#                         type=str,
#                         default=None,
#                         help="Where to store the model.")
#     parser.add_argument("--seed",
#                         type=int,
#                         default=42,
#                         help="A seed for reproducible training.")
#     # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
#     # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
#     parser.add_argument("--local_rank",
#                         type=int,
#                         default=-1,
#                         help="local_rank for distributed training on gpus")
#     parser.add_argument('--gradient_checkpointing',
#                         action='store_true',
#                         help='Enable HF gradient checkpointing for model.')
#     # store_true 表示如果命令行中有这个参数，则 args.disable_dropout 为 True, 否则默认为 False
#     parser.add_argument("--dropout",
#                         type=float,
#                         default=None,
#                         help="If dropout configured, use it. "
#                         "Otherwise, keep the default dropout configuration of the model.")
#     # deepspeed features
#     parser.add_argument('--offload',
#                         action='store_true',
#                         help='Enable ZeRO Offload techniques.')
#     parser.add_argument('--zero_stage',
#         type=int,
#         default=0,
#         help='ZeRO optimization stage for Actor model (and clones).')
    
#     ## Tensorboard logging
#     parser.add_argument('--enable_tensorboard',
#                         action='store_true',
#                         help='Enable tensorboard logging')
#     parser.add_argument('--tensorboard_path',
#                         type=str,
#                         default="step1_tensorboard")
#     ## Print loss
#     parser.add_argument('--print_loss',
#                         action='store_true',
#                         help='Prints loss at each step.')

#     return parser

# # 这里实现从.sh文件中（也可以等价于是控制台）获取参数
# # 这里仅实现了接收一条条参数的形式（实际上也可以传入一个json文件，但是这需要对应修改代码 TODO ）
# def parse_get_args():
#     # 创建ArgumentParser对象，用来设置需要获取的参数及获取.sh中传入的参数
#     # .sh中传入的参数必须是parser中设置的参数，才能够接收
#     parser = argparse.ArgumentParser(
#         description=
#         "Finetune a transformers model on a causal language modeling task")
    
#     # 设置需要获取的参数列表并自动获取传过来的参数
#     parser = parser_add_argument(parser=parser)

#     # 添加 deepspeed 特定的配置参数到解析器中，使得用户能够轻松地控制 deepspeed 的配置和行为，以满足不同的训练需求。
#     parser = deepspeed.add_config_arguments(parser)

#     # 使用解析器来解析命令行参数，并将结果存储在 args 变量中。args 变量包含了用户在命令行中指定的参数的值。
#     args = parser.parse_args()

#     return args

# # 获取 deepspeed 训练配置
# GLOBAL_BATCH_SIZE = 32
# MICRO_BATCH_SIZE = 4
# def get_train_ds_config(offload,
#                         stage=2,
#                         enable_hybrid_engine=False,
#                         inference_tp_size=1,
#                         release_inference_cache=False,
#                         pin_parameters=True,
#                         tp_gather_partition_size=8,
#                         max_out_tokens=512,
#                         enable_tensorboard=False,
#                         tb_path="",
#                         tb_name=""):
#     """
#     这是一个函数，它接受多个参数，用于配置 DeepSpeed 训练\n
#     offload: 是否启用参数离载（parameter offload）。如果为 True，参数会被加载到 CPU，否则为 "none"\n
#     stage：这是零优化（Zero Optimization）的阶段，默认值为 2\n
#     enable_hybrid_engine：是否启用混合引擎（hybrid engine）\n
#     inference_tp_size：用于配置推断的线程池（thread pool）大小，默认值为 1\n
#     release_inference_cache：是否释放推断缓存\n
#     pin_parameters：是否将参数固定（pin）到内存中\n
#     tp_gather_partition_size：用于配置线程池的收集分区大小\n
#     max_out_tokens：最大输出token数\n
#     enable_tensorboard：是否启用 TensorBoard 日志记录\n
#     tb_path：用于指定 TensorBoard 日志文件保存路径的参数\n
#     tb_name：用于指定 TensorBoard 日志的名称\n
#     """ 
#     device = "cpu" if offload else "none"
#     zero_opt_dict = {
#         "stage": stage,
#         "offload_param": {
#             "device": device
#         },
#         "offload_optimizer": {
#             "device": device
#         },
#         "stage3_param_persistence_threshold": 1e4,
#         "stage3_max_live_parameters": 3e7,
#         "stage3_prefetch_bucket_size": 3e7,
#         "memory_efficient_linear": False
#     }
#     return {
#         "train_batch_size": GLOBAL_BATCH_SIZE,
#         "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
#         "steps_per_print": 10,
#         "zero_optimization": zero_opt_dict,
#         "bfloat16": {
#             "enabled": "auto",
#             "loss_scale": 0,
#             "loss_scale_window": 1000,
#             "initial_scale_power": 16,
#             "hysteresis": 2,
#             "min_loss_scale": 1
#         },
#         "gradient_clipping": 1.0,
#         "prescale_gradients": False,
#         "wall_clock_breakdown": False,
#         "hybrid_engine": {
#             "enabled": enable_hybrid_engine,
#             "max_out_tokens": max_out_tokens,
#             "inference_tp_size": inference_tp_size,
#             "release_inference_cache": release_inference_cache,
#             "pin_parameters": pin_parameters,
#             "tp_gather_partition_size": tp_gather_partition_size,
#         },
#         "tensorboard": {
#             "enabled": enable_tensorboard,
#             "output_path": f"{tb_path}/ds_tensorboard_logs/",
#             "job_name": f"{tb_name}_tensorboard"
#         }
#     }


# # 设置随机种子
# def set_random_seed(seed):
#     if seed is not None:
#         set_seed(seed)
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)


# # 获取指定模型的tokenizer(及设置padding)
# def get_tokenizer(model_name_or_path, fast_tokenizer=True):
#     if "llama" in model_name_or_path:
#         from transformers.models.llama import LlamaTokenizer
#         tokenizer = LlamaTokenizer.from_pretrained(
#             model_name_or_path, fast_tokenizer=fast_tokenizer)
#         if tokenizer.pad_token is None:
#             # assert tokenizer.eos_token is not None
#             # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
#             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#             tokenizer.padding_side = 'right'
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name_or_path, fast_tokenizer=fast_tokenizer, trust_remote_code=True)
#         tokenizer.pad_token = tokenizer.eos_token
#         # make sure tokenizer is right pad in our logic
#         tokenizer.padding_side = 'right'
#     return tokenizer


# # 加载 Hugging Face Transformers 库中的 tokenizer
# def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
#     if os.path.exists(model_name_or_path):
#         # Locally tokenizer loading has some issue, so we need to force download
#         model_json = os.path.join(model_name_or_path, "config.json")
#         if os.path.exists(model_json):
#             model_json_file = json.load(open(model_json))
#             model_name = model_json_file.get("_name_or_path",
#                                              model_name_or_path)
#             tokenizer = get_tokenizer(model_name,
#                                       fast_tokenizer=fast_tokenizer)
#     else:
#         tokenizer = get_tokenizer(model_name_or_path,
#                                   fast_tokenizer=fast_tokenizer)

#     return tokenizer


# def configure_dropout(model_config, dropout):
#     if dropout is not None:
#         for key in ('dropout', 'attention_dropout', 'hidden_dropout',
#                     'activation_dropout'):
#             if hasattr(model_config, key):
#                 print(f"Setting model_config.{key} to {dropout}")
#                 setattr(model_config, key, dropout)

# # 创建model
# def create_hf_model(model_class,
#                     model_name_or_path,
#                     tokenizer,
#                     ds_config=None,
#                     dropout=None):
#     model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
#     configure_dropout(model_config, dropout)

#     # Note: dschf is defined in function scope to avoid global effects
#     # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
#     if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
#         dschf = HfDeepSpeedConfig(ds_config)
#     else:
#         dschf = None
    
#     model = model_class.from_pretrained(
#             model_name_or_path,
#             from_tf=bool(".ckpt" in model_name_or_path),
#             config=model_config,
#             trust_remote_code=True)
    
#     # llama use eos_token_id but not end_token_id
#     model.config.end_token_id = tokenizer.eos_token_id
#     # compatible with OPT and llama2
#     model.config.pad_token_id = model.config.eos_token_id
#     model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

#     return model


# def print_rank_0(msg, rank=0):
#     if rank <= 0:
#         print(msg)


# def to_device(batch, device):
#     output = {}
#     for k, v in batch.items():
#         try:
#             output[k] = v.to(device)
#         except:
#             output[k] = v
#     return output

# def get_all_reduce_mean(tensor):
#     torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
#     tensor = tensor / torch.distributed.get_world_size()
#     return tensor



# def get_optimizer_grouped_parameters(
#     model,
#     weight_decay,
#     lora_lr=5e-4,
#     no_decay_name_list=["bias", "LayerNorm.weight"],
#     lora_name_list=["lora_right_weight", "lora_left_weight"],
# ):
#     optimizer_grouped_parameters = [
#         {
#             "params": [
#                 p for n, p in model.named_parameters()
#                 if (not any(nd in n for nd in no_decay_name_list)
#                     and p.requires_grad and not any(nd in n
#                                                     for nd in lora_name_list))
#             ],
#             "weight_decay":
#             weight_decay,
#         },
#         {
#             "params": [
#                 p for n, p in model.named_parameters()
#                 if (not any(nd in n for nd in no_decay_name_list)
#                     and p.requires_grad and any(nd in n
#                                                 for nd in lora_name_list))
#             ],
#             "weight_decay":
#             weight_decay,
#             "lr":
#             lora_lr
#         },
#         {
#             "params": [
#                 p for n, p in model.named_parameters()
#                 if (any(nd in n
#                         for nd in no_decay_name_list) and p.requires_grad)
#             ],
#             "weight_decay":
#             0.0,
#         },
#     ]
#     if not optimizer_grouped_parameters[1]["params"]:
#         optimizer_grouped_parameters.pop(1)

#     return optimizer_grouped_parameters

# def save_hf_format(model, tokenizer, args, sub_folder=""):
#     # used to save huggingface format, so we can use it for hf.from_pretrained
#     model_to_save = model.module if hasattr(model, 'module') else model
#     CONFIG_NAME = "config.json"
#     WEIGHTS_NAME = "pytorch_model.bin"
#     output_dir = os.path.join(args.output_dir, sub_folder)
#     os.makedirs(output_dir, exist_ok=True)
#     output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
#     output_config_file = os.path.join(output_dir, CONFIG_NAME)
#     save_dict = model_to_save.state_dict()
#     for key in list(save_dict.keys()):
#         if "lora" in key:
#             del save_dict[key]
#     torch.save(save_dict, output_model_file)
#     model_to_save.config.to_json_file(output_config_file)
#     tokenizer.save_vocabulary(output_dir)

# def _z3_params_to_fetch(param_list):
#     return [
#         p for p in param_list
#         if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
#     ]

# def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
#     zero_stage_3 = (zero_stage == 3)
#     os.makedirs(save_dir, exist_ok=True)
#     WEIGHTS_NAME = "pytorch_model.bin"
#     output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

#     model_to_save = model_ema.module if hasattr(model_ema,
#                                                 'module') else model_ema
#     if not zero_stage_3:
#         if global_rank == 0:
#             torch.save(model_to_save.state_dict(), output_model_file)
#     else:
#         output_state_dict = {}
#         # 问题出在这里，不会保存不在named_parameters()里的参数，若是 model_to_save.state_dict() 则 OK
#         for k, v in model_to_save.named_parameters():

#             if hasattr(v, 'ds_id'):
#                 with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
#                                                                             ]),
#                                                        enabled=zero_stage_3):
#                     v_p = v.data.cpu()
#             else:
#                 v_p = v.cpu()
#             if global_rank == 0 and "lora" not in k:
#                 output_state_dict[k] = v_p
#         if global_rank == 0:
#             torch.save(output_state_dict, output_model_file)
#         del output_state_dict




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


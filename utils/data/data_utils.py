import os
import sys
import torch
from torch.utils.data import Subset, ConcatDataset
import numpy as np
import os

# 这里设置一下路径，免得找不到一些包
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from utils.data import raw_datasets


# 加载原始数据集
def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    if "Dahoas/rm-static" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank, "mkqa")
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank, "mkqa")
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(
            output_path, seed, local_rank, dataset_name
        )
    elif "local/jsonfile" in dataset_name:
        chat_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.path.pardir,
                os.path.pardir,
                os.path.pardir,
            )
        )
        if not (
            os.path.isfile(chat_path + '/data/train.json')
            and os.path.isfile(chat_path + '/data/eval.json')
        ):
            raise RuntimeError(
                f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
            )
        return raw_datasets.LocalJsonFileDataset(
            output_path, seed, local_rank, dataset_name, chat_path
        )
    elif "CherryDurian/shadow-alignment" in dataset_name:
        return raw_datasets.ShadowAlignmentDataset(output_path, seed,
                                                   local_rank, dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


# 对于多个dataset进行洗牌
def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


# 在raw_datasets中被使用到。
# 用于对原始数据集进行切分（对于那种只有一个train的，需要切分成train和evel这样）
def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    # reindex each time when using local jsonfile since it's more likely to get modified
    if (not os.path.isfile(index_file_name)) or (dataset_name == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


# 创建数据集(SFT格式的设计在此处完成)
def create_dataset(local_rank, dataset_name, output_path, seed):
    raw_dataset = get_raw_dataset(
        dataset_name,
        output_path,
        seed,
        local_rank,
    )

    train_dataset = raw_dataset.get_train_data()

    eval_dataset = raw_dataset.get_eval_data()

    return train_dataset, eval_dataset


# 创建具有“特定格式”的数据集
def create_prompt_dataset(local_rank, data_path: list, output_path, seed):
    """
    Creates the prompt dataset\n
    local_rank：本地进程的标志，通常用于分布式训练\n
    data_path：训练数据集的路径（可以是单个路径或多个路径）\n
    output_path：数据集输出路径，用于保存数据集的缓存文件\n
    seed：随机数生成器的种子，用于数据处理的随机性\n
    """
    os.makedirs(output_path, exist_ok=True)

    if len(data_path) == 1:  # Single dataset.
        train_dataset, eval_dataset = create_dataset(
            local_rank, data_path[0], output_path, seed
        )
    else:  # Blending datasets.
        train_datasets = []
        eval_datasets = []
        train_size = 0
        eval_size = 0
        for d_path in data_path:
            train_dataset, eval_dataset = create_dataset(
                local_rank, d_path, output_path, seed
            )
            train_datasets.append(train_dataset)
            eval_datasets.append(eval_dataset)
            train_size += len(train_dataset)
            eval_size += len(eval_dataset)
        train_dataset = ConcatDataset(train_datasets)
        shuffle_idx = get_shuffle_idx(seed, train_size)
        train_dataset = Subset(train_dataset, shuffle_idx.tolist())
        eval_dataset = ConcatDataset(eval_datasets)
        shuffle_idx = get_shuffle_idx(seed, eval_size)
        eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

    torch.distributed.barrier()

    return train_dataset, eval_dataset

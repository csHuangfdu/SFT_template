import json
import os
import math


from transformers import (
    AutoTokenizer,
    AutoConfig,
)

from transformers.deepspeed import HfDeepSpeedConfig

# 获取指定模型的tokenizer(及设置padding)
def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path:
        from transformers.models.llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    return tokenizer


# 加载 Hugging Face Transformers 库中的 tokenizer
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    return tokenizer


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)

# 创建model
def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    configure_dropout(model_config, dropout)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    
    model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True)
    
    # llama use eos_token_id but not end_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    # compatible with OPT and llama2
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

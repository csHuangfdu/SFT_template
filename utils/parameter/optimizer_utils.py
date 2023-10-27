def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad
                    and not any(nd in n for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad
                    and any(nd in n for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)

    return optimizer_grouped_parameters

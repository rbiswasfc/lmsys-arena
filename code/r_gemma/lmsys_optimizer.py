import bitsandbytes as bnb
from torch import optim


def get_optimizer(cfg, model, print_fn=None):
    _optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "AdamW8bit": bnb.optim.Adam8bit,
    }
    assert cfg.optimizer.name in _optimizers, f"Optimizer {cfg.optimizer.name} not supported"

    no_decay = ["bias", "LayerNorm.weight"]
    head_layer_name = "classification_head"

    # start with all of the candidate parameters
    param_dict = {name: param for name, param in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

    # head & body params
    param_dict_head = {name: param for name, param in param_dict.items() if head_layer_name in name}
    param_dict_body = {name: param for name, param in param_dict.items() if head_layer_name not in name}

    # create groups ---
    head_params_no_decay = [param for name, param in param_dict_head.items() if any(nd in name for nd in no_decay)]
    head_params_decay = [param for name, param in param_dict_head.items() if not any(nd in name for nd in no_decay)]

    body_params_no_decay = [param for name, param in param_dict_body.items() if any(nd in name for nd in no_decay)]
    body_params_decay = {name: param for name, param in param_dict_body.items() if not any(nd in name for nd in no_decay)}

    # --
    body_params_decay_lora_b = [param for name, param in body_params_decay.items() if "lora_B" in name]

    body_params_decay_lora_a = [param for name, param in body_params_decay.items() if "lora_B" not in name]

    lora_b_scale = 10.0  # 10.0
    optim_groups = [
        {"params": head_params_no_decay, "lr": cfg.optimizer.head_lr, "weight_decay": 0},
        {"params": head_params_decay, "lr": cfg.optimizer.head_lr, "weight_decay": cfg.optimizer.weight_decay},
        {"params": body_params_no_decay, "lr": cfg.optimizer.lr, "weight_decay": 0},
        {
            "params": body_params_decay_lora_b,
            "lr": cfg.optimizer.lr * lora_b_scale,
            "weight_decay": cfg.optimizer.weight_decay,
        },  # less weight decay for body
        {
            "params": body_params_decay_lora_a,
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
        },
    ]

    if print_fn is not None:
        n_head_params_no_decay = round(sum(p.numel() for p in head_params_no_decay) / 1e6, 2)
        n_head_params_decay = round(sum(p.numel() for p in head_params_decay) / 1e6, 2)
        n_body_params_no_decay = round(sum(p.numel() for p in body_params_no_decay) / 1e6, 2)
        # n_body_params_decay = round(sum(p.numel() for p in body_params_decay)/1e6, 2)
        n_body_params_decay_lora_b = round(sum(p.numel() for p in body_params_decay_lora_b) / 1e6, 2)
        n_body_params_decay_lora_a = round(sum(p.numel() for p in body_params_decay_lora_a) / 1e6, 2)

        print_fn(f"n_head_params_no_decay: {n_head_params_no_decay} M")
        print_fn(f"n_head_params_decay: {n_head_params_decay} M")
        print_fn(f"n_body_params_no_decay: {n_body_params_no_decay} M")
        # print_fn(f"n_body_params_decay: {n_body_params_decay}M")
        print_fn(f"n_body_params_decay_lora_b: {n_body_params_decay_lora_b} M")
        print_fn(f"n_body_params_decay_lora_a: {n_body_params_decay_lora_a} M")

    # Create AdamW optimizer and use the fused version if it is available
    optimizer = _optimizers[cfg.optimizer.name](
        optim_groups,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    return optimizer

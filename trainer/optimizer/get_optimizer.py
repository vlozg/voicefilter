import torch
from torch.optim import Adam, AdamW, lr_scheduler

from utils.adabound import AdaBound


def get_optimizer(config, model,chkpt=None):
    # Set optimizer
    if config.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=config.train.optimizer_param.initial_lr,
                             final_lr=config.train.optimizer_param.final_lr)
    elif config.train.optimizer == 'adam':
        optimizer = Adam(model.parameters(),
                                     lr=config.train.optimizer_param.lr)
    elif config.train.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(),
                                     lr=config.train.optimizer_param.lr,
                                     weight_decay=config.train.optimizer_param.weight_decay)    
    else:
        raise NotImplementedError("%s optimizer not supported" % config.train.optimizer)

    if chkpt is not None:
        optimizer.load_state_dict(chkpt['optimizer'])
        step = chkpt['step']-1
    else:
        step = -1

    # Set scheduler
    param = config.train.scheduler_param
    if config.train.get("scheduler") is None:
        scheduler = None
    elif config.train.scheduler == '1cycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                     max_lr=param.max_lr,
                                     total_steps=config.train.max_step,
                                     pct_start=param.get("pct_start", 0.3),
                                     div_factor=param.max_lr/param.min_lr,
                                     final_div_factor=param.min_lr/param.get("final_lr", param.min_lr/20),
                                     three_phase=param.get("three_phase", False),
                                     anneal_strategy=param.get("anneal_strategy", "cos"),
                                     last_epoch=step)
    else:
        raise NotImplementedError("%s scheduler not supported" % config.train.optimizer)

    return optimizer, scheduler
import torch
from torch.optim import Adam, AdamW, lr_scheduler

from utils.adabound import AdaBound


def get_optimizer(config, model):

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

    # Set scheduler
    if config.train.scheduler == '1cycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                     max_lr=config.train.scheduler_param.max_lr,
                                     total_steps=config.train.max_step,
                                     pct_start=config.train.scheduler_param.pct_start,
                                     div_factor=config.train.scheduler_param.max_lr/config.train.scheduler_param.min_lr,
                                     anneal_strategy=config.train.scheduler_param.anneal_strategy,)
    elif config.train.scheduler is None:
        scheduler = None
    else:
        raise NotImplementedError("%s scheduler not supported" % config.train.optimizer)

    return optimizer, scheduler
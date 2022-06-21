# Reference: 
# [1] Learning rate finder (Sylvain Gugger@fast.ai)
#    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
# [2] pytorch-lr-finder
#    https://github.com/davidtvs/pytorch-lr-finder/tree/acc5e7ee7711a460bf3e1cc5c5f05575ba1e1b4b

import os
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np

from utils.audio import Audio

from model.get_model import get_vfmodel, get_embedder, get_forward
from loss.get_criterion import get_criterion
from trainer.optimizer.get_optimizer import get_optimizer
from trainer.validate import validate

from datasets.dataloader import create_dataloader

import matplotlib.pylab as plt

import json


class LinearLR(_LRScheduler):

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)

        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)

        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


def lr_finder(
        config, pt_dir, writer, logger, hp_str, 
        init_lr = 1e-8, end_lr=10., step_mode="exp", num_iter=100,
        smooth_f = 0.05, diverge_th=5
    ):

    # Create and load dataset
    logger.info("Start making train set")
    trainloader = create_dataloader(config, scheme="train")

    # Start using exp config from this onward (for simplication)
    config = config.experiment

    # Train variables init
    step = 0
    accum = 0
    accum_loss = 0
    it = iter(trainloader) # use iterator instead of for loop
    device = "cuda" if config.use_cuda else "cpu"

    # Remove scheduler from this trainer
    config.train.scheduler = None

    # Init model, embedder, optim, criterion
    audio = Audio(config)
    embedder = get_embedder(config, train=False, device=device)
    model, chkpt = get_vfmodel(config, train=True, device=device)
    train_forward, _ = get_forward(config)
    criterion = get_criterion(config)


    if config.train.get("resume_from_chkpt") is True:
        logger.info("Resuming optimizer and scheduler from checkpoint: %s" % config.model.pretrained_chkpt)
        optimizer, _ = get_optimizer(config, model, chkpt)
    else:
        logger.info("New optimizer")
        optimizer, _ = get_optimizer(config, model, None)

    # Check resume from checkpoint
    if chkpt is not None:
        logger.info("Resuming from checkpoint: %s" % config.model.pretrained_chkpt)
        # will use new given hparams.
        if hp_str != chkpt['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting learning rate finding")


    # Replace with init learning rate
    if not isinstance(init_lr, list):
        new_lrs = [init_lr] * len(optimizer.param_groups)
    if len(new_lrs) != len(optimizer.param_groups):
        raise ValueError(
            "Length of `new_lrs` is not equal to the number of parameter groups "
            + "in the given optimizer"
        )

    for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
        param_group["lr"] = new_lr
    

    history = {"lr": [], "loss": [], "raw_loss": []}
    best_loss = 0.

    if step_mode.lower() == "exp":
        lr_schedule = ExponentialLR(optimizer, end_lr, num_iter)
    elif step_mode.lower() == "linear":
        lr_schedule = LinearLR(optimizer, end_lr, num_iter)
    else:
        raise ValueError("expected one of (exp, linear), got {}".format(step_mode))


    # Start training
    while (config.train.max_step == -1 or step < num_iter):
        
        try:
            batch = next(it)
        except StopIteration:
            logger.info("Last element of dataloader reached, restart dataloader")
            it = iter(trainloader)


        _, _, loss = train_forward(model, embedder, batch, criterion, device)

        loss /= config.train.grad_accumulate
        loss.backward()
        accum_loss += loss.item()
        accum += 1
        
        if accum_loss > 1e8 or math.isnan(accum_loss):
            save_path = os.path.join(pt_dir, 'err_chkpt_%d.pt' % step)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'hp_str': hp_str,
            }, save_path)

            logger.error("Loss exploded to %.02f at step %d!" % (accum_loss, step))
            raise Exception("Loss exploded")

        # Skip gradient step if not accumulated enough
        if accum % config.train.grad_accumulate != 0: continue

        optimizer.step()
        optimizer.zero_grad()
        loss = accum_loss

        # Update the learning rate
        history["lr"].append(lr_schedule.get_last_lr()[0])
        history["raw_loss"].append(loss)
        lr_schedule.step()

        # Track the best loss and smooth it if smooth_f is specified
        if step == 0:
            best_loss = loss
        else:
            if smooth_f > 0:
                loss = smooth_f * loss + (1 - smooth_f) * history["loss"][-1]
            if loss < best_loss:
                best_loss = loss

        # Check if the loss has diverged; if it has, stop the test
        history["loss"].append(loss)
        logger.info("Wrote history at step %d" % step)
        if loss > diverge_th * best_loss:
            print("Stopping early, the loss has diverged")
            break

        accum = 0
        step += 1
        accum_loss = 0


    # Get the data to plot from the history dictionary.
    lrs = history["lr"]
    losses = history["loss"]

    # # Create the figure and axes object
    # fig, ax = plt.subplots()

    # # # Plot loss as a function of the learning rate
    # ax.plot(lrs, losses)

    # Plot the suggested LR
    # 'steepest': the point with steepest gradient (minimal gradient)
    logger.info("LR suggestion: steepest gradient")
    min_grad_idx = None
    try:
        min_grad_idx = (np.gradient(np.array(losses))).argmin()
        logger.info("Suggested LR: {:.2E}".format(lrs[min_grad_idx]))
    except ValueError:
        logger.error(
            "Failed to compute the gradients, there might not be enough points."
        )

    # Save finding result to json
    result = {
        "min_grad_lr": float(lrs[min_grad_idx]) if min_grad_idx is not None else None,
        "min_grad_idx": int(min_grad_idx),
        "history": history
    }

    json_object = json.dumps(result, indent = 4)
    with open(os.path.join("test_results/lr_finding", config.name + "_lr_finding.json"), "w") as f:
        f.write(json_object)

    
    # if min_grad_idx is not None:
    #     ax.scatter(
    #         lrs[min_grad_idx],
    #         losses[min_grad_idx],
    #         s=75,
    #         marker="o",
    #         color="red",
    #         zorder=3,
    #         label="steepest gradient",
    #     )
    #     ax.legend()

    # ax.set_xscale("log")
    # ax.set_xlabel("Learning rate")
    # ax.set_ylabel("Loss")

    # # Save figure
    # plt.savefig(config.name + "_lr_finding.png")
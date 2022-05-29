from cmath import log
import os
import math
import torch

from utils.adabound import AdaBound
from utils.audio import Audio

from model.get_model import get_vfmodel, get_embedder, get_forward
from loss.get_criterion import get_criterion
from .validate import validate

def trainer(config, pt_dir, trainloader, testloader, writer, logger, hp_str):

    # Train variables init
    step = 0
    accum = 0
    accum_loss = 0
    it = iter(trainloader) # use iterator instead of for loop


    # Init model, embedder, optim, criterion
    audio = Audio(config)
    embedder = get_embedder(config, train=False, device="cuda")
    model, chkpt = get_vfmodel(config, train=True, device="cuda")
    train_forward, _ = get_forward(config)
    criterion = get_criterion(config)

    if config.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=config.train.adabound.initial,
                             final_lr=config.train.adabound.final)
    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.train.optimizer_param.lr)
    else:
        raise Exception("%s optimizer not supported" % config.train.optimizer)


    # Check resume from checkpoint
    if chkpt is not None:
        logger.info("Resuming from checkpoint: %s" % config.model.pretrained_chkpt)
        optimizer.load_state_dict(chkpt['optimizer'])
        step = chkpt['step']

        # will use new given hparams.
        if hp_str != chkpt['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")


    # Start training
    while (config.train.max_step == -1 or step < config.train.max_step):
        
        try:
            batch = next(it)
        except StopIteration:
            logger.info("Last element of dataloader reached, restart dataloader")
            it = iter(trainloader)


        _, _, loss = train_forward(model, embedder, batch, criterion, "cuda")

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
        accum = 0
        step += 1
        accum_loss /= config.train.grad_accumulate

        # write loss to tensorboard
        if step % config.train.summary_interval == 0:
            writer.log_training(accum_loss, step)
            logger.info("Wrote summary at step %d" % step)

        accum_loss = 0

        # save checkpoint file
        if step % config.train.checkpoint_interval == 0:
            save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'hp_str': hp_str,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)
            validate(model, embedder, testloader, train_forward, criterion, audio, writer, logger, step)
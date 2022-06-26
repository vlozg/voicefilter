import os
import math
import torch

from utils.audio import Audio

from model.get_model import get_vfmodel, get_embedder, get_forward
from loss.get_criterion import get_criterion
from trainer.optimizer.get_optimizer import get_optimizer
from trainer.validate import validate

from datasets.dataloader import create_dataloader


def trainer(config, pt_dir, writer, logger, hp_str):

    # Create and load dataset
    logger.info("Start making validate set")
    testloader = create_dataloader(config, scheme="eval")
    logger.info("Start making first train set")
    config.experiment.dataset.train = config.experiment.dataset.multireader.train_1
    trainloader_1 = create_dataloader(config, scheme="train")
    multiread_w1 = config.experiment.train.multireader_w[0]
    logger.info("Start making seccond train set")
    config.experiment.dataset.train = config.experiment.dataset.multireader.train_2
    trainloader_2 = create_dataloader(config, scheme="train")
    multiread_w2 = config.experiment.train.multireader_w[1]

    # Start using exp config from this onward (for simplication)
    config = config.experiment

    # Train variables init
    step = 0
    accum = 0
    accum_loss = 0
    it1 = iter(trainloader_1) # use iterator instead of for loop
    it2 = iter(trainloader_2) # use iterator instead of for loop
    device = "cuda" if config.use_cuda else "cpu"


    # Init model, embedder, optim, criterion
    audio = Audio(config)
    embedder = get_embedder(config, train=False, device=device)
    model, chkpt = get_vfmodel(config, train=True, device=device)
    train_forward, _ = get_forward(config)
    criterion = get_criterion(config)

    if config.train.get("resume_from_chkpt") is True:
        logger.info("Resuming optimizer and scheduler from checkpoint: %s" % config.model.pretrained_chkpt)
        optimizer, scheduler = get_optimizer(config, model, chkpt)
        if chkpt is not None:
            step = chkpt['step']
    else:
        logger.info("New optimizer")
        optimizer, scheduler = get_optimizer(config, model, None)

    # Check resume from checkpoint
    if chkpt is not None:
        logger.info("Resuming from checkpoint: %s" % config.model.pretrained_chkpt)
        # will use new given hparams.
        if hp_str != chkpt['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")


    # Start training
    while (config.train.max_step == -1 or step < config.train.max_step):
        
        try:
            batch_1 = next(it1)
        except StopIteration:
            logger.info("Last element of dataloader reached, restart dataloader")
            it1 = iter(trainloader_1)
        
        try:
            batch_2 = next(it2)
        except StopIteration:
            logger.info("Last element of dataloader reached, restart dataloader")
            it2 = iter(trainloader_2)

        _, _, loss_1 = train_forward(model, embedder, batch_1, criterion, device)
        _, _, loss_2 = train_forward(model, embedder, batch_2, criterion, device)

        loss = multiread_w1*loss_1 + multiread_w2*loss_2
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
        if scheduler is not None:
            scheduler.step()
            
        optimizer.zero_grad()
        accum = 0
        step += 1

        # write loss to tensorboard
        if step % config.train.summary_interval == 0:
            writer.log_training(accum_loss, step)
            logger.info("Wrote summary at step %d" % step)

        accum_loss = 0

        # save a beckup checkpoint file if backup interval configured
        if config.train.get("backup_interval"):
            if step % config.train.backup_interval == 0:
                save_path = os.path.join(pt_dir, 'backup.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'hp_str': hp_str,
                }, save_path)
                logger.info("Backuped at step %d" % step)

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
            validate(model, embedder, testloader, train_forward, criterion, device, audio, writer, logger, step)
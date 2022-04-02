import os
import math
import torch
from contextlib import ExitStack

from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.voicefilter import VoiceFilter
from model.embedder import SpeechEmbedder
from .power_law_loss import PowerLawCompLoss
from .gdrive import GDrive
from torch.profiler import profile, record_function, ProfilerActivity


def trainer(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str, profiling=False):
    # load embedder
    embedder_pt = torch.load(args.embedder_path, "cpu")
    embedder = SpeechEmbedder(hp).cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    model = VoiceFilter(hp).cuda()
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    # drive = GDrive()

    step = 0
    accum = 0
    accum_loss = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path, "cpu")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    # criterion = nn.MSELoss()
    criterion = PowerLawCompLoss()
    model.train()

    # Create iterator
    it = iter(trainloader)
    
    
    while (hp.train.max_step == -1 or step < hp.train.max_step):
        # Next training batch (randomly generated)
        dvec_mels, target_mag, target_phase, mixed_mag, mixed_phase, target_stft, mixed_stft = next(it)
        
        # Move to cude
        with ExitStack() as stack:
            if profiling:
                stack.enter_context(record_function("to_cuda"))
            target_stft = target_stft.cuda(non_blocking=True)
            mixed_stft = mixed_stft.cuda(non_blocking=True)
            # mixed_mag = mixed_mag.cuda(non_blocking=True)
            # mixed_phase = mixed_phase.cuda(non_blocking=True)
            # target_mag = target_mag.cuda(non_blocking=True)
            # target_phase = target_phase.cuda(non_blocking=True)
            dvec_mels = [mel.cuda(non_blocking=True) for mel in dvec_mels]

        # Get dvec
        with ExitStack() as stack:
            if profiling:
                stack.enter_context(record_function("dvec"))
            dvec_list = list()
            for mel in dvec_mels:
                dvec = embedder(mel)
                dvec_list.append(dvec)
            dvec = torch.stack(dvec_list, dim=0)
            dvec = dvec.detach()
            # torch.cuda.empty_cache()
            # mask = model(mixed_mag, dvec)

        # Forward
        with ExitStack() as stack:
            if profiling:
                stack.enter_context(record_function("forward"))
            mask = model(torch.pow(mixed_stft.abs(), 0.3), dvec)
            loss = criterion(mask, mixed_stft, target_stft)

        # Backward
        with ExitStack() as stack:
            if profiling:
                stack.enter_context(record_function("backward"))
            loss.backward()
            accum_loss += loss.item()
            accum += 1
        
        # Check exploding gradient
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

        # Optimizer step (w/ gradient accumulation)
        if accum % hp["train"]["grad_accumulate"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            accum = 0
            step += 1
            accum_loss /= hp["train"]["grad_accumulate"]

            # write loss to tensorboard
            if step % hp.train.summary_interval == 0:
                writer.log_training(accum_loss, step)
                logger.info("Wrote summary at step %d" % step)

            accum_loss = 0

            # 1. save checkpoint file to resume training
            # 2. evaluate and save sample to tensorboard
            # backup brrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
            if step % hp.train.checkpoint_interval == 0:
                save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'hp_str': hp_str,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)
                validate(audio, model, embedder, testloader, writer, logger, step)

                # drive.Upload(save_path, "1sWAUt5vfyD97Cq85J8_zuwMeX4tmfEiZ")
                # # Nén file
                # os.system(f'zip -j ./tensorboard.zip ./{log_dir}/*')
                # drive.Upload('tensorboard.zip', "1sWAUt5vfyD97Cq85J8_zuwMeX4tmfEiZ")
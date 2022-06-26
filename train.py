import os
import time
import shutil
import logging
import argparse
import traceback

from utils.hparams import HParam
from utils.writer import MyWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/best.yaml',
                        help="folder contain yaml files for configuration")
    parser.add_argument('--clean_rerun', type=bool, default=False,
                        help="remove old checkpoint and log. Default: false")
    parser.add_argument('-r', '--resume', type=str, default="backup",
                        help="resume from checkpoint. Default: backup")
    args = parser.parse_args()

    config = HParam(args.config)
    exp = config["experiment"]
    env = config["env"]

    with open(args.config, 'r') as f:
        # store hparams as string
        hp_str = ''.join(f.readlines())


    chkpt_dir = os.path.join(env.base_dir, env.log.chkpt_dir, exp.name)
    log_dir = os.path.join(env.base_dir, env.log.log_dir, exp.name)

    # Cleanup existed logs
    if args.clean_rerun:
        shutil.rmtree(chkpt_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)

    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
        

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (exp.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = MyWriter(exp.audio, log_dir)

    if args.resume:
        if args.resume == "backup":
            if os.path.isfile(os.path.join(chkpt_dir, 'backup.pt')):
                args.resume = os.path.join(chkpt_dir, 'backup.pt')
                config.experiment.train["resume_from_chkpt"] = True

        if os.path.exists(args.resume):
            logger.info(f"Resume training from checkpoint {args.resume}")
            exp["model"]["pretrained_chkpt"] = args.resume
            config.experiment.train["resume_from_chkpt"] = True


    if exp.get("trainer_type") == "regular" or exp.get("trainer_type") is None:
        from trainer.trainer import trainer
    elif exp.get("trainer_type") == "multireader":
        from trainer.multireader_trainer import trainer
    else:
        raise NotImplementedError(f"{exp.get('trainer_type')} trainer not implemented")


    try:
        trainer(config, chkpt_dir, writer, logger, hp_str)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
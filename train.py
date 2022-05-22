import os
import time
import shutil
import logging
import argparse
import traceback

from trainer.trainer import trainer
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/best.yaml',
                        help="folder contain yaml files for configuration")
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
    if exp.clean_rerun:
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

    logger.info("Start making validate set")
    testloader = create_dataloader(config, scheme="eval")
    logger.info("Start making train set")
    trainloader = create_dataloader(config, scheme="train")

    try:
        trainer(exp, chkpt_dir, trainloader, testloader, writer, logger, hp_str)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
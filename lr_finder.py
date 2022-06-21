import os
import time
import shutil
import logging
import argparse
import traceback

from utils.hparams import HParam

from trainer.lr_finder_trainer import lr_finder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/best.yaml',
                        help="folder contain yaml files for configuration")
    parser.add_argument('-n', '--num_iter', type=int, default=100,
                        help="number of iter to find learning rate")
    parser.add_argument('--max_lr', type=float, default=10.,
                        help="maximum learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help="minimum learning rate")
    args = parser.parse_args()

    config = HParam(args.config)
    exp = config["experiment"]
    env = config["env"]

    with open(args.config, 'r') as f:
        # store hparams as string
        hp_str = ''.join(f.readlines())


    chkpt_dir = os.path.join(env.base_dir, env.log.chkpt_dir, exp.name)
    log_dir = os.path.join(env.base_dir, env.log.log_dir, exp.name)

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

    try:
        lr_finder(config, chkpt_dir, None, logger, hp_str, init_lr=args.min_lr, end_lr=args.max_lr, num_iter=args.num_iter)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
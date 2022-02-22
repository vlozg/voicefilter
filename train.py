import os
import time
import logging
import argparse

from utils.trainer import trainer
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Root directory of run.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="folder contain yaml files for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model. Used for both logging and saving checkpoints.")
    parser.add_argument('--train_set', type=str, nargs="+",
                        help="List of dataset subset name for train set.")
    parser.add_argument('--test_set', type=str, nargs="+",
                        help="List of dataset subset name for train set.")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        # store hparams as string
        hp_str = ''.join(f.readlines())

    pt_dir = os.path.join(args.base_dir, hp.log.chkpt_dir, args.model)
    os.makedirs(pt_dir, exist_ok=True)

    log_dir = os.path.join(args.base_dir, hp.log.log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)

    chkpt_path = args.checkpoint_path if args.checkpoint_path is not None else None

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = MyWriter(hp, log_dir)

    trainloader = create_dataloader(hp, "generate", dataset_detail=args.train_set, scheme="train")
    testloader = create_dataloader(hp, "generate", dataset_detail=args.test_set, scheme="test")

    trainer(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str)

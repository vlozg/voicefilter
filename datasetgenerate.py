import argparse

from utils.hparams import HParam
from datasets.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/best.yaml',
                        help="folder contain yaml files for configuration")
    args = parser.parse_args()

    config = HParam(args.config)

    create_dataloader(config, scheme="eval")
    create_dataloader(config, scheme="train")
    create_dataloader(config, scheme="test")
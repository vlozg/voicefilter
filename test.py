import os
import time
import logging
import argparse
import traceback

import pandas as pd

from utils.hparams import HParam
from tester.tester import tester
from datasets.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/best.yaml',
                        help="folder contain yaml files for configuration")
    args = parser.parse_args()

    config = HParam(args.config)
    exp = config["experiment"]
    env = config["env"]

    log_dir = os.path.join(env.base_dir, env.log.log_dir, exp.name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d-test.log' % (exp.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info("Start making test set")
    testloader = create_dataloader(config, scheme="test")

    try:
        test_result = tester(exp, testloader, logger)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()

    dataset_path = config.experiment.dataset.test.file_path
    logger.info(f"Appending test result to {dataset_path}")
    data_df = pd.read_csv(config.experiment.dataset.test.file_path)
    
    for key in test_result:
        data_df[key+"_"+exp.name] = test_result[key]

    data_df.to_csv(dataset_path,index=False)
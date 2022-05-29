import os
import time
import logging
import argparse
import traceback
from pathlib import Path
import json

import pandas as pd

from utils.hparams import HParam
from tester.tester import tester
from datasets.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/best.yaml',
                        help="yaml files for model configuration. Default: config/best.yaml")
    parser.add_argument('-p', '--chkpt', type=str, required=True,
                        help="path to pre-trained VF model checkpoint")
    parser.add_argument('-d', '--data_config', type=str, required=True,
                        help="yaml files for data used to test and testing configuration. Default: false")
    args = parser.parse_args()


    ###
    # Merge config, only do merge for known key
    ###
    config = HParam(args.config)
    data_config = HParam(args.data_config)

    config.experiment.use_cuda = data_config.experiment.use_cuda
    config.experiment.dataset = data_config.experiment.dataset
    config.experiment.model.pretrained_chkpt = args.chkpt

    exp = config["experiment"]
    env = config["env"]
    test_exp_name = Path(args.data_config).stem # Ex: ../../libri_eval.yaml -> libri_eval


    ###
    # Init logger and create dataloader
    ###
    log_dir = os.path.join(env.base_dir, env.log.log_dir, "test", test_exp_name, exp.name)
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


    ###
    # Load test record and write test config
    ###
    test_record_path = os.path.join(env.base_dir, "test_results", test_exp_name + ".json")
    if os.path.exists(test_record_path):
        with open(test_record_path, "r") as f:
            test_record = json.load(f)
    else:
        test_record = dict()

    if test_record.get("data"):
        if test_record["data"] != config.experiment.dataset.test.file_path:
            logger.info("Different dataset file path!")
            logger.info(f"Data path in config: {config.experiment.dataset.test.file_path}")
            logger.info(f"Data path in record: {test_record['data']}")
            raise Exception("Different dataset file path")
    else:
        test_record["data"] = config.experiment.dataset.test.file_path


    ###
    # Conduct testing
    ###
    try:
        test_result = tester(exp, testloader, logger)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
        exit()


    ###
    # Append new test result to file, then rewrite
    ###
    logger.info(f"Appending test result to {test_record_path}")
    
    for key in test_result.keys():
        test_result[key] = test_result[key].tolist()

    test_result = {
        "config": args.config,
        "metrics": test_result
    }

    if test_record.get(exp.name):
        # Test result on same model exist
        if test_record[exp.name] != test_result:
            logger.info("Test result exist, but different result this time!")
            i = 0
            while test_record.get(f"{exp.name}_{i}"): i+=1
            logger.info(f"Result will be stored within {exp.name}_{i}")
            test_record[f"{exp.name}_{i}"] = test_result
    else:
        test_record[exp.name] = test_result

    json_object = json.dumps(test_record, indent = 4)
    with open(test_record_path, "w") as f:
        f.write(json_object)
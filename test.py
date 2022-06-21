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
                        help="yaml files for data used to test and testing configuration")
    parser.add_argument('--use_cuda', type=bool, default=None,
                        help="use cuda for testing, overwrite test config. Default: follow test config file")
    parser.add_argument('--postfix', type=str, default=None,
                        help="add postfix to test result name. Default: not add postfix")
    parser.add_argument('--skip_exist', type=str, default="False",
                        help="skip testing if result exist. Default: False")
    parser.add_argument('--output_audio', type=str, default=None,
                        help="Output infered audio. Default: do not output infered audio")                        
    args = parser.parse_args()


    ###
    # Merge config, only do merge for known key
    ###
    config = HParam(args.config)
    data_config = HParam(args.data_config)

    if args.use_cuda is None:
        config.experiment.use_cuda = data_config.experiment.use_cuda
    else:
        config.experiment.use_cuda = args.use_cuda
    config.experiment.dataset = data_config.experiment.dataset
    config.experiment.model.pretrained_chkpt = args.chkpt
    config.experiment.model.batch_size = 1

    exp = config["experiment"]
    env = config["env"]
    test_exp_name = Path(args.data_config).stem # Ex: ../../libri_eval.yaml -> libri_eval
    if args.postfix:
        exp.name = exp.name + args.postfix

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


    if test_record.get(exp.name):
        # Test result on same model exist
        logger.info("Test result exist")
        i = 0
        while test_record.get(f"{exp.name}_{i}"): i+=1
        exp.name = f"{exp.name}_{i}"
        logger.info(f"Result will be stored within {exp.name}")
        
    if args.output_audio == "auto":
        output_dir = os.path.join(f"test_results/{test_exp_name}/{exp.name}")
    else:
        output_dir = args.output_audio


    if test_record.get("data") or test_record.get("data_config"):
        if test_record["data"] != config.experiment.dataset.test.file_path:
            logger.info("Different dataset file path!")
            logger.info(f"Data path in config: {config.experiment.dataset.test.file_path}")
            logger.info(f"Data path in record: {test_record['data']}")
            raise Exception("Different dataset file path")
    else:
        test_record["data"] = config.experiment.dataset.test.file_path
        test_record["config"] = data_config
        test_record["output_dir"] = output_dir


    if args.skip_exist.upper() in ("Y", "YES", "T", "TRUE") and test_record.get(exp.name):
        # Test result on same model exist
        logger.info(f"Test result for experiment name {exp.name} exits. Abort this test.")
        exit()

    ###
    # Conduct testing
    ###
    try:
        test_result = tester(exp, testloader, logger, output_dir)
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
        "chkpt": args.chkpt,
        "metrics": test_result
    }

        
    test_record[exp.name] = test_result

    json_object = json.dumps(test_record, indent = 4)
    with open(test_record_path, "w") as f:
        f.write(json_object)
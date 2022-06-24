from datasets.dataloader import create_dataloader

from .inference import inference
from .metric_compute import metric_compute
from .asr_inference import asr_inference

def tester(config, logger, out_dir, asr_lang=None,*, skip_features=[]):

    exp = config.experiment
    result = {}

    if "audio" not in skip_features:
        logger.info("Start making test set and inferencing")
        testloader = create_dataloader(config, scheme="test")
        result.update(inference(exp, testloader, logger, out_dir))

    if asr_lang is not None and "asr" not in skip_features:
        logger.info("Start making test set (with only file path and wav for faster ASR computing)")
        testloader = create_dataloader(config, scheme="test", features=["asr"])
        result.update(asr_inference(exp, testloader, logger, out_dir, asr_lang))

    logger.info("Start making test set (with only file path for faster metrics computing)")
    testloader = create_dataloader(config, scheme="test", features=None)
    result.update(metric_compute(exp, testloader, logger, out_dir, skip_features=skip_features))

    return result

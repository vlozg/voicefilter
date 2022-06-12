import os
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.audio import Audio
from utils.hparams import HParam

from model.get_model import get_embedder, get_embedder_forward
from datasets.GenerateDataset import vad_merge
from datasets.dataloader import get_dataset

from tqdm import tqdm
import json


class CustomDataset(Dataset):
    def __init__(self, dataset, embedder, embedder_forward, device, prefix):
        self.dataset = dataset
        self.embedder_forward = lambda sample: embedder_forward(embedder, sample, device)
        self.prefix = prefix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Only get metadata first
        sample = self.dataset.data.iloc[idx].to_dict()
        path = str(Path(sample["embedding_utterance_path"]).parent) + "/" + self.prefix + Path(sample["embedding_utterance_path"]).stem + ".pt"
        if os.path.exists(path):
            return path

        # Start getting features and compute dvec
        sample = self.dataset[idx]
        if sample.get("dvec_tensor") is not None:
            print("Somehow we can get dvec tensor dataset, but can't find the pickle path based on our rule?")
            return
        sample["dvec_wav"] = [sample["dvec_wav"]]
        sample["dvec"] = [sample["dvec_mel"]]
        dvec = self.embedder_forward(sample)[0]

        # Change extension to pt and save
        
        torch.save(dvec, path)

        return path


def custom_collate_func(batch):
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/best.yaml',
                        help="folder contain yaml files for configuration")
    parser.add_argument('--use_cuda', type=str, default=None,
                    help="use cuda for dvec computing, overwrite config. Default: follow config file")
    parser.add_argument('-d', '--data_config', type=str,
                    help="yaml files for data used to test and testing configuration")
    parser.add_argument('--prefix', type=str, default=None,
                    help="add prefix to out file name. Default: not add prefix")
    parser.add_argument('--scheme', type=str, default=None,
                    help="add prefix to out file name. Default: not add prefix")
    args = parser.parse_args()

    ###
    # Merge config, only do merge for known key
    ###
    config = HParam(args.config)

    if args.use_cuda is not None:
        config.experiment.use_cuda = (args.use_cuda == "True")

    if args.data_config:
        data_config = HParam(args.data_config)
        config.experiment.dataset = data_config.experiment.dataset

    exp = config["experiment"]

    prefix = args.prefix if args.prefix else ""

    ###
    # Load model
    ###
    device = "cuda" if exp.use_cuda else "cpu"
    embedder = get_embedder(exp, train=False, device=device)
    embedder_forward = get_embedder_forward(exp)


    ###
    # Get dataset
    # Then create custom dataset and dataloader 
    # to conduct computing dvec in parallel
    ###
    dataset = get_dataset(config, scheme=args.scheme, features=["dvec_mel"])
    data_df = dataset.data
    # Replace data_df with drop duplicated one (since there is large duplicate in generated data)
    unique_df = data_df.drop_duplicates("embedding_utterance_path")
    dataset.data = unique_df

    dataloader = DataLoader(
        dataset=CustomDataset(dataset, embedder, embedder_forward, device, prefix),
        # batch_size=config.experiment.train.batch_size,
        batch_size=1,
        # batch_size=4,
        shuffle=False,
        # num_workers=config.experiment.train.num_workers,
        num_workers=0,
        # num_workers=4,
        collate_fn=custom_collate_func,
        sampler=None)

    
    # data_df["dvec_tensor"] = 
    dvec_tensor = []

    for dvec_paths in tqdm(dataloader):
        dvec_tensor += dvec_paths

    with open("tmp.txt", "w") as f:
        json_object = json.dumps(dvec_tensor, indent = 4)
        f.write(json_object)

    unique_df["dvec_tensor_path"] = dvec_tensor

    data_df = data_df.merge(unique_df[["embedding_utterance_path", "dvec_tensor_path"]], left_on='embedding_utterance_path', right_on='embedding_utterance_path')

    dataset_path = os.path.join(config.env.base_dir, config.experiment.dataset[args.scheme].file_path)
    data_df.to_csv(dataset_path, index=False)
import argparse
import os
from pathlib import Path

import pandas as pd

VCTK_TEST = 'VCTK/test_tuples.csv'
VCTK_TRAIN = 'VCTK/train_tuples.csv'
LIBRI_TEST = 'LibriSpeech/dev_tuples.csv'
LIBRI_TRAIN = 'LibriSpeech/train_tuples.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--basedir', type=str, default=".",
                        help="base directory, which should contain 2 folders VCTK and LibriSpeech from Google Speaker ID repo.")
    args = parser.parse_args()

    ###
    # Preprocess VCTK dataset path
    ###
    for data_path in [Path(args.basedir)/VCTK_TEST, Path(args.basedir)/VCTK_TRAIN]:
        df = pd.read_csv(data_path)
        for col in df.columns:
            if "_path" in col:
                continue
            tmp_split = df[col].str.split(pat="_", n=1, expand=True)
            df[col+"_path"] = "datasets/VCTK/wav48/" + tmp_split[0] + "/" + df[col] + ".wav"

        check_file_exist = df.apply(lambda x: Path(x["clean_utterance_path"]).is_file() & Path(x["embedding_utterance_path"]).is_file() & Path(x["interference_utterance_path"]).is_file(), axis=1)
        missing_df = df[~(check_file_exist)]
        if len(missing_df) > 0:
            print(Path(missing_df["clean_utterance_path"].iloc[0]))
            print(Path(missing_df["clean_utterance_path"].iloc[0]).is_file())
            print(Path(missing_df["embedding_utterance_path"].iloc[0]))
            print(Path(missing_df["embedding_utterance_path"].iloc[0]).is_file())
            print(Path(missing_df["interference_utterance_path"].iloc[0]))
            print(Path(missing_df["interference_utterance_path"].iloc[0]).is_file())
            raise Exception(f"There exist file(s) that are missing in {data_path}. Please check for them")
        
        df.to_csv(data_path, index=False)
    
    ###
    # Preprocess VCTK dataset path
    ###
    with open("datasets/LibriSpeech/SPEAKERS.TXT", "r") as f:
        speakers = f.readlines()[11:]
        speakers = [s.split("|", 4) for s in speakers]
        speakers = [[s.strip() for s in speaker] for speaker in speakers]
        speakers = {i: s for (i, _, s, _, _) in speakers}

    for data_path in [Path(args.basedir)/LIBRI_TEST, Path(args.basedir)/LIBRI_TRAIN]:
        df = pd.read_csv(data_path)
        for col in df.columns:
            if "_set" in col or "_path" in col:
                continue
            tmp_split = df[col].str.split(pat="-", n=2, expand=True)
            df[col+"_set"] = tmp_split[0].map(speakers)
            df[col+"_path"] =  'datasets/LibriSpeech/'+df[col+"_set"]+'/'+tmp_split[0]+'/'+tmp_split[1]+'/'+df[col]+'.flac'

        check_file_exist = df.apply(lambda x: Path(x["clean_utterance_path"]).is_file() & Path(x["embedding_utterance_path"]).is_file() & Path(x["interference_utterance_path"]).is_file(), axis=1)
        missing_df = df[~(check_file_exist)]
        if len(missing_df) > 0:
            print(missing_df)
            raise Exception(f"There exist file(s) that are missing in {data_path}. Please check for them")

        df.to_csv(data_path, index=False)

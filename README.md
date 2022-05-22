# VoiceFilter

## Dependencies

1. Python and packages

    This code was tested on Python 3.8 with PyTorch 1.10.0.
    Other packages can be installed by:

    ```bash
    pip install -r requirements.txt
    ```

## Prepare Dataset

1. Download LibriSpeech dataset

    Install axel first (`apt install axel`).

    Use axel to download datasets.

    ```bash
    axel -n 10 -a -c "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    axel -n 10 -a -c "https://www.openslr.org/resources/12/train-clean-360.tar.gz"
    axel -n 10 -a -c "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    axel -n 10 -a -c "https://www.openslr.org/resources/12/test-clean.tar.gz"
    ```

    Then, unzip `tar.gz` file to `datasets` folder:
    ```bash
    tar -xvzf train-clear-100.tar.gz
    tar -xvzf train-clear-360.tar.gz
    tar -xvzf dev-clean.tar.gz
    tar -xvzf test-clean.tar.gz
    ```

1. Edit `config.yaml`

    ```bash
    cd config
    cp default.yaml config.yaml
    ```

## Train VoiceFilter

1. Get pretrained model for speaker recognition system

    The model can be downloaded at [this GDrive link](https://drive.google.com/file/d/1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL/view?usp=sharing).

    Using gdown command for convenient download (gdown was installed via pip).
    ```bash
    gdown --id 1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL
    ```

1. Run

    After specifying `train_dir`, `test_dir` at `config.yaml`, run:
    ```bash
    python train.py -c [config.yaml] -e [path of embedder pt file] -m [name] --train_set [list of datasets used to generate train data] --test_set [list of datasets used to generate test data]
    ```
    This will create `chkpt/name` and `logs/name` at base directory(`-b` option, `.` in default)

    For reproducing the original experiment, use: the following bash command:
    ```bash
    python train.py -c config.yaml -e embedder.pt -m powlaw_loss --train_set librispeech-train --test_set librispeech-test
    ```

    Supported dataset include (for detail implementation, see the source code in `datasets/GenerateDataset.py`):
        `librispeech-train`, `librispeech-test`, `vctk`
        , `vin`, `voxceleb1-train`, `voxceleb1-test`, `voxceleb2-train`
        , `voxceleb1-test`, `zalo-train`, `zalo-test`

1. View tensorboardX

    ```bash
    tensorboard --logdir ./logs
    ```

1. Resuming from checkpoint

    ```bash
    python trainer.py -c [config yaml] --checkpoint_path [chkpt/name/chkpt_{step}.pt] -e [path of embedder pt file] -m name --train_set [list of datasets used to generate train data] --test_set [list of datasets used to generate test data]
    ```

    For example, finetune with VN dataset:
    ```bash
    python train.py -c config.yaml -e embedder.pt -m powlaw_loss_finetune --checkpoint_path chkpt/powlaw_loss/chkpt_168000.pt --train_set vin zalo-train --test_set zalo-test
    ```

## License

Apache License 2.0

This repository contains codes adapted/copied from the followings:
- [utils/adabound.py](./utils/adabound.py) from https://github.com/Luolc/AdaBound (Apache License 2.0)
- [utils/audio.py](./utils/audio.py) from https://github.com/keithito/tacotron (MIT License)
- [utils/hparams.py](./utils/hparams.py) from https://github.com/HarryVolek/PyTorch_Speaker_Verification (No License specified)
- [utils/normalize-resample.sh](./utils/normalize-resample.sh.) from https://unix.stackexchange.com/a/216475

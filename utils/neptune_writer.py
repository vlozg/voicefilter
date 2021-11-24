import numpy as np
import neptune.new as neptune
from getpass import getpass

from .plotting import plot_spectrogram_to_numpy


class NeptuneWriter:
    def __init__(self, hp, logdir):
        api_token = getpass('Enter your private Neptune API token: ')
        self.run = run = neptune.init(
            project="vulong61/voicefilter",
            api_token=api_token,)
        self.run["hp"] = hp

    def log_training(self, train_loss, step):
        self.run['train/loss'].log(train_loss)

    def log_evaluation(self, test_loss, sdr,
                       mixed_wav, target_wav, est_wav,
                       mixed_spec, target_spec, est_spec, est_mask,
                       step):
        
        self.run['test/loss'].log(test_loss)
        self.run['test/SDR'].log(sdr)

        self.add_audio('mixed_wav', mixed_wav, step, self.hp.audio.sample_rate)
        self.add_audio('target_wav', target_wav, step, self.hp.audio.sample_rate)
        self.add_audio('estimated_wav', est_wav, step, self.hp.audio.sample_rate)

        self.add_image('data/mixed_spectrogram',
            plot_spectrogram_to_numpy(mixed_spec), step, dataformats='HWC')
        self.add_image('data/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), step, dataformats='HWC')
        self.add_image('result/estimated_spectrogram',
            plot_spectrogram_to_numpy(est_spec), step, dataformats='HWC')
        self.add_image('result/estimated_mask',
            plot_spectrogram_to_numpy(est_mask), step, dataformats='HWC')
        self.add_image('result/estimation_error_sq',
            plot_spectrogram_to_numpy(np.square(est_spec - target_spec)), step, dataformats='HWC')
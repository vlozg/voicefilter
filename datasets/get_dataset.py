import os
import glob

from .GenerateDataset import VFDataset, generate_dataset_df
from .GGSpeakerIDDataset import VFGGDataset

def get_dataset(config, scheme):
    dataset_config = config.experiment.dataset[scheme]

    # Genearate dataset
    if config.experiment.dataset.name == "generate":
        
        speaker_folders=[]
    
        # Get all file paths
        dataset_detail = dataset_config.detail

        if "librispeech-train" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.libri_dir, 'train-clean-100', '*'))
                            if os.path.isdir(x)] + \
                            [x for x in glob.glob(os.path.join(config.env.data.libri_dir, 'train-clean-360', '*'))
                            if os.path.isdir(x)]

        if "librispeech-test" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.libri_dir, 'dev-clean', '*'))] + \
                            [x for x in glob.glob(os.path.join(config.env.data.libri_dir, 'test-clean', '*'))]

        if "vctk" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.vctk_dir, 'wav48', '*')) if os.path.isdir(x)]
        
        if "vivos-train" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.vivos_dir, 'train/waves', '*')) if os.path.isdir(x)]
        if "vivos-test" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.vivos_dir, 'test/waves', '*')) if os.path.isdir(x)]

        if "voxceleb1-train" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.voxceleb1_dir, 'dev/wav', '*')) if os.path.isdir(x)]
        if "voxceleb1-test" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.voxceleb1_dir, 'test/wav', '*')) if os.path.isdir(x)]

        if "voxceleb2-train" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.voxceleb2_dir, 'dev/aac', '*')) if os.path.isdir(x)]
        if "voxceleb1-test" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.voxceleb2_dir, 'aac', '*')) if os.path.isdir(x)]

        if "zalo-train" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.zalo_dir, 'dataset', '*')) if os.path.isdir(x)]
        if "zalo-test" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.zalo_dir, 'private-test', '*')) if os.path.isdir(x)]

        if "vin" in dataset_detail:
            speaker_folders += [x for x in glob.glob(os.path.join(config.env.data.vin_dir, 'speakers', '*')) if os.path.isdir(x)]

        # Get all audio paths & filter only speaker folder that contain more then 2 samples
        speaker_sets = [glob.glob(os.path.join(spk, '**', "*.flac"), recursive=True) + \
            glob.glob(os.path.join(spk, '**', "*.wav"), recursive=True)
                        for spk in speaker_folders]
        speaker_sets = [x for x in speaker_sets if len(x) >= 2]
        
        # Randomly generate dataset recipe if not exist
        dataset_path = os.path.join(config.env.base_dir, dataset_config.file_path)
        if not os.path.exists(dataset_path):
            dataset_df = generate_dataset_df(config.experiment, dataset_config, speakers=speaker_sets)
            dataset_df.to_csv(dataset_path,index=False)

        return VFDataset(config.experiment, dataset_path=dataset_path)

    elif config.experiment.dataset.name == "gg":
        return VFGGDataset(config.experiment, dataset=dataset_config, base_dir=".")
    
    else:
        raise NotImplementedError(f"Dataset scheme {scheme} is not implemented")
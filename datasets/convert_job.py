from pathlib import Path
import tqdm
from multiprocessing import Pool, cpu_count
from pydub import AudioSegment
import argparse

############ Convert audio format job ############
def m4a_2_wav(p):
        track = AudioSegment.from_file(p,  format= 'm4a')
        track.export(p.with_suffix(".wav"), format='wav')
        p.unlink(missing_ok=True) # Remove m4a file

def job_m4a2wav(d):
    m4a_list=list(Path(d).rglob("*.m4a"))
    cpu_num = cpu_count()
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(m4a_2_wav, m4a_list), total=len(m4a_list)))

########### Resample job ############
def resample(p, sr=16000):
        if (p.name[:3] == "16k"):
            return
        track = AudioSegment.from_file(p).set_frame_rate(sr)
        new_path = Path.joinpath(p.parent, "16k_" + p.with_suffix(".wav").name)
        track.export(new_path, format='wav')
        p.unlink(missing_ok=True) # Remove old file

def job_to16khz(d):
    wav_list=list(Path(d).rglob("*.wav"))
    cpu_num = cpu_count()
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(resample, wav_list), total=len(wav_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job', type=str, required=True,
                        help="Job name.")
    parser.add_argument('-d', '--directory', type=str, required=True,
                        help="Base directory to do the job.")
    args = parser.parse_args()

    if args.job == "m4a2wav":
        job_m4a2wav(args.directory)
    elif args.job == "to16khz":
        job_to16khz(args.directory)
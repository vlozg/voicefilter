import io
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile
import torch
from google.cloud import speech
from torchmetrics.functional import word_error_rate
from tqdm import tqdm


def transcribe_file(speech_file, label=None, out_file=None, lang="en-US", sr=16000):
    """Transcribe the given audio file."""

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
        language_code=lang,
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    asr = {"transcript": [], "confidence": []}
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        asr["confidence"].append(result.alternatives[0].confidence)
        asr["transcript"].append(result.alternatives[0].transcript)

    asr["transcript"] = " ".join(asr["transcript"])
    if len(asr["confidence"]) > 0:
        asr["confidence"] = np.mean(asr["confidence"])
    else:
        asr["confidence"] = float("nan")

    if label is not None:
        wer = word_error_rate(asr["transcript"].upper(), label).item()
    else:
        wer = float("nan")

    asr["wer"] = wer

    if out_file is not None:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(asr, indent = 4, ensure_ascii=False))

    return asr, wer


def asr_inference(config, testloader, logger, out_dir=None, lang="en-US"):

    os.makedirs(out_dir if out_dir != '' else ".", exist_ok=True)
    
    wers = {}

    with ProcessPoolExecutor() as executor:
        futu_asr_w = {}

        for batch in tqdm(testloader):
            if batch.get("target_text") is None:
                batch["target_text"] = [None] * len(batch["index"])

            for idx, target_text in zip(batch["index"], batch["target_text"]):
                out_file = os.path.join(out_dir, f"{idx}_asr.json")
                # ASR result exist, skip
                if os.path.isfile(out_file):
                    with open(out_file, "r") as f:
                        asr = json.load(f)
                        wers[idx] = asr["wer"]

                    continue

                est_path = os.path.join(out_dir, f"{idx}.wav")

                futu_asr_w[executor.submit(transcribe_file, est_path, target_text, out_file, lang, config.audio.sample_rate)] = idx


        logger.info("Waiting for ASR computing and writing...")
        for f in tqdm(as_completed(futu_asr_w), total=len(futu_asr_w)):
            idx = futu_asr_w[f]
            try:
                wers[idx] = f.result()[1]
            except Exception as exc:
                print('%r generated an exception: %s' % (idx, exc))

    wers = [wers[k] for k in sorted(wers.keys())]
    wers = np.array(wers)

    logger.info(f"Complete ASR inference")

    results = {
        "wer": wers,
    }

    return results
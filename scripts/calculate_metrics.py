import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
from pyctcdecode import build_ctcdecoder
from pyroomacoustics.experimental.rt60 import measure_rt60
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
)
from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.text import CharErrorRate, WordErrorRate

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
SAMPLE_RATE = 16000
DECAY_DB = 20

ASR_MODEL = nemo_asr.models.EncDecCTCModel.restore_from(
    ROOT_PATH / "data" / "QuartzNet5x5LS-En.nemo"
)
METRICS = {
    "pesq": PerceptualEvaluationSpeechQuality(SAMPLE_RATE, mode="wb"),
    "stoi": ShortTimeObjectiveIntelligibility(SAMPLE_RATE),
    "cer": CharErrorRate(),
    "wer": WordErrorRate(),
    "srmr": SpeechReverberationModulationEnergyRatio(SAMPLE_RATE),
    "sdr": SignalDistortionRatio(),
    "si-sdr": ScaleInvariantSignalDistortionRatio(),
    "t60": partial(measure_rt60, fs=SAMPLE_RATE, decay_db=DECAY_DB, plot=False),
}


def transcribe(audio):
    ASR_MODEL.eval()

    # input_signal = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)

    asr_vocab = ASR_MODEL.decoder.vocabulary
    decoder = build_ctcdecoder(asr_vocab)

    with torch.no_grad():
        preds = ASR_MODEL(
            input_signal=audio, input_signal_length=torch.tensor([audio.shape[-1]])
        )[0]

    preds = torch.nn.functional.log_softmax(preds, dim=-1)[0].numpy()

    return decoder.decode(preds, beam_width=1)  # beam_width = 1 is equivalent to argmax


def get_single_metrics(data):
    speech = torch.from_numpy(data["speech"]).unsqueeze(0).to(torch.float32)
    rir = data["rir"]
    dereverb_rir = data["dereverb_rir"]
    reverb_speech = (
        torch.from_numpy(data["reverb_speech"]).unsqueeze(0).to(torch.float32)
    )
    dereverb_speech = (
        torch.from_numpy(data["dereverb_speech"]).unsqueeze(0).to(torch.float32)
    )

    # speech_text = transcribe(speech)
    speech_text = data["text"]
    dereverb_text = transcribe(dereverb_speech)
    reverb_text = transcribe(reverb_speech)

    metrics = {}
    for metric_name, calculator in METRICS.items():
        if metric_name in ["srmr"]:
            metric_value = calculator(dereverb_speech) - calculator(reverb_speech)

        if metric_name in ["sdr", "si-sdr", "pesq", "stoi"]:
            dereverb_value = calculator(dereverb_speech, speech)
            reverb_value = calculator(reverb_speech, speech)
            metric_value = dereverb_value - reverb_value

            if metric_name in ["sdr", "si-sdr"]:
                print("SDR", dereverb_value)
                print("SDR", reverb_value)

        if metric_name in ["cer", "wer"]:
            dereverb_value = calculator(dereverb_text, speech_text)
            reverb_value = calculator(reverb_text, speech_text)
            print("text", dereverb_value, reverb_value)
            print(dereverb_text)
            print(speech_text)
            metric_value = dereverb_value - reverb_value

        if metric_name == "t60":
            metric_value = calculator(dereverb_rir) - calculator(rir)

        metrics[metric_name] = metric_value

    return metrics


def calculate_metrics(dataset_name, algorithm_name):
    data_path = ROOT_PATH / "data" / "dereverberated"
    assert data_path.exists(), "dir not found, evaluate algorithm on dataset"

    save_dir = f"{dataset_name}_{algorithm_name}"

    result_metrics = defaultdict(float)
    amount = 0

    for file in os.listdir(str(data_path / save_dir)):
        data = torch.load(data_path / save_dir / file)
        metrics = get_single_metrics(data)

        for k, v in metrics.items():
            result_metrics[k] += v

        amount += 1

    for k, v in result_metrics.items():
        result_metrics[k] = v / amount
        print(f"Metric: {k},\tValue: {result_metrics[k]}")

    torch.save(
        result_metrics, data_path / f"{dataset_name}_{algorithm_name}_metrics.pth"
    )


if __name__ == "__main__":
    args = ArgumentParser(description="Calculate all metrics on a given saved dataset")

    args.add_argument(
        "-d",
        "--dataset_name",
        default=None,
        type=str,
        help="Dataset name inside data dir (default: None)",
    )

    args.add_argument(
        "-a",
        "--algorithm_name",
        default=None,
        type=str,
        help="Algorithm Name (default: None)",
    )

    args = args.parse_args()

    calculate_metrics(args.dataset_name, args.algorithm_name)

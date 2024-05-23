import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
from pyctcdecode import build_ctcdecoder
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
)
from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.text import CharErrorRate, WordErrorRate

# from pyroomacoustics.experimental.rt60 import measure_rt60
from scripts.measure_t60 import measure_rt60

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
SAMPLE_RATE = 16000
DECAY_DB = 60

ASR_MODEL = nemo_asr.models.EncDecCTCModel.restore_from(
    ROOT_PATH / "data" / "QuartzNet5x5LS-En.nemo"
)
ASR_MODEL_STRONG = nemo_asr.models.EncDecCTCModel.from_pretrained(
    "QuartzNet15x5Base-En"
)
METRICS = {
    "pesq": PerceptualEvaluationSpeechQuality(SAMPLE_RATE, mode="wb"),
    "stoi": ShortTimeObjectiveIntelligibility(SAMPLE_RATE),
    "cer": CharErrorRate(),
    "wer": WordErrorRate(),
    "cer_strong": CharErrorRate(),
    "wer_strong": WordErrorRate(),
    "cer_reverb": CharErrorRate(),
    "wer_reverb": WordErrorRate(),
    "cer_dereverb": CharErrorRate(),
    "wer_dereverb": WordErrorRate(),
    "cer_reverb_strong": CharErrorRate(),
    "wer_reverb_strong": WordErrorRate(),
    "cer_dereverb_strong": CharErrorRate(),
    "wer_dereverb_strong": WordErrorRate(),
    "srmr": SpeechReverberationModulationEnergyRatio(SAMPLE_RATE),
    "sdr": SignalDistortionRatio(),
    "si-sdr": ScaleInvariantSignalDistortionRatio(),
    "t60": partial(measure_rt60, fs=SAMPLE_RATE, decay_db=DECAY_DB, plot=False),
}


def transcribe(audio, model=ASR_MODEL):
    model.eval()

    # input_signal = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)

    asr_vocab = model.decoder.vocabulary
    decoder = build_ctcdecoder(asr_vocab)

    with torch.no_grad():
        preds = model(
            input_signal=audio, input_signal_length=torch.tensor([audio.shape[-1]])
        )[0]

    preds = torch.nn.functional.log_softmax(preds, dim=-1)[0].numpy()

    return decoder.decode(preds, beam_width=1)  # beam_width = 1 is equivalent to argmax


def get_single_metrics(data):
    speech = torch.from_numpy(data["speech"]).unsqueeze(0).to(torch.float32)
    rir = data.get("rir")
    dereverb_rir = data.get("dereverb_rir")
    reverb_speech = (
        torch.from_numpy(data["reverb_speech"]).unsqueeze(0).to(torch.float32)
    )
    dereverb_speech = (
        torch.from_numpy(data["dereverb_speech"]).unsqueeze(0).to(torch.float32)
    )

    reverb_min_length = min(speech.shape[-1], reverb_speech.shape[-1])
    dereverb_min_length = min(speech.shape[-1], dereverb_speech.shape[-1])

    # speech_text = transcribe(speech)
    speech_text = data["text"]
    dereverb_text = transcribe(dereverb_speech, model=ASR_MODEL)
    reverb_text = transcribe(reverb_speech, model=ASR_MODEL)

    dereverb_text_strong = transcribe(dereverb_speech, model=ASR_MODEL_STRONG)
    reverb_text_strong = transcribe(reverb_speech, model=ASR_MODEL_STRONG)

    metrics = {}
    for metric_name, calculator in METRICS.items():
        if metric_name in ["srmr"]:
            metric_value = calculator(dereverb_speech) - calculator(reverb_speech)

        if metric_name in ["sdr", "si-sdr", "pesq", "stoi"]:
            dereverb_value = calculator(
                dereverb_speech[:, :dereverb_min_length],
                speech[:, :dereverb_min_length],
            )
            reverb_value = calculator(
                reverb_speech[:, :reverb_min_length], speech[:, :reverb_min_length]
            )
            metric_value = dereverb_value - reverb_value

            if metric_name in ["sdr", "si-sdr"]:
                print("SDR", dereverb_value)
                print("SDR", reverb_value)

        if metric_name in ["cer", "wer"]:
            dereverb_value = calculator(dereverb_text, speech_text)
            reverb_value = calculator(reverb_text, speech_text)
            print("text", dereverb_value, reverb_value)
            print("dereverb text", dereverb_text)
            print("reverb text", reverb_text)
            print("speech text", speech_text)
            metric_value = dereverb_value - reverb_value

        if metric_name in ["cer_reverb", "wer_reverb"]:
            metric_value = calculator(reverb_text, speech_text)

        if metric_name in ["cer_dereverb", "wer_dereverb"]:
            metric_value = calculator(dereverb_text, speech_text)

        if metric_name in ["cer_reverb_strong", "wer_reverb_strong"]:
            metric_value = calculator(reverb_text_strong, speech_text)

        if metric_name in ["cer_dereverb_strong", "wer_dereverb_strong"]:
            metric_value = calculator(dereverb_text_strong, speech_text)

        if metric_name in ["cer_strong", "wer_strong"]:
            dereverb_value = calculator(dereverb_text_strong, speech_text)
            reverb_value = calculator(reverb_text_strong, speech_text)
            print("text_strong", dereverb_value, reverb_value)
            print("dereverb text strong", dereverb_text_strong)
            print("reverb text strong", reverb_text_strong)
            print("speech text strong", speech_text)
            metric_value = dereverb_value - reverb_value

        if metric_name == "t60":
            if dereverb_rir is None:
                metric_value = 0
            else:
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

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import functional as F
from torchaudio.utils import download_asset

from src.utils import ROOT_PATH, normalize, read_json, write_json


class ExampleDataset(Dataset):
    def __init__(self, sr=16000):
        super().__init__()

        data_path = ROOT_PATH / "data" / "ExampleDataset"
        data_path.mkdir(exist_ok=True, parents=True)

        self.sr = sr

        self.index = self.load_index(data_path)

    def __len__(self):
        return len(self.index)

    def load_index(self, data_path):
        index_path = data_path / "index.json"

        if index_path.exists():
            return read_json(index_path)
        else:
            return self.create_index(index_path)

    def create_index(self, index_path):
        rir_path = download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav"
        )
        speech_path = download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav"
        )

        text = "i had that curiosity beside me at this moment"
        text = normalize(text)

        index = [
            {
                "rir_path": rir_path,
                "speech_path": speech_path,
                "text": text,
            }
        ]

        write_json(index, index_path)

        return index

    def __getitem__(self, i):
        data = self.index[i]
        speech_path = data["speech_path"]
        rir_path = data["rir_path"]
        text = data["text"]

        rir, rir_sr = torchaudio.load(rir_path)
        rir = rir[:, int(rir_sr * 1.01) : int(rir_sr * 1.3)]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        rir = torchaudio.transforms.Resample(rir_sr, self.sr)(rir)

        speech, speech_sr = torchaudio.load(speech_path)
        speech = torchaudio.transforms.Resample(speech_sr, self.sr)(speech)

        # take only start, like lfilter in ss
        reverb_speech = F.fftconvolve(speech, rir)[:, : speech.shape[-1]]

        rir = rir.to(torch.float64).numpy().sum(axis=0)
        speech = speech.to(torch.float64).numpy().sum(axis=0)
        reverb_speech = reverb_speech.to(torch.float64).numpy().sum(axis=0)

        return {
            "speech": speech,
            "rir": rir,
            "reverb_speech": reverb_speech,
            "text": text,
        }

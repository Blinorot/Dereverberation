import os
import shutil

import gdown
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import functional as F
from torchaudio.utils import download_asset

from src.utils import ROOT_PATH, normalize, read_json, write_json


class SynthesizedDataset(Dataset):
    def __init__(self, sr=16000):
        super().__init__()

        self.data_path = ROOT_PATH / "data" / "SynthesizedDataset"
        if not self.data_path.exists():
            arc_path = ROOT_PATH / "data" / "SynthesizedDataset.zip"
            gdown.download(id="122FMJ9iGjoLLuoHJGBN7E20w8gDr1WDe", output=str(arc_path))
            shutil.unpack_archive(arc_path, self.data_path)

        self.sr = sr

        self.index = self.load_index()

    def __len__(self):
        return len(self.index)

    def load_index(self):
        index_path = self.data_path / "index.json"

        if index_path.exists():
            return read_json(index_path)
        else:
            return self.create_index(index_path)

    def create_index(self, index_path):
        index = []

        for speech_name in os.listdir(self.data_path / "speech"):
            speech_path = self.data_path / "speech" / speech_name
            rir_path = self.data_path / "rir" / speech_name
            text_path = self.data_path / "text" / f"{speech_name[:-4]}.txt"

            with open(text_path, "r") as f:
                text = normalize(f.read())

            index.append(
                {
                    "rir_path": str(rir_path),
                    "speech_path": str(speech_path),
                    "text": text,
                }
            )

            print(speech_name, rir_path, speech_path)

        write_json(index, index_path)

        return index

    def __getitem__(self, i):
        data = self.index[i]
        speech_path = data["speech_path"]
        rir_path = data["rir_path"]
        text = data["text"]

        rir, rir_sr = torchaudio.load(rir_path)
        # rir = rir[:, int(rir_sr * 1.01) : int(rir_sr * 1.3)]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        rir = torchaudio.transforms.Resample(rir_sr, self.sr)(rir)

        speech, speech_sr = torchaudio.load(speech_path)
        speech = torchaudio.transforms.Resample(speech_sr, self.sr)(speech)

        reverb_speech = F.fftconvolve(speech, rir)

        rir = rir.to(torch.float64).numpy().sum(axis=0)
        speech = speech.to(torch.float64).numpy().sum(axis=0)
        reverb_speech = reverb_speech.to(torch.float64).numpy().sum(axis=0)
        reverb_speech = reverb_speech / np.abs(reverb_speech).max()

        return {
            "speech": speech,
            "rir": rir,
            "reverb_speech": reverb_speech,
            "text": text,
            "speech_path": speech_path,
            "rir_path": rir_path,
        }

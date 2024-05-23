import argparse
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import torchaudio
import wget
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
ROOT_DATA_PATH = ROOT_PATH / "data"

URL_LINKS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "rirs_noises": "https://www.openslr.org/resources/28/rirs_noises.zip",
}


def download_dataset(data_path):
    if not (data_path / "LibriSpeech").exists():
        arch_path = data_path / "test-clean.tar.gz"
        print("Loading test-clean")
        wget.download(URL_LINKS["test-clean"], str(arch_path))
        shutil.unpack_archive(arch_path, data_path)
        os.remove(str(arch_path))

    if not (data_path / "RIRS_NOISES").exists():
        arch_path = data_path / "rirs_noises.zip"
        print("Loading rir_noises")
        wget.download(URL_LINKS["rirs_noises"], str(arch_path))
        shutil.unpack_archive(arch_path, data_path)
        os.remove(str(arch_path))


def download_and_prepare(N, max_duration):
    data_path = ROOT_DATA_PATH / "RawSynthesizedDataset"
    data_path.mkdir(exist_ok=True, parents=True)

    download_dataset(data_path)

    # get all rir_files
    rir_protocol_path = (
        data_path / "RIRS_NOISES" / "real_rirs_isotropic_noises" / "rir_list"
    )
    with rir_protocol_path.open("r") as f:
        rir_lines = f.readlines()
    rir_files = [line.split()[4] for line in rir_lines if int(line.split()[1]) <= 24]
    rir_files = [str(data_path / elem) for elem in rir_files]

    # get all utterances
    speech_path = data_path / "LibriSpeech" / "test-clean"

    trans_files = []
    for filename in glob.glob(str(speech_path) + "/*/*/*.txt"):
        trans_files.append(filename)

    all_speech_files = []
    all_texts = []
    for trans_file in trans_files:
        with open(trans_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            speech_name = line.split()[0]
            text = " ".join(line.split()[1:]).strip()

            filename = Path(trans_file).parent / f"{speech_name}.flac"
            all_speech_files.append(filename)
            all_texts.append(text)

    speech_files = []
    texts = []
    for i in range(len(all_speech_files)):
        speech = all_speech_files[i]
        info = torchaudio.info(speech)
        length = info.num_frames / info.sample_rate
        if length <= max_duration:
            speech_files.append(all_speech_files[i])
            texts.append(all_texts[i])

    print(f"Filtered {100 * len(texts) / len(all_texts)} % of the dataset")

    # create prepared dataset
    data_path = ROOT_DATA_PATH / "SynthesizedDataset"
    speech_path = data_path / "speech"
    rir_path = data_path / "rir"
    text_path = data_path / "text"
    speech_path.mkdir(exist_ok=True, parents=True)
    rir_path.mkdir(exist_ok=True, parents=True)
    text_path.mkdir(exist_ok=True, parents=True)

    # we include all rir files
    # to have different kind of rooms
    dataset_length = N * len(rir_files)

    np.random.seed(1)
    dataset_speech_files = np.random.choice(
        len(speech_files), size=dataset_length, replace=False
    )

    print("Preparing Synthesized Dataset")
    for i in tqdm(range(dataset_length)):
        speech = speech_files[dataset_speech_files[i]]
        rir = rir_files[i % len(rir_files)]
        text = texts[dataset_speech_files[i]]

        rir_type = rir.split("/")[-1].split("_")[3]

        signal, sr = torchaudio.load(speech)
        torchaudio.save(
            str(speech_path / f"{i:05}_{rir_type}.wav"), signal, sample_rate=sr
        )

        signal, sr = torchaudio.load(rir)
        torchaudio.save(
            str(rir_path / f"{i:05}_{rir_type}.wav"), signal, sample_rate=sr
        )

        # shutil.copy(speech, speech_path / f"{i:05}_{rir_type}.wav")
        # shutil.copy(rir, rir_path / f"{i:05}_{rir_type}.wav")

        with open(text_path / f"{i:05}_{rir_type}.txt", "w") as f:
            f.write(text)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Prepare Synthesized Dataset")
    args.add_argument(
        "-N",
        type=int,
        default=4,
        help="Number of elements in the dataset per each reverberation (default 4)",
    )
    args.add_argument(
        "-s",
        type=float,
        default=5.0,
        help="Max duration of clean speech in seconds (default 5.0)",
    )
    args = args.parse_args()

    download_and_prepare(N=args.N, max_duration=args.s)

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


def download_and_prepare(N, but_dataset_path, max_duration):
    data_path = ROOT_DATA_PATH / "RawRealDataset"
    data_path.mkdir(exist_ok=True, parents=True)

    download_dataset(data_path)

    # get all room_paths
    np.random.seed(1)
    rooms = ["D105", "L207", "L212", "L227", "Q301"]
    rir_room_paths = []
    room_paths = []
    for room in rooms:
        room_path = Path(but_dataset_path) / f"VUT_FIT_{room}" / "MicID01"
        rir_path = Path(but_dataset_path) / "RIR" / f"VUT_FIT_{room}" / "MicID01"
        spk_ids = [elem for elem in os.listdir(room_path)]
        spk_id = spk_ids[np.random.randint(0, high=len(spk_ids))]
        room_path = room_path / spk_id
        rir_path = rir_path / spk_id
        mic_ids = [elem for elem in os.listdir(room_path)]

        mic_ids_ids = np.random.choice(len(mic_ids), size=N, replace=False)
        for id in mic_ids_ids:
            mic_id = mic_ids[id]
            room_paths.append(room_path / mic_id / "english")
            rir_room_paths.append(rir_path / mic_id)

            # print(room_path / mic_id / "english")
            # print(rir_path / mic_id)

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
    data_path = ROOT_DATA_PATH / "RealDataset"
    speech_path = data_path / "speech"
    reverb_speech_path = data_path / "reverb_speech"
    rir_path = data_path / "rir"
    text_path = data_path / "text"
    speech_path.mkdir(exist_ok=True, parents=True)
    reverb_speech_path.mkdir(exist_ok=True, parents=True)
    rir_path.mkdir(exist_ok=True, parents=True)
    text_path.mkdir(exist_ok=True, parents=True)

    # we include all rir files
    # to have different kind of rooms
    dataset_length = len(room_paths)

    np.random.seed(1)
    dataset_speech_files = np.random.choice(
        len(speech_files), size=dataset_length, replace=False
    )

    print("Preparing Real Dataset")
    for i in tqdm(range(dataset_length)):
        speech = speech_files[dataset_speech_files[i]]

        room_path = room_paths[i]
        reverb_speech = room_path / "/".join(str(speech).split("/")[-5:])
        reverb_speech = str(reverb_speech)[:-5] + ".v00.wav"

        rir_room_path = rir_room_paths[i]
        rir = rir_room_path / "RIR" / "IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav"

        text = texts[dataset_speech_files[i]]

        room_type = str(room_path).split("/")[-5].split("_")[-1]

        # print(room_path)
        # print(rir_room_path)

        signal, sr = torchaudio.load(speech)
        torchaudio.save(
            str(speech_path / f"{i:05}_{room_type}.wav"), signal, sample_rate=sr
        )

        signal, sr = torchaudio.load(reverb_speech)
        torchaudio.save(
            str(reverb_speech_path / f"{i:05}_{room_type}.wav"), signal, sample_rate=sr
        )

        signal, sr = torchaudio.load(rir)
        torchaudio.save(
            str(rir_path / f"{i:05}_{room_type}.wav"), signal, sample_rate=sr
        )

        # shutil.copy(speech, speech_path / f"{i:05}_{room_type}.wav")
        # shutil.copy(reverb_speech, reverb_speech_path / f"{i:05}_{room_type}.wav")

        with open(text_path / f"{i:05}_{room_type}.txt", "w") as f:
            f.write(text)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Prepare Synthesized Dataset")
    args.add_argument(
        "-N",
        type=int,
        default=20,
        help="Number of elements in the dataset per each reverberation (default 20)",
    )
    args.add_argument(
        "-d", type=str, default=None, help="Path to the BUT dataset (default: None)"
    )
    args.add_argument(
        "-s",
        type=float,
        default=5.0,
        help="Max duration of clean speech in seconds (default 5.0)",
    )
    args = args.parse_args()

    download_and_prepare(N=args.N, but_dataset_path=args.d, max_duration=args.s)

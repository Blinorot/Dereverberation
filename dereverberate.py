from argparse import ArgumentParser

import numpy as np
import scipy.signal as ss
import torch
from tqdm.auto import tqdm

import src.datasets
from src.HERB.algorithm import dereverberate
from Dereverberation.src.LP.algorithm import LP_dereverberation
from src.utils import ROOT_PATH


def main(dataset_name, algorithm_name):
    dataset_class = getattr(src.datasets, dataset_name)
    dataset = dataset_class()

    data_path = (
        ROOT_PATH / "data" / "dereverberated" / f"{dataset_name}_{algorithm_name}"
    )
    data_path.mkdir(exist_ok=True, parents=True)

    if algorithm_name == "HERB":
        algorithm = dereverberate
    elif algorithm_name == "LP":
        algorithm = LP_dereverberation
    else:
        raise NotImplementedError()

    for i in tqdm(range(len(dataset))):
        data = dataset[i]

        print(data["speech_path"])

        dereverb_speech, inverse_filter = algorithm(data["reverb_speech"])
        dereverb_speech = dereverb_speech / np.abs(dereverb_speech).max()

        if data.get("rir") is not None:
            dereverb_rir = ss.lfilter(inverse_filter, [1], data["rir"])
            data["dereverb_rir"] = dereverb_rir

        data["dereverb_speech"] = dereverb_speech

        torch.save(data, data_path / f"{i:03}.pth")


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

    main(args.dataset_name, args.algorithm_name)

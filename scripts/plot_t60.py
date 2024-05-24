import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.interpolate import interp1d

# from pyroomacoustics.experimental.rt60 import measure_rt60
from scripts.measure_t60 import measure_rt60

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
SAMPLE_RATE = 16000
DECAY_DB = 60

sns.set_style("whitegrid")

METRICS = {
    "t60": partial(measure_rt60, fs=SAMPLE_RATE, decay_db=DECAY_DB, plot=False),
}

DEREVERB_COLOR = "#d7191c"
RIR_COLOR = "#2b83ba"


def get_t60_curve(signal):
    fs, i_5db, energy, energy_db, power = METRICS["t60"](signal, plot=True)

    # Remove clip power below to minimum energy (for plotting purpose mostly)
    energy_min = energy[-1]
    # energy_db_min = energy_db[-1]
    power[power < energy[-1]] = energy_min
    power_db = 10 * np.log10(power)
    power_db -= np.max(power_db)

    # time vector
    def get_time(x, fs):
        return np.arange(x.shape[0]) / fs - i_5db / fs

    # plot power and energy
    time_curve = get_time(energy_db, fs)

    energy_curve = energy_db
    return time_curve, energy_curve


def get_single_curves(data, rir_info, dereverb_info):
    rir = data.get("rir")
    dereverb_rir = data.get("dereverb_rir")

    # rir
    time_curve, energy_curve = get_t60_curve(rir)
    rir_info["time"].append(time_curve)
    rir_info["energy"].append(energy_curve)

    # dereverb rir
    time_curve, energy_curve = get_t60_curve(dereverb_rir)
    dereverb_info["time"].append(time_curve)
    dereverb_info["energy"].append(energy_curve)

    return rir_info, dereverb_info


def get_data(dataset_name, algorithm_name):
    data_path = ROOT_PATH / "data" / "dereverberated"
    assert data_path.exists(), "dir not found, evaluate algorithm on dataset"

    save_dir = f"{dataset_name}_{algorithm_name}"

    rir_info = defaultdict(list)
    dereverb_info = defaultdict(list)
    amount = 0

    for file in os.listdir(str(data_path / save_dir)):
        data = torch.load(data_path / save_dir / file)
        rir_info, dereverb_info = get_single_curves(data, rir_info, dereverb_info)

        amount += 1

        # if amount >= 2:
        #     break

    return rir_info, dereverb_info, amount


def interpolate_t60(rir_info, dereverb_info, amount):
    time_min = 100000000000
    time_max = 0

    # rir
    for i in range(amount):
        rir_time = rir_info["time"][i]
        rir_energy = rir_info["energy"][i]

        values = (rir_energy[0], rir_energy[-1])

        rir_interp = interp1d(
            rir_time, rir_energy, bounds_error=False, fill_value=values
        )
        rir_info["interp"].append(rir_interp)

        dereverb_time = dereverb_info["time"][i]
        dereverb_energy = dereverb_info["energy"][i]

        values = (dereverb_energy[0], dereverb_energy[-1])

        dereverb_interp = interp1d(
            dereverb_time, dereverb_energy, bounds_error=False, fill_value=values
        )
        dereverb_info["interp"].append(dereverb_interp)

        time_min = min(time_min, rir_time[0], dereverb_time[0])
        time_max = max(time_max, rir_time[-1], dereverb_time[-1])

    time_min = 0

    time_curve = np.linspace(time_min, time_max + 0.01, num=100000)

    rir_curves = np.zeros((amount, len(time_curve)))
    dereverb_curves = np.zeros((amount, len(time_curve)))

    for i in range(amount):
        rir_curves[i, :] = rir_info["interp"][i](time_curve)
        dereverb_curves[i, :] = dereverb_info["interp"][i](time_curve)

    return time_curve, rir_curves, dereverb_curves


def plot_t60(
    axes, time_curve, rir_curves, dereverb_curves, dataset_name, algorithm_name
):
    max_rir = rir_curves.max(axis=0)
    min_rir = rir_curves.min(axis=0)
    mean_rir = rir_curves.mean(axis=0)

    max_dereverb = dereverb_curves.max(axis=0)
    min_dereverb = dereverb_curves.min(axis=0)
    mean_dereverb = dereverb_curves.mean(axis=0)

    print(time_curve)

    axes.fill_between(time_curve, max_rir, min_rir, color=RIR_COLOR, alpha=0.2)
    axes.fill_between(
        time_curve, max_dereverb, min_dereverb, color=DEREVERB_COLOR, alpha=0.2
    )

    axes.plot(time_curve, mean_rir, color=RIR_COLOR, label="RIR", linewidth=2)
    axes.plot(
        time_curve,
        mean_dereverb,
        color=DEREVERB_COLOR,
        label="DereverbRIR",
        linewidth=2,
    )

    axes.hlines(
        -60,
        time_curve[0],
        time_curve[-1],
        color="black",
        linestyle="--",
        label="-60db",
        linewidth=2,
    )
    # plt.hlines(-5, time_curve[0], time_curve[-1], color="black", linestyle="--")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("db")

    axes.legend()

    axes.set_title(f"{dataset_name} -- {algorithm_name}")


def get_data_and_plot_t60(dataset_name, algorithm_name):
    save_path = ROOT_PATH / "data" / "plots"
    save_path.mkdir(exist_ok=True, parents=True)
    save_path = save_path / f"{dataset_name}_{algorithm_name}.pdf"

    rir_info, dereverb_info, amount = get_data(dataset_name, algorithm_name)

    time_curve, rir_curves, dereverb_curves = interpolate_t60(
        rir_info, dereverb_info, amount
    )

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    plot_t60(
        axes, time_curve, rir_curves, dereverb_curves, dataset_name, algorithm_name
    )

    plt.savefig(save_path, dpi=600)

    plt.show()


def get_all():
    dataset_names = ["SynthesizedDataset", "RealDataset"]
    algorithm_names = ["LP", "HERB", "WPE"]

    save_path = ROOT_PATH / "data" / "plots"
    save_path.mkdir(exist_ok=True, parents=True)
    save_path = save_path / "all_t60_plots.pdf"

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for i, dataset_name in enumerate(dataset_names):
        for j, algorithm_name in enumerate(algorithm_names):
            rir_info, dereverb_info, amount = get_data(dataset_name, algorithm_name)

            time_curve, rir_curves, dereverb_curves = interpolate_t60(
                rir_info, dereverb_info, amount
            )

            plot_t60(
                axes[i][j],
                time_curve,
                rir_curves,
                dereverb_curves,
                dataset_name,
                algorithm_name,
            )

    plt.legend()

    fig.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()


if __name__ == "__main__":
    args = ArgumentParser(description="Plot t60 curve based on a given saved dataset")

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

    if args.dataset_name == "all":
        get_all()
        exit(0)

    get_data_and_plot_t60(args.dataset_name, args.algorithm_name)

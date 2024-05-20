import numba
import numpy as np
import scipy.signal as ss
import torch
from numba.typed import List
from scipy.signal.windows import hamming

from src.HERB.features import (
    LTFTConfig,
    STFTConfig,
    get_all_features,
    get_long_spectrogram,
)


def get_filter(audio, audio_hat):
    spectrogram = get_long_spectrogram(audio)
    spectrogram_hat = get_long_spectrogram(audio_hat)

    print("SPEC shape", spectrogram_hat.shape)

    spectr_filter = np.zeros(spectrogram.shape[0], dtype=spectrogram.dtype)

    for i in range(spectrogram.shape[-1]):
        nonzeros = spectrogram[:, i] != 0
        hat_part = spectrogram_hat[:, i][nonzeros]
        orig_part = spectrogram[:, i][nonzeros]
        spectr_filter[i] = (hat_part / orig_part).sum() / spectrogram.shape[-1]

    # spectr_filter = spectr_filter.mean(axis=-1)

    # convert to time domain
    spectr_filter = np.fft.irfft(spectr_filter)

    return spectr_filter


def apply_filter(audio, spectr_filter, ltft_config=LTFTConfig()):
    print(spectr_filter.shape)
    print(spectr_filter)
    predicted_audio = ss.convolve(audio, spectr_filter, mode="same")
    # predicted_audio = ss.lfilter(spectr_filter, [1], audio)
    # spectrogram = get_long_spectrogram(audio)
    # print("spec_shape_before", spectrogram.shape)
    # spectrogram = spectrogram * spectr_filter[:, None]

    # print("spec_shape", spectrogram.shape)

    # _, predicted_audio = ss.istft(
    #     spectrogram,
    #     fs=ltft_config.sr,
    #     nperseg=ltft_config.nperseg,
    #     noverlap=ltft_config.noverlap,
    #     window=ltft_config.window,
    #     nfft=ltft_config.nfft,
    #     boundary=False
    # )
    return predicted_audio


@numba.njit
def get_audio_hat_numbda(
    amplitude_list, phase_list, f0_list, nperseg, noverlap, sr, N, window
):
    audio_hat_list = []
    for l in range(len(amplitude_list)):
        amplitude_l = amplitude_list[l]
        phase_l = phase_list[l]
        f0_l = f0_list[l]

        n = np.arange(N)
        x_l = np.zeros(N)
        n_l = (nperseg - noverlap) * l + (nperseg // 2 + 1)
        for k in range(len(amplitude_l)):
            x_l = x_l + amplitude_l[k] * np.cos(f0_l[k] / sr * (n - n_l) + phase_l[k])
        audio_hat_list.append(x_l)

    audio_hat = np.zeros(N)
    for i in range(N):
        overlap = 0
        for l in range(len(audio_hat_list)):
            n_l = n_l = (nperseg - noverlap) * l + (nperseg // 2 + 1)
            if nperseg > nperseg // 2 + 1 + i - n_l >= 0:
                coef = window[nperseg // 2 + 1 + i - n_l]
            else:
                coef = 0
            overlap += audio_hat_list[l][i] * coef
        audio_hat[i] = overlap
    return audio_hat


def get_audio_hat(audio_for_f0, audio_for_other, stft_config=STFTConfig()):
    amplitude_list, phase_list, f0_list = get_all_features(
        audio_for_f0, audio_for_other
    )

    numba_list = List()
    for lst in amplitude_list:
        numba_list.append(lst)
    amplitude_list = numba_list

    numba_list = List()
    for lst in phase_list:
        numba_list.append(lst)
    phase_list = numba_list

    numba_list = List()
    for lst in f0_list:
        numba_list.append(lst)
    f0_list = numba_list

    N = audio_for_f0.shape[0]

    window = hamming(stft_config.nperseg)

    audio_hat = get_audio_hat_numbda(
        amplitude_list,
        phase_list,
        f0_list,
        stft_config.nperseg,
        stft_config.noverlap,
        stft_config.sr,
        N,
        window,
    )

    return audio_hat


def one_step(audio_for_f0, audio_for_other):
    audio_hat = get_audio_hat(audio_for_f0, audio_for_other)

    spectr_filter = get_filter(audio_for_other, audio_hat)

    predicted_audio = apply_filter(audio_for_other, spectr_filter)

    return predicted_audio, spectr_filter


def pad_audio(audio, n_repeats):
    if n_repeats == 0:
        return audio
    audio = np.concatenate([audio, np.zeros(1000)])
    audio = np.tile(audio, n_repeats)
    return audio


def dereverberate(short_audio, steps=3, n_repeats=0):
    # Step 0
    audio = pad_audio(short_audio, n_repeats)

    # STEP 1
    predicted_audio_1, spectr_filter_1 = one_step(audio, audio)

    if steps == 1:
        return predicted_audio_1, spectr_filter_1

    # STEP 2
    predicted_audio_2, spectr_filter_2 = one_step(predicted_audio_1, audio)

    if steps == 2:
        return predicted_audio_2, spectr_filter_2

    # STEP 3
    predicted_audio_3, spectr_filter_3 = one_step(predicted_audio_2, predicted_audio_2)

    print(spectr_filter_1.shape, spectr_filter_2.shape, spectr_filter_3.shape)

    spectr_filter_full = ss.convolve(spectr_filter_2, spectr_filter_3, mode="full")

    if steps == 3:
        return predicted_audio_3, spectr_filter_full

    if steps == "two_filters":
        return (
            apply_filter(apply_filter(short_audio, spectr_filter_2), spectr_filter_3),
            spectr_filter_full,
        )

    if steps == "one_filter":
        return apply_filter(short_audio, spectr_filter_2), spectr_filter_2

    return (
        apply_filter(short_audio, spectr_filter_full),
        spectr_filter_1,
        spectr_filter_2,
        spectr_filter_3,
        spectr_filter_full,
    )

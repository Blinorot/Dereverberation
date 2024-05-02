import numpy as np
import scipy.signal as ss
import torch
from scipy.signal.windows import hamming

from src.herb.features import (
    LTFTConfig,
    STFTConfig,
    get_all_features,
    get_long_spectrogram,
)


def get_filter(audio, audio_hat):
    spectrogram = get_long_spectrogram(audio)
    spectrogram_hat = get_long_spectrogram(audio_hat)

    spectr_filter = spectrogram_hat / spectrogram

    spectr_filter = spectr_filter.mean(axis=-1)

    return spectr_filter


def apply_filter(audio, spectr_filter, ltft_config=LTFTConfig()):
    print(spectr_filter.shape)
    print(spectr_filter)
    spectr_filter = np.fft.irfft(spectr_filter)
    print(spectr_filter)
    predicted_audio = ss.convolve(audio, spectr_filter, mode="same")
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


def get_audio_hat(audio_for_f0, audio_for_other, stft_config=STFTConfig()):
    sr = stft_config.sr

    amplitude_list, phase_list, f0_list = get_all_features(
        audio_for_f0, audio_for_other
    )

    N = audio_for_f0.shape[0]

    audio_hat_list = []
    for l in range(len(amplitude_list)):
        amplitude_l = amplitude_list[l]
        phase_l = phase_list[l]
        f0_l = f0_list[l]

        n = np.arange(N)
        x_l = np.zeros(N)
        n_l = (stft_config.nperseg - stft_config.noverlap) * l + (
            stft_config.nperseg // 2 + 1
        )
        for k in range(len(amplitude_l)):
            x_l = x_l + amplitude_l[k] * np.cos(f0_l[k] / sr * (n - n_l) + phase_l[k])
        audio_hat_list.append(x_l)

    window = hamming(stft_config.nperseg)

    audio_hat = np.zeros(N)
    for i in range(N):
        overlap = 0
        for l in range(len(audio_hat_list)):
            n_l = n_l = (stft_config.nperseg - stft_config.noverlap) * l + (
                stft_config.nperseg // 2 + 1
            )
            if stft_config.nperseg > stft_config.nperseg // 2 + 1 + i - n_l >= 0:
                coef = window[stft_config.nperseg // 2 + 1 + i - n_l]
            else:
                coef = 0
            overlap += audio_hat_list[l][i] * coef
        audio_hat[i] = overlap

    return audio_hat


def dereverberate(audio):
    audio_hat = get_audio_hat(audio, audio)

    spectr_filter = get_filter(audio, audio_hat)

    predicted_audio = apply_filter(audio, spectr_filter)

    return predicted_audio

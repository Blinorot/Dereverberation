import numpy as np
import scipy.signal as ss
import torch
from scipy.signal.windows import hamming

from src.herb.features import get_all_features, get_long_spectrogram


def get_filter(audio, audio_hat):
    spectrogram = get_long_spectrogram(audio)
    spectrogram_hat = get_long_spectrogram(audio_hat)

    spectr_filter = spectrogram_hat / spectrogram

    spectr_filter = spectr_filter.mean(axis=-1)

    return spectr_filter


def apply_filter(audio, spectr_filter, sr=16000):
    spectrogram = get_long_spectrogram(audio)
    spectrogram = spectrogram * spectr_filter[:, None]

    _, predicted_audio = ss.istft(
        spectrogram,
        fs=sr,
        nperseg=16000,
        noverlap=16000 // 8,
        window="hamming",
        nfft=16000,
    )
    return predicted_audio


def get_audio_hat(audio_for_f0, audio_for_other, sr=16000):
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
        for k in range(len(amplitude_l)):
            x_l = x_l + amplitude_l[k] * np.cos(f0_l[k] / sr * n + phase_l[k])
        audio_hat_list.append(x_l)

    window = hamming(2 * N)
    frame_shift = 256 // 8

    audio_hat = np.zeros(N)
    for i in range(N):
        overlap = 0
        for l in range(len(audio_hat_list)):
            overlap += audio_hat_list[l][i] * window[N + i - l * frame_shift]
        audio_hat[i] = overlap

    return audio_hat


def dereverberate(audio):
    audio_hat = get_audio_hat(audio, audio)

    spectr_filter = get_filter(audio, audio_hat)

    predicted_audio = apply_filter(audio, spectr_filter)

    return predicted_audio

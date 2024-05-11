from dataclasses import dataclass

import numpy as np
import pyworld as pw
import scipy.signal as ss
from scipy.interpolate import interp1d


@dataclass
class STFTConfig:
    nperseg: int = 512
    noverlap: int = 256
    nfft: int = 512
    sr: int = 16000
    window: str = "hann"


class LTFTConfig:
    nperseg: int = 2048
    noverlap: int = 0
    nfft: int = 2048
    sr: int = 16000
    window: str = "boxcar"


def get_f0(audio, spectrogram, stft_config=STFTConfig()):
    sr = stft_config.sr
    # audio = audio.to(torch.float64).numpy().sum(axis=0)

    # with frame_period=k  k * f0.shape == len(audio) in seconds * 1000
    # we want spectrogram.shape[0] frames, we want f0.shape = spectrogram.shape[0]
    # frame_period = (len(audio) / sr) * 1000  / spectrogram.shape[0]

    frame_period = (audio.shape[0] / sr * 1000) / spectrogram.shape[-1]

    _f0, t = pw.dio(audio, sr, frame_period=frame_period)
    f0 = pw.stonemask(audio, _f0, t, sr)[: spectrogram.shape[-1]]

    nonzeros = np.nonzero(f0)

    x = np.arange(f0.shape[0])[nonzeros]

    values = (f0[nonzeros][0], f0[nonzeros][-1])

    f = interp1d(x, f0[nonzeros], bounds_error=False, fill_value=values)

    new_f0 = f(np.arange(f0.shape[0]))

    return new_f0


def get_amplitude_and_phase(spectrogram):
    amplitude = np.abs(spectrogram)
    phase = np.angle(spectrogram)
    return amplitude, phase


def get_spectrogram(audio, stft_config=STFTConfig()):
    freqs, times, spectrogram = ss.spectrogram(
        audio,
        fs=stft_config.sr,
        nperseg=stft_config.nperseg,
        noverlap=stft_config.noverlap,
        window=stft_config.window,
        nfft=stft_config.nfft,
        mode="complex",
        return_onesided=True,
    )
    print(freqs)
    print(times)
    print(spectrogram)
    print(freqs.shape, times.shape, spectrogram.shape)

    return freqs, times, spectrogram


def get_long_spectrogram(audio, ltft_config=LTFTConfig()):
    freqs, times, spectrogram = ss.spectrogram(
        audio,
        fs=ltft_config.sr,
        nperseg=ltft_config.nperseg,
        noverlap=ltft_config.noverlap,
        window=ltft_config.window,
        nfft=ltft_config.nfft,
        mode="complex",
        return_onesided=True,
    )

    return spectrogram


def extract_amplitude_and_phase_via_f0(amplitude, phase, f0, freqs):
    # amp/phase.shape = F x T
    # each F has info about 1 x 10
    time_shape = amplitude.shape[-1]
    # freq_shape = amplitude.shape[-2]

    amplitude_list = []
    phase_list = []
    f0_list = []
    for l in range(time_shape):
        f0_l = f0[l]
        phase_l = phase[..., l]
        amplitude_l = amplitude[..., l]

        amplitude_list_l = []
        phase_list_l = []
        f0_list_l = []

        k = 1
        while k * f0_l <= freqs[-1]:
            f0_lk = k * f0_l

            diff_freqs = np.abs(freqs - f0_lk)

            freq_bin = np.argmin(diff_freqs)

            phase_lk = phase_l[freq_bin]
            amplitude_lk = amplitude_l[freq_bin]

            amplitude_list_l.append(amplitude_lk)
            phase_list_l.append(phase_lk)
            f0_list_l.append(f0_lk)

            k += 1

        amplitude_list.append(amplitude_list_l)
        phase_list.append(phase_list_l)
        f0_list.append(f0_list_l)

    return amplitude_list, phase_list, f0_list


def get_all_features(audio_for_f0, audio_for_other):
    freqs, times, spectrogram = get_spectrogram(audio_for_other)
    amplitude, phase = get_amplitude_and_phase(spectrogram)

    f0 = get_f0(audio_for_f0, spectrogram)

    print(amplitude.shape, phase.shape, f0.shape)
    print(f0)

    amplitude_list, phase_list, f0_list = extract_amplitude_and_phase_via_f0(
        amplitude, phase, f0, freqs
    )

    print([len(elem) for elem in phase_list])

    return amplitude_list, phase_list, f0_list

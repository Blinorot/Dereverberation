import numpy as np
from nara_wpe.utils import istft, stft
from nara_wpe.wpe import wpe


# define a funciton that given an input signal, it returns the dereverberated signal using wpe
def wpe_dereverberation(signal, taps=10, delay=3, iterations=3):
    signal = signal.reshape(1, -1)
    # apply WPE to the signal
    Y = stft(signal, size=512, shift=128)
    Y = Y.transpose(2, 0, 1)
    Z = wpe(Y, taps=taps, delay=delay, iterations=iterations, statistics_mode="full")
    dereverberated_signal = istft(Z.transpose(1, 2, 0), size=512, shift=128)

    inverse_filter = get_wpe_filter(signal, dereverberated_signal)

    return dereverberated_signal.reshape(-1), inverse_filter


def get_wpe_filter(signal, dereverberated_signal):
    # signal = signal.reshape(1, -1)
    # dereverberated_signal = dereverberated_signal.reshape(1, -1)

    window_size = 2048
    shift = 1024

    # B x T x F
    Y = stft(dereverberated_signal, size=window_size, shift=shift)[0]
    X = stft(signal, size=window_size, shift=shift)[0]

    inverse_filter = (Y / X).mean(0)
    inverse_filter = np.fft.irfft(inverse_filter)

    return inverse_filter

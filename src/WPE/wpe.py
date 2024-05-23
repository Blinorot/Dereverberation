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
    return dereverberated_signal.reshape(-1)

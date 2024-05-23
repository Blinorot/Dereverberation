
import numpy as np
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

# define a funciton that given an input signal, it returns the dereverberated signal
def wpe_dereverberation(signal, taps=10, delay=3, iterations=3):
    # apply WPE to the signal
    Y = stft(signal, size=512, shift=128)
    Y = Y.transpose(2, 0, 1)
    Z = wpe(Y, taps=taps, delay=delay, iterations=iterations, statistics_mode='full')
    dereverberated_signal = istft(Z.transpose(1, 2, 0), size=512, shift=128)
    return dereverberated_signal


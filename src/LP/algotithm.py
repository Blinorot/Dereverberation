import numpy as np
import scipy
from scipy.signal import istft, lfilter, lfilter_zi, stft


def linear_prediction_analysis(signal):
    # Order of the linear prediction analysis
    order = 10

    # Compute the coefficients of the linear prediction analysis
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    r = autocorr[: order + 1]
    R = np.zeros((order, order))
    for i in range(order):
        R[i, :] = autocorr[i : i + order]
    a = np.linalg.solve(R, r[1:])

    # Compute the predicted signal
    predicted_signal = np.zeros_like(signal)
    for i in range(order, len(signal)):
        predicted_signal[i] = np.dot(a, signal[i - order : i])

    # Compute the residual signal
    residual_signal = signal - predicted_signal

    return predicted_signal, residual_signal


def calculate_kurtosis(residual_signal):
    kurtosis = (
        np.mean(
            (residual_signal - np.mean(residual_signal, axis=1)[:, None]) ** 4, axis=1
        )
        / np.mean(
            (residual_signal - np.mean(residual_signal, axis=1)[:, None]) ** 2, axis=1
        )
        ** 2
    )
    return kurtosis


def adaptive_inverse_filtering(signal, nIterations, predicted_signal, residual_signal):
    # Set the initial estimate of the inverse filter coefficients as a ramp
    inverse_filter_coeffs = np.linspace(0, 1, len(signal), dtype=np.float32)

    # Apply the adaptive inverse filtering algorithm for the specified number of iterations
    for i in range(nIterations):
        # Calculate the gradient of the kurtosis with respect to the inverse filter coefficients
        gradient = np.zeros_like(signal)

        # Perturb the current estimate of the inverse filter coefficients
        perturbed_coeffs = inverse_filter_coeffs.copy()
        perturbed_coeffs += 1e-6

        # Calculate the perturbed kurtosis
        perturbed_predicted_signal = predicted_signal + np.dot(
            perturbed_coeffs[:, None], residual_signal[-1::-1][None, :]
        )
        print(np.dot(perturbed_coeffs[:, None], residual_signal[-1::-1][None, :]).shape)
        perturbed_residual_signal = signal - perturbed_predicted_signal
        perturbed_kurtosis = calculate_kurtosis(perturbed_residual_signal)

        # Calculate the gradient of the kurtosis with respect to the j-th inverse filter coefficient
        gradient = (
            perturbed_kurtosis - calculate_kurtosis(residual_signal[None, :])
        ) / 1e-6

        # Update the estimate of the inverse filter coefficients using the gradient descent algorithm
        inverse_filter_coeffs -= 1e-3 * gradient

    return inverse_filter_coeffs


def inverse_filter_frequency_domain(rev_residual, mu=3e-6, LF=300, niter=200):
    p = LF  # Filter order
    ss = LF // 5  # Step size
    bs = ss + p - 1  # Block size

    # Generate initial guess for inverse filter
    gh = np.arange(1, p + 1) / np.sqrt(np.sum(np.arange(1, p + 1)) ** 2)
    Gh = np.fft.fft(np.concatenate((gh, np.zeros(ss - 1))))

    zkurt = np.zeros(niter)

    zr2 = np.zeros(bs)
    zr3 = np.zeros(bs)
    zr4 = np.zeros(bs)

    for m in range(niter):
        yrn = np.zeros(bs)
        for k in range(0, len(rev_residual), ss):
            yrn[: p - 1] = yrn[-(p - 1) :]
            rev_slice = rev_residual[k : k + ss - 1]
            if rev_slice.shape[-1] < yrn[p:].shape[-1]:
                rev_slice = np.concatenate(
                    (rev_slice, np.zeros(yrn[p:].shape[-1] - rev_slice.shape[-1]))
                )
            yrn[p:] = rev_slice

            Yrn = np.fft.fft(yrn)
            # cYrn = np.conj(Yrn)
            zrn = np.fft.ifft(Gh * Yrn)

            zrn[: p - 1] = 0
            zr2[p:] = zrn[p:] ** 2
            zr3[p:] = zrn[p:] ** 3
            zr4[p:] = zrn[p:] ** 4

            Z2 = np.sum(zr2[p:])
            Z4 = np.sum(zr4[p:])

            zkurt[m] = max(zkurt[m], Z4 / (Z2**2 + 1e-15) * ss)

            z3y = np.fft.fft(zr3) * Yrn
            zy = np.fft.fft(zrn) * Yrn

            gJ = 4 * (Z2 * z3y - Z4 * zy) / (Z2**3 + 1e-20) * ss
            Gh = Gh + mu * gJ

            # Normalize Gh
            Gh = Gh / np.sqrt(np.sum(np.abs(Gh) ** 2) / bs)

            G_time = np.fft.irfft(Gh)

    return G_time


def spectral_subtraction(
    z,
    fs,
    wlen=64e-3,
    hop=8e-3,
    nfft=1024,
    ro_w=7,
    a_w=5,
    gS=0.32,
    epS=1e-3,
    nu1=0.05,
    nu2=4,
):
    # STFT calculation
    wlen_samples = int(wlen * fs)
    hop_samples = int(hop * fs)
    _, _, Sz = stft(
        z, fs, nperseg=wlen_samples, noverlap=wlen_samples - hop_samples, nfft=nfft
    )

    # Power spectrogram calculation
    Pz = np.abs(Sz) ** 2

    # Windowing function
    i_w = np.arange(-ro_w, 16)
    wS = (i_w + a_w) / (a_w**2) * np.exp(-0.5 * (i_w / a_w + 1) ** 2)
    wS[i_w < -a_w] = 0

    # Attenuation calculation
    Pl = gS * lfilter(wS, 1, Pz.T).T
    P_att = (Pz - Pl) / Pz
    P_att[P_att < epS] = epS

    # Apply spectral subtraction
    Pxh = Pz * P_att
    Sxh = np.sqrt(Pxh) * np.exp(1j * np.angle(Sz))

    # Calculate energy of original and modified signal
    Ez = np.sum(np.abs(Sz) ** 2, axis=0) / Sz.shape[0]
    Exh = np.sum(np.abs(Sxh) ** 2, axis=0) / Sxh.shape[0]

    # Adjust spectral envelope
    P_att = np.ones(Sz.shape[1])
    P_att[(Ez < nu1) & (Ez / Exh > nu2)] = 1e-3
    # Expand P_att to match the shape of Sxh
    expanded_P_att = np.tile(P_att, (Sxh.shape[0], 1))
    # Multiply Sxh with expanded_P_att
    Sxh *= expanded_P_att

    # Inverse STFT to reconstruct signal
    xh = istft(
        Sxh, fs, nperseg=wlen_samples, noverlap=wlen_samples - hop_samples, nfft=nfft
    )[1]
    xh /= np.max(np.abs(xh))

    return xh


def LP_dereverberation(signal, mu=3e-9, Lf=300, nIterations=200):
    # Apply linear prediction analysis to the signal
    predicted_signal, residual_signal = linear_prediction_analysis(signal)
    # inverse_filter = adaptive_inverse_filtering(signal, 100, predicted_signal, residual_signal)
    inverse_filter = inverse_filter_frequency_domain(
        residual_signal, mu, Lf, nIterations
    )
    inverse_filtered_signal = lfilter(inverse_filter, 1, signal)
    dereverberated_signal = spectral_subtraction(inverse_filtered_signal, 16000)

    return dereverberated_signal, inverse_filter

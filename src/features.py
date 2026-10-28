# ============================================================
# FEATURE EXTRACTION
# Each RF signal has 128 I/Q samples. From each signal we
# calculate 176 numbers (features) that describe it.
# ============================================================
import numpy as np
import scipy.signal


def basic_features(I, Q):
    # The original 27 features: amplitude, phase, frequency,
    # moments, I/Q statistics and spectrum shape
    z = I + 1j * Q
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    freq = np.diff(phase)

    m20 = np.mean(z ** 2)
    m40 = np.mean(z ** 4)
    c40 = m40 - 3 * (m20 ** 2)

    freqs, psd = scipy.signal.welch(amp, nperseg=64)
    centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / (np.sum(psd) + 1e-12))

    skewness = np.mean((amp - np.mean(amp)) ** 3) / (np.std(amp) ** 3 + 1e-12)
    kurtosis = np.mean((amp - np.mean(amp)) ** 4) / (np.std(amp) ** 4 + 1e-12)

    return [
        np.mean(amp), np.std(amp), np.var(amp), np.sqrt(np.mean(amp ** 2)), skewness, kurtosis,
        np.mean(phase), np.std(phase), np.var(phase),
        np.mean(freq), np.std(freq), np.var(freq),
        np.real(m20), np.imag(m20), np.real(m40), np.imag(m40), np.real(c40), np.imag(c40),
        np.real(np.mean(z * np.conj(z))), np.real(np.mean((z ** 2) * (np.conj(z) ** 2))),
        np.mean(I), np.std(I), np.mean(Q), np.std(Q),
        centroid, spread, np.max(psd),
    ]


def cumulant_features(z):
    # 4th-order cumulants: different modulations have different
    # textbook values (e.g. C42 = -2 for BPSK, -1 for QPSK, -0.68 for QAM16)
    x = z - np.mean(z)
    m20 = np.mean(x ** 2)
    m21 = np.mean(np.abs(x) ** 2)
    m40 = np.mean(x ** 4)
    m41 = np.mean(x ** 3 * np.conj(x))
    m42 = np.mean(np.abs(x) ** 4)

    c40 = m40 - 3 * m20 ** 2
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2

    # divide by power^2 so the values do not depend on signal strength
    return [np.abs(c40) / m21 ** 2, np.abs(c41) / m21 ** 2, c42 / m21 ** 2]


def power_peak_features(z):
    # Raising the signal to a power removes PSK modulation and leaves a
    # strong spectral peak: BPSK at z^2, QPSK at z^4, 8PSK at z^8
    result = []
    for k in [1, 2, 4, 8]:
        spectrum = np.abs(np.fft.fft(z ** k)) ** 2
        result.append(np.max(spectrum) / (np.sum(spectrum) + 1e-12))
    return result


def phase_features(z):
    # Phase moments: tell how many phase levels the signal uses
    amp = np.abs(z)
    unit = z / (amp + 1e-12)          # keep only the phase
    result = []
    for k in [1, 2, 4, 8]:
        result.append(np.abs(np.mean(unit ** k)))
    return result


def amplitude_features(z):
    # Amplitude distribution: 10%, 25%, 50%, 75% and 90% points
    amp = np.abs(z)
    amp = amp / (np.mean(amp) + 1e-12)
    return list(np.quantile(amp, [0.10, 0.25, 0.50, 0.75, 0.90]))


def autocorrelation_features(z):
    # How similar the signal is to a shifted copy of itself
    result = []
    for lag in [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]:
        result.append(np.abs(np.mean(z[lag:] * np.conj(z[:-lag]))))
    # same for z^2 (useful for phase modulations)
    z2 = z ** 2
    for lag in [1, 2, 4, 8]:
        result.append(np.abs(np.mean(z2[lag:] * np.conj(z2[:-lag]))))
    return result


def snr_feature(z):
    # Rough estimate of the signal-to-noise ratio (M2M4 estimator)
    amp = np.abs(z)
    m2 = np.mean(amp ** 2)
    m4 = np.mean(amp ** 4)
    s = np.sqrt(max(2 * m2 ** 2 - m4, 0))
    noise = max(m2 - s, 1e-6)
    ratio = min(max(s / noise, 1e-3), 1e3)
    return [10 * np.log10(ratio)]


def spectrum_bins(signal, n_bins):
    # Power spectrum split into n_bins equal parts (each part = share of total power)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(signal)) ** 2)
    spectrum = spectrum / (np.sum(spectrum) + 1e-12)
    size = len(spectrum) // n_bins
    result = []
    for i in range(n_bins):
        result.append(np.sum(spectrum[i * size:(i + 1) * size]))
    return result


def spectrum_features(z):
    result = []
    result += spectrum_bins(z, 64)          # spectrum of the signal (bandwidth, shape)
    result += spectrum_bins(z ** 2, 16)     # spectrum of z^2
    result += spectrum_bins(z ** 4, 16)     # spectrum of z^4
    # envelope spectrum: how the amplitude changes (AM signals)
    amp = np.abs(z)
    envelope = np.abs(np.fft.fft(amp - np.mean(amp)))[:64] ** 2
    envelope = envelope / (np.sum(envelope) + 1e-12)
    for i in range(16):
        result.append(np.sum(envelope[i * 4:(i + 1) * 4]))
    return result


def frequency_features(z):
    # Instantaneous frequency distribution (useful for FSK / FM)
    freq = np.diff(np.unwrap(np.angle(z)))
    return list(np.quantile(freq, [0.05, 0.25, 0.50, 0.75, 0.95]))


def extract_features(signal):
    # signal has shape (2, 128): row 0 = I, row 1 = Q
    I = signal[0]
    Q = signal[1]

    # complex signal scaled to power 1 (so features do not depend on signal strength)
    z = I.astype(np.float64) + 1j * Q.astype(np.float64)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    features = []
    features += basic_features(I, Q)          # 27
    features += cumulant_features(z)          # 3
    features += power_peak_features(z)        # 4
    features += phase_features(z)             # 4
    features += amplitude_features(z)         # 5
    features += autocorrelation_features(z)   # 15
    features += snr_feature(z)                # 1
    features += spectrum_features(z)          # 112
    features += frequency_features(z)         # 5
    return np.nan_to_num(np.array(features, dtype=np.float32))

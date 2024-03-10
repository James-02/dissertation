import numpy as np
from scipy.signal import find_peaks, correlate, correlation_lags

def compute_period(signal, dt):
    ac = np.correlate(signal, signal, mode='full')

    # Find peaks in the auto-correlation
    peaks, _ = find_peaks(ac)

    # Calculate the period as the mean of the differences between consecutive peaks (excluding first and last)
    return np.mean(np.diff(peaks[1:-1])) * dt

def prof_pulse(t, period, phase):
    return 1 * ((np.mod((t - phase) * 0.5 * (period / np.pi), period) / period) < 0.5)

def prof_cos(t, period, phase):
    return 0.5 * (1  + np.cos(2 * np.pi * t / period + phase))

def compute_phase(signal_1, signal_2, period, dt):
    # Compute cross-correlation between the two signals
    correlation = correlate(signal_1, signal_2)
    lags = correlation_lags(len(signal_1), len(signal_2))

    # Extract the positive lags and their corresponding cross-correlation values
    correlation = correlation[len(correlation) // 2:]
    lags = lags[len(lags) // 2:]    # Consider only positive lags

    # Find peaks in the cross-correlation
    peaks, _ = find_peaks(correlation)

    if len(peaks) == 0:
        raise ValueError("No peaks found in cross-correlation function.")

    lag = lags[peaks[0]]

    # Calculate phase difference
    phase_diff = (360 * lag / period * dt) % 360

    # Adjust if greater than 180 degrees
    return min(phase_diff, 360 - phase_diff)
    
import numpy as np
from scipy.signal import hilbert

def compute_phase(derivatives):
    return np.angle(hilbert(derivatives))

def prof_pulse(t, period, phase):
    return 1 * ((np.mod((t - phase) * 0.5 * (period / np.pi), period) / period) < 0.5)

def prof_cos(t, period, phase):
    return 0.5 * (1  + np.cos(2 * np.pi * t / period + phase))

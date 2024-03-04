from dde import ddeint, dde_system
from preprocessing import load_mnist
from utils import compute_phase
from visualisation import plot_danino

import numpy as np

def initial_conditions(t):
    """
    Define the initial conditions for the dde system
    """
    return [0, 100, 0, 0]

def run_danino():
    # Parameter specification
    params = {
        'CA': 1, 'CI': 4, 'del_': 1e-3, 'alpha': 2500, 'k': 1, 'k1': 0.1, 'b': 0.06, 'gammaA': 15,
        'gammaI': 24, 'gammaH': 0.01, 'f': 0.3, 'g': 0.01, 'd': 0.7, 'd0': 0.88, 'D': 2.5, 'mu': 0.6,
        'period': 1.5, 'phase': 0.1, 'coupling': 0.1, 'delay': 10, 'n': 10
    }

    # Load training and testing data
    X_train, Y_train, X_test, Y_test = load_mnist()

    # Time points [0 - 1000], with 1000 intervals (+1 increments)
    time = np.linspace(0, 1000, 1000)

    # TODO: Use data as input to the oscillator
        # TODO: Create input phase and set and set as params to the simulator

    # Solve the delayed differential equations to get the derivatives of the system variables
    results = ddeint(dde_system, initial_conditions, time, params['delay'], args=(params,))

    # Compute the phases of the waveform
    # TODO: Is this correct to calculate the phases?
    phases = np.array([compute_phase(y) for y in results])

    # TODO: Use phases to feed into next node for coupled oscillator model

    # Plot the phases over time
    plot_danino(time, phases)

    # Plot the derivatives over time
    plot_danino(time, results)


if __name__ == "__main__":
    run_danino()
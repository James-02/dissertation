from dde import ddeint, dde_system
from preprocessing import load_mnist, load_ecg, normalise_to_phases
from utils import compute_phase, prof_pulse, compute_period
from visualisation import plot_danino

import numpy as np

def initial_conditions(t):
    return [0, 100, 0, 0]

def run_danino():
    # Parameter specification
    params = {
        'CA': 1, 'CI': 4, 'del_': 1e-3, 'alpha': 2500, 'k': 1, 'k1': 0.1, 'b': 0.06, 'gammaA': 15,
        'gammaI': 24, 'gammaH': 0.01, 'f': 0.3, 'g': 0.01, 'd': 0.7, 'd0': 0.88, 'D': 2.5, 'mu': 0.6,
        'delay': 10, 'n': 10, 'period': 0, 'phase': 0, 'coupling': 0
    }

    # Load train and test datasets
    X_train, Y_train, X_test, Y_test = load_ecg()

    # Time points [0 - 1000], with intervals of the training set size
    intervals = 187
    time = np.linspace(0, 1000, intervals)

    # Normalise data to 0-180 degree phases and use as input phase
    X_train_normalised = normalise_to_phases(X_train)
    input_phase = X_train_normalised[0][0]
    print("Input Phase: " + str(input_phase))

    # initial run with no coupling to compute the period
    results = ddeint(dde_system, initial_conditions, time, params['delay'], args=(params,))

    # Extract the A signal from initial run for use to calculate the period
    A_signal = [row[0] for row in results]

    # Calculate the period
    period = compute_period(A_signal, np.mean(np.diff(time)))
    print("Calculated Period: " + str(period))

    # Update the parameters to add coupling, the calculated period and input phase
    params.update({'period': period, 'input_phase': input_phase, 'coupling': 0.007})

    # Generate a reference signal at phase 0
    reference_signal = prof_pulse(time, params['period'], 0)

    # Solve the delayed differential equations to get the derivatives of the system variables
    results = ddeint(dde_system, initial_conditions, time, params['delay'], args=(params,))

    # Extract the A signal
    A_signal = [row[0] for row in results]

    # Calculate the phase between the reference signal and the A signal
    phase = compute_phase(A_signal, reference_signal, period, np.mean(np.diff(time)))
    print("Output Phase: " + str(phase))
    # This phase is now the output of the node (the state)

    # Plot the derivatives over time
    plot_danino(time, results)


if __name__ == "__main__":
    run_danino()

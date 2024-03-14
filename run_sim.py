from oscillator.dde import ddeint, dde_system

import matplotlib.pyplot as plt

import numpy as np

def run_danino():
    # Parameter specification
    params = {
        'CA': 1, 'CI': 4, 'del_': 1e-3, 'alpha': 2500, 'k': 1, 'k1': 0.1, 'b': 0.06, 'gammaA': 15,
        'gammaI': 24, 'gammaH': 0.01, 'f': 0.3, 'g': 0.01, 'd': 0.7, 'd0': 0.88, 'D': 2.5, 'mu': 0.6,
        'delay': 10, 'n': 10, 'phase': 0, 'coupling': 0.01, 'period': 25, 'time': np.linspace(0, 1, 2),
        'initial_conditions': [0, 100, 0, 0]
    }

    # Example training data is a sine wave over 187 time points
    X_train = np.linspace(0, np.pi, 187)

    # set the full time scale 
    full_time = np.linspace(0, 187, 187)

    # Start a plotting figure
    plt.figure()

    # Capture the states of each time step
    states = []

    # run dde for each timestep of data
    for i in range(len(X_train)):

        # input phase is the timestep of the data
        input_phase = X_train[i]

        # Update the parameters with the input phase
        params.update({'phase': input_phase})
    
        # Solve the delayed differential equations to get the derivatives of the system variables
        results = ddeint(dde_system, lambda _: params['initial_conditions'], params['time'], params['delay'], args=(params,))

        # Collect the results as the states of the time step
        states.append(results)

        # Set the initial conditions to be the final row of previous dde results
        params['initial_conditions'] = results[-1]

    # Plot the mean of each state over the full timeseries (each state is 100 intervals of 1 timestep)
    states = np.mean(states, axis=1)
    plt.plot(full_time, states[:, 0])
    plt.plot(full_time, states[:, 1])
    plt.plot(full_time, states[:, 2])
    plt.plot(full_time, states[:, 3])

    # Display the plotted figure
    plt.show()

if __name__ == "__main__":
    run_danino()

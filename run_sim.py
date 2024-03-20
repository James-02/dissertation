from oscillator.dde import ddeint, dde_system

import matplotlib.pyplot as plt

import numpy as np

import time


# Parameter specification
params = {
    'CA': 1, 'CI': 4, 'del_': 1e-3, 'alpha': 2500, 'k': 1, 'k1': 0.1, 'b': 0.06, 'gammaA': 15,
    'gammaI': 24, 'gammaH': 0.01, 'f': 0.3, 'g': 0.01, 'd': 0.7, 'd0': 0.88, 'D': 2.5, 'mu': 0.6,
    'delay': 10, 'input': 71, 'coupling': 5e-4, 'time': np.linspace(0, 1, 100), 'initial_conditions': [0, 100, 0, 0]
}

# create the states array and set the initial conditions as the first element
states = np.array(params['initial_conditions']).reshape(1, -1)

def history(t):
    timesteps = states.shape[0]
    features = states.shape[1]

    if timesteps < abs(t):
        return states[0]

    return [np.interp(t, np.arange(-timesteps + 1, 1), states[:, i]) for i in range(features)]


def run_danino():
    timepoints = 187
    # Example training data is a sine wave over 187 time points
    X_train = np.sin(np.linspace(0, 8 * np.pi, timepoints))

    # set the full time scale 
    full_time = np.linspace(0, timepoints, timepoints)

    # Start a plotting figure
    plt.figure()

    global states

    start = time.time()

    # run dde for each timestep of data
    for i in range(len(X_train)):
        # input phase is the timestep of the data
        input_val = X_train[i]

        # Update the parameters with the input phase
        params.update({'input': input_val})

        # Solve the delayed differential equations to get the derivatives of the system variables
        results = ddeint(dde_system, history, params['time'], params['delay'], args=(params,))

        # Collect the results as the states of the time step
        states = np.concatenate((states, results[-1].reshape(1, -1)), axis=0)
    
    end = time.time()
    print("Time: " + str(end - start))

    # Plot the mean of each state over the full timeseries (each state is 100 intervals of 1 timestep)
    plt.plot(full_time, states[1:, 0])
    plt.plot(full_time, states[1:, 1])
    plt.plot(full_time, states[1:, 2])
    plt.plot(full_time, states[1:, 3])

    # Display the plotted figure
    plt.show()

if __name__ == "__main__":
    run_danino()

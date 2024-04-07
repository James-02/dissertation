

import numpy as np
import matplotlib.pyplot as plt

from reservoir.reservoir import OscillatorReservoir

from reservoirpy.datasets import mackey_glass, to_forecasting
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.nodes import Reservoir, Ridge

# Load Mackey-Glass dataset
timesteps = 2510
tau = 17
X = mackey_glass(timesteps, tau=tau)

# Rescale between -1 and 1
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

# Define ESN parameters
units = 100
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.1
ridge = 1e-8
seed = 1234

# Function to reset ESN
def reset_esn():
    reservoir = OscillatorReservoir(units, timesteps=timesteps, delay=10, input_scaling=input_scaling, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, seed=seed)
    readout = Ridge(1, ridge=ridge)
    return reservoir >> readout

# Initialize ESN
esn = reset_esn()

def plot_train_test(X_train, y_train, X_test, y_test):
    sample = 500
    test_len = X_test.shape[0]
    fig = plt.figure(figsize=(15, 5))
    plt.plot(np.arange(0, 500), X_train[-sample:], label="Training data")
    plt.plot(np.arange(0, 500), y_train[-sample:], label="Training ground truth")
    plt.plot(np.arange(500, 500+test_len), X_test, label="Testing data")
    plt.plot(np.arange(500, 500+test_len), y_test, label="Testing ground truth")
    plt.legend()
    plt.savefig("results/forecasting/mackey-glass.png", dpi=800)


def plot_results(y_pred, y_test, sample=500):
# Function to plot forecasting results
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
    ax.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
    ax.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Forecasting Results")
    plt.savefig("results/forecasting/forecasting.png", dpi=800)

# Prepare data for the second forecasting task
x, y = to_forecasting(X, forecast=10)
X_train2, y_train2 = x[:2000], y[:2000]
X_test2, y_test2 = x[2000:], y[2000:]
plot_train_test(X_train2, y_train2, X_test2, y_test2)

# Fit ESN and run for the second set of data
esn.fit(X_train2, y_train2)
results = esn.run(X_test2)
plot_results(results, y_test2, sample=400)

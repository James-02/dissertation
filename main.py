from reservoirpy.nodes import Reservoir, Ridge

import reservoirpy as rpy
import numpy as np

from utils.logger import Logger
from utils.preprocessing import load_ecg_data, load_npz
from utils.classification import classify
from utils.visualisation import plot_states, plot_dataset_info, plot_class_mean, plot_data_distribution, \
    plot_class_std, plot_average_instance, plot_confusion_matrix, plot_metrics_across_folds, plot_metrics_table, plot_class_metrics
from reservoir.reservoir import OscillatorReservoir, Oscillator

# Define global constants
SEED = 1337
VERBOSITY = 0
LOG_LEVEL = 1

OSCILLATOR_RESERVOIR = "oscillator_reservoir"
NEURON_RESERVOIR = "neuron_reservoir"

# set global config
rpy.set_seed(SEED)
rpy.verbosity(VERBOSITY)
np.random.seed(SEED)

def plot_dde_states():
    timesteps = np.linspace(0, 1000, 1000)

    # Calculate the sine values for each time step
    sine_wave = np.sin(timesteps)
    oscillator = Oscillator(timesteps=1000, delay=10, initial_values=[0, 100, 0, 0])
    states = []
    for i in range(1000):
        states.append(oscillator.forward(sine_wave[i] * 1e-5))

    plot_states(np.array(states), labels=["A", "I", "Hi", "He"], ylabel="Concentration", title="", legend=True, show=True)
    
def plot_reservoir_states(reservoir, X_train, iterations=1):
    labels = [f"Node: {i}" for i in range(nodes)]
    for x in X_train[:iterations]:
        plot_states(reservoir.run(x), labels, legend=True)

def analyse_dataset(X_train, Y_train, X_test, Y_test):
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)

    plot_data_distribution(Y, filename="binary-data-distribution.png", show=True)
    plot_dataset_info(X_train, Y_train, X_test, Y_test, filename="binary-dataset-info.png", show=True)
    plot_average_instance(X, Y, filename="binary-average-instance", show=True)
    plot_class_std(X, Y, filename="binary-std.png", show=True)
    plot_class_mean(X, Y, filename="binary-means.png", show=True)


def main(use_oscillators: bool = True, analyse_data: bool = False, plot_states: bool = False):
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=instances,
        test_ratio=0.2,
        normalize=True,
        encode_labels=True,
        repeat_targets=False,
        binary=False)

    if analyse_data:
        analyse_dataset(X_train, Y_train, X_test, Y_test)
        return

    if use_oscillators:
        reservoir = OscillatorReservoir(
            units=nodes,
            timesteps=X_train[0].shape[0],
            delay=7,
            rc_scaling=5e-7,
            initial_values=[0, 150, 0, 0],
            seed=SEED)
    else:
        reservoir = Reservoir(units=nodes)

    # # Initialize readout node
    readout = Ridge(ridge=ridge)

    if plot_states:
        plot_reservoir_states(reservoir, X_train)
        return

    # Perform classification
    save_file = f"results/runs/{reservoir.name}-{nodes}-{len(X_train)}"
    classify(reservoir, readout, X_train, Y_train, X_test, Y_test, folds=folds, save_file=save_file)

    if not folds:
        metrics = load_npz(save_file + ".npz", allow_pickle=True)['metrics'].item()
        plot_class_metrics(metrics['class_metrics'])
        plot_metrics_table(metrics['metrics'].item())
        plot_confusion_matrix(metrics['confusion_matrix'])
    else:
        for fold in range(folds):
            filename = save_file + f"-fold-{str(fold)}.npz"
            fold_metrics = load_npz(filename, allow_pickle=True)['metrics'].item()
            class_metrics = fold_metrics['class_metrics']
            # plot class metrics across each fold

if __name__ == "__main__":
    instances = 500
    nodes = 40
    folds = None
    ridge = 1e-5

    log_file = f"logs/{nodes}_nodes.log"
    logger = Logger(level=LOG_LEVEL, log_file=log_file)

    main()

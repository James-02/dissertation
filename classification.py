from multiprocessing import Pool
import time

import numpy as np
import reservoirpy as rpy

from reservoirpy.nodes import Reservoir, Ridge
from sklearn.metrics import accuracy_score

from reservoir.reservoir import OscillatorReservoir

from preprocessing import load_ecg_data
from visualisation import plot_states

SEED = 1337
rpy.set_seed(SEED)

def __train(args):
    x, reservoir = args
    return reservoir.run(x)

def __predict(args):
    x, reservoir, readout = args
    states = reservoir.run(x)
    return readout.run(states[-1, np.newaxis])

def _train(X_train, reservoir, use_multiprocessing, plot):
    if use_multiprocessing:
        with Pool() as pool:
            results_train = pool.map(__train, [(x, reservoir) for x in X_train])
    else:
        results_train = [__train((x, reservoir)) for x in X_train]
        if plot:
            for states in results_train:
                _plot(states)

    return [result[-1, np.newaxis] for result in results_train]

def _predict(X_test, reservoir, readout, use_multiprocessing):
    if use_multiprocessing:
        with Pool() as pool:
            results_pred = pool.map(__predict, [(x, reservoir, readout) for x in X_test])
        return results_pred
    else:
        return [__predict((x, reservoir, readout)) for x in X_test]
    
def _plot(states):
    timesteps = len(states)
    timespan = np.linspace(0, timesteps, timesteps)
    plot_states(timespan, states)

def classification(use_oscillator=True, use_multiprocessing=True, plot_states=False):
    # Load dataset
    X_train, Y_train, X_test, Y_test = load_ecg_data(class_size=10)
    timesteps = X_train[0].shape[0]

    # Use the oscillator node as the reservoir if the flag is set
    reservoir = OscillatorReservoir(units=5, timesteps=timesteps) if use_oscillator else Reservoir(100, sr=0.9, lr=0.1)

    # Initialize reservoir and readout
    readout = Ridge(ridge=1e-6)

    # Training
    start = time.time()
    states_train = _train(X_train, reservoir, use_multiprocessing, plot_states)
    end = time.time()
    print("Training Time: " + str(end - start) + "s")

    # Fitting
    print("Fitting")
    readout.fit(states_train, Y_train)

    # Predicting
    print("Predicting")
    Y_pred = _predict(X_test, reservoir, readout, use_multiprocessing)

    Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
    Y_test_class = [np.argmax(y_t) for y_t in Y_test]

    # Calculate prediction accuracy
    score = accuracy_score(Y_test_class, Y_pred_class)
    print("Accuracy: ", f"{score * 100:.3f} %")

if __name__ == "__main__":
    classification(use_oscillator=True, use_multiprocessing=False, plot_states=False)

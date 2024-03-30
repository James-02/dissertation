from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.datasets import japanese_vowels
from sklearn.metrics import accuracy_score

from preprocessing import load_ecg_data

from reservoir.reservoir import OscillatorReservoir

import numpy as np
import reservoirpy as rpy

from visualisation import *

SEED = 1337

rpy.set_seed(SEED)

def transduction(use_oscillator=True):
    X_train, Y_train, X_test, Y_test = load_ecg_data(class_size=10, repeat_targets=True)
    reservoir = OscillatorReservoir(units=10, timesteps=X_train[0].shape[0])

    if not use_oscillator:
        reservoir = Reservoir(10, sr=0.9, lr=0.1)

    source = Input()
    readout = Ridge(ridge=1e-6)
    model = source >> reservoir >> readout
    
    # train and test the model
    Y_pred = model.fit(X_train, Y_train).run(X_test)

    Y_pred_class = [np.argmax(y_p, axis=1) for y_p in Y_pred]
    Y_test_class = [np.argmax(y_t, axis=1) for y_t in Y_test]

    # check the accuracy of predicted vs real targets
    score = accuracy_score(np.concatenate(Y_test_class, axis=0), np.concatenate(Y_pred_class, axis=0))
    print("Accuracy: ", f"{score * 100:.3f} %")

def classification(use_oscillator=True, plot=True):
    # Load dataset
    X_train, Y_train, X_test, Y_test = load_ecg_data(class_size=10)
    timespan = np.linspace(0, 187, 187)

    # initialize genetic oscillator node
    reservoir = OscillatorReservoir(units=10, timesteps=187)

    # initialize other nodes
    readout = Ridge(ridge=1e-6)

    # Use the oscillator node as the reservoir if the flag is set
    if not use_oscillator:
        reservoir = Reservoir(10, sr=0.9, lr=0.1)

    states_train = []
    print("Training")
    for x in X_train:
        # train reservoir
        states = reservoir.run(x)

        # use final state as output ?
        states_train.append(states[-1, np.newaxis])

        if plot:
            plot_states(timespan, states)
            return

    print("Fitting")
    readout.fit(states_train, Y_train)

    Y_pred = []
    print("Predicting")
    for x in X_test:
        states = reservoir.run(x)

        y = readout.run(states[-1, np.newaxis])
        Y_pred.append(y)

    Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
    Y_test_class = [np.argmax(y_t) for y_t in Y_test]

    # Calculate prediction accuracy
    score = accuracy_score(Y_test_class, Y_pred_class)
    print("Accuracy: ", f"{score * 100:.3f} %")

if __name__ == "__main__":
    # transduction(use_oscillator=True)
    classification(use_oscillator=True)

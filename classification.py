from reservoirpy.nodes import Reservoir, Ridge, Input
# from reservoirpy.datasets import japanese_vowels
from sklearn.metrics import accuracy_score

from oscillator.preprocessing import load_ecg
from oscillator.oscillator import Oscillator

import numpy as np
import reservoirpy as rpy

# import matplotlib.pyplot as plt

SEED = 1337

rpy.verbosity(0)
rpy.set_seed(SEED)

def classify(use_oscillator=True):
    # Load dataset
    # X_train, Y_train, X_test, Y_test = japanese_vowels()
    X_train, Y_train, X_test, Y_test = load_ecg(class_size=10)

    # Initialise genetic oscillator node
    oscillator = Oscillator(name="genetic-oscillator-1")

    # Example of updating the node's hyperparameters
    oscillator.hypers.update({'period': 25})

    # Initialise other nodes
    reservoir = Reservoir(500, sr=0.9, lr=0.1)
    readout = Ridge(ridge=1e-6)

    # Use the oscillator node as the reservoir if the flag is set
    if use_oscillator:
        reservoir = oscillator

    print("Training States")
    states_train = []
    for i, x in enumerate(X_train):
        print(i)
        states = reservoir.run(x, reset=True)
        states_train.append(states[-1, np.newaxis])

    print("Fitting Readout Layer")
    readout.fit(states_train, Y_train)

    Y_pred = []
    print("Predicting from Reservoir")
    for x in X_test:
        states = reservoir.run(x, reset=True)
        y = readout.run(states[-1, np.newaxis])
        Y_pred.append(y)

    Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
    Y_test_class = [np.argmax(y_t) for y_t in Y_test]

    # Calculate prediction accuracy
    score = accuracy_score(Y_test_class, Y_pred_class)
    print("Accuracy: ", f"{score * 100:.3f} %")

if __name__ == "__main__":
    classify(use_oscillator=True)

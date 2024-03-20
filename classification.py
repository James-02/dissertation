from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.datasets import japanese_vowels
from sklearn.metrics import accuracy_score

from preprocessing import load_ecg
from oscillator.oscillator import Oscillator

import numpy as np
import reservoirpy as rpy
import time

from visualisation import *

SEED = 1337

rpy.verbosity(0)
rpy.set_seed(SEED)

def transduction(use_oscillator=True):
    X_train, Y_train, X_test, Y_test = load_ecg(class_size=10, repeat_targets=True)
    reservoir = Oscillator(timesteps=X_train[0].shape[0], name="genetic-oscillator-1")

    if not use_oscillator:
        reservoir = Reservoir(500, sr=0.9, lr=0.1)

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

def classification(use_oscillator=True):
    # Load dataset
    print("Loading Dataset")
    # X_train, Y_train, X_test, Y_test = japanese_vowels()
    # X_train =  [np.sin(np.linspace(0, 8 * np.pi, 187)).reshape(-1, 1)] * 2

    X_train, Y_train, X_test, Y_test = load_ecg(class_size=100)
    timespan = np.linspace(0, 187, 187)

    # Initialise genetic oscillator node
    reservoir = Oscillator(timesteps=X_train[0].shape[0], name="genetic-oscillator-1")

    # Initialise other nodes
    readout = Ridge(ridge=1e-6)

    # Use the oscillator node as the reservoir if the flag is set
    if not use_oscillator:
        reservoir = Reservoir(500, sr=0.9, lr=0.1)

    print("Training States")
    states_train = []

    for i, x in enumerate(X_train):
        # start timer
        start = time.time()
    
        # train reservoir
        states = reservoir.run(x)
    
        # end timer
        end = time.time()
        print(str(i) + " Time: " + str(end - start))

        # use final state as output ?
        states_train.append(states[-1, np.newaxis])

    print("Fitting Readout Layer")
    readout.fit(states_train, Y_train)

    Y_pred = []
    print("Predicting from Reservoir")
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

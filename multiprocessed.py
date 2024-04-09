from utils.preprocessing import DataLoader
from utils.classification import Classifier
from utils.visualisation import Visualizer
from reservoir.reservoir import OscillatorReservoir
from reservoirpy.nodes import Reservoir, Ridge

import reservoirpy as rpy
import numpy as np
import os
import multiprocessing

# Define global constants
NODES = 10
SEED = 1337
VERBOSITY = 1
LOG_LEVEL = 1

OSCILLATOR_RESERVOIR = "oscillator_reservoir"
NEURON_RESERVOIR = "neuron_reservoir"

# set global config
rpy.set_seed(SEED)
rpy.verbosity(VERBOSITY)
np.random.seed(SEED)

def classify_with_reservoir(units, delay, sr, initial_conditions, use_oscillators, reservoir_name, X_train, Y_train, X_test, Y_test, log_file):
    process_id = os.getpid()
    unique_log_file = f"{log_file}_{process_id}"
    
    if use_oscillators:
        reservoir = OscillatorReservoir(
            units=units,
            timesteps=X_train[0].shape[0],
            delay=delay,
            sr=sr,
            initial_values=initial_conditions,
            seed=SEED,
            name=reservoir_name)
    else:
        reservoir = Reservoir(
            units=units, 
            sr=0.9, lr=0.1,
            seed=SEED,
            name=reservoir_name)
    
    # Initialize readout node
    readout = Ridge(ridge=1e-5)

    # Initialize classifier
    classifier = Classifier(
        reservoir=reservoir,
        readout=readout,
        train_set=(X_train, Y_train), 
        test_set=(X_test, Y_test),
        log_level=LOG_LEVEL,
        log_file=unique_log_file)
    
    classifier.log_params()

    # Perform classification
    metrics = classifier.classify( 
        save_states=True, 
        load_states=False)

    classifier.log_metrics(metrics)

def main(use_oscillators: bool = True, plot_states: bool = True, plot_distribution: bool = False):
    reservoir_name = OSCILLATOR_RESERVOIR if use_oscillators else NEURON_RESERVOIR
    log_file = f"logs/{reservoir_name}_{NODES}_nodes.log"
    data_loader = DataLoader(
        log_level=LOG_LEVEL,
        log_file=log_file)

    X_train, Y_train, X_test, Y_test = data_loader.load_ecg_data(
        rows=80,
        test_ratio=0.2,
        normalize=True,
        encode_labels=True,
        repeat_targets=False)

    # Log dataset information
    data_loader.log_dataset_info(X_train, Y_train, X_test, Y_test)

    # Initialize multiple Reservoirs using multiprocessing
    num_processes = 15  # Define the number of processes
    processes = []

    delay = 7
    sr = 1.0
    initials = [300, 300, 0, 0]

    delays = [1, 3, 5, 7, 9]
    srs = [0.1, 0.3, 0.5, 0.7, 0.9]
    initial_conditions = [[100, 0, 0, 0], [100, 100, 0, 0], [0, 100, 0, 0], [200, 200, 0, 0], [300, 300, 0, 0]]

    for i in range(num_processes):
        units = NODES
        if i < num_processes / 3:  # Cycling through delays
            delay = delays[i % len(delays)]  # Cycling through delays
        elif i < 2 * num_processes / 3:  # Cycling through sr values
            initials = initial_conditions[i % len(initial_conditions)]  # Cycling through initial conditions
            sr = srs[i % len(srs)]  # Cycling through sr values
        else:  # Cycling through initial conditions
            initials = initial_conditions[i % len(initial_conditions)]  # Cycling through initial conditions

        process = multiprocessing.Process(
            target=classify_with_reservoir,
            args=(units, delay, sr, initials, use_oscillators, reservoir_name, X_train, Y_train, X_test, Y_test, log_file)
        )
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    # Join all processes
    for process in processes:
        process.join()

if __name__ == "__main__":
    main(True, False, False)

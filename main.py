from utils.preprocessing import DataLoader
from utils.classification import Classifier
from utils.visualisation import Visualizer
from reservoir.reservoir import OscillatorReservoir
from reservoirpy.nodes import Reservoir, Ridge

import reservoirpy as rpy
import numpy as np

# Define global constants
NODES = 100
SEED = 1337
VERBOSITY = 1
LOG_LEVEL = 1

OSCILLATOR_RESERVOIR = "oscillator_reservoir"
NEURON_RESERVOIR = "neuron_reservoir"

# set global config
rpy.set_seed(SEED)
rpy.verbosity(VERBOSITY)
np.random.seed(SEED)

def main(use_oscillators: bool = True, plot_states: bool = True, plot_distribution: bool = False):
    reservoir_name = OSCILLATOR_RESERVOIR if use_oscillators else NEURON_RESERVOIR
    log_file = f"logs/{reservoir_name}_{NODES}_nodes.log"
    data_loader = DataLoader(
        log_level=LOG_LEVEL,
        log_file=log_file)

    X_train, Y_train, X_test, Y_test = data_loader.load_ecg_data(
        rows=5000,
        test_ratio=0.2,
        normalize=True,
        encode_labels=True,
        repeat_targets=False)

    # Log dataset information
    data_loader.log_dataset_info(X_train, Y_train, X_test, Y_test)

    # Initialize Reservoir
    if use_oscillators:
        reservoir = OscillatorReservoir(
            units=NODES,
            timesteps=X_train[0].shape[0],
            delay=7,
            initial_values=[300, 300, 0, 0],
            seed=SEED,
            name=reservoir_name)
    else:
        reservoir = Reservoir(
            units=NODES, 
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
        log_file=log_file)

    # log the classification parameters
    classifier.log_params()

    # Initialize visualizer object
    visualizer = Visualizer(results_path="results/", style="whitegrid", dpi=800)

    # Plot data distribution
    if plot_distribution:
        labels = ['Normal', 'Unknown', 'Ventricular ectopic', 'Supraventricular ectopic', 'Fusion']
        train_counts = data_loader._get_label_counts(Y_train).values()
        test_counts = data_loader._get_label_counts(Y_test).values()
        visualizer.plot_data_distribution(counts=train_counts, labels=labels, filename="train_distribution", show=False)
        visualizer.plot_data_distribution(counts=test_counts, labels=labels, filename="test_distribution", show=False)

    # Plot states if set
    if plot_states:
        states = reservoir.run(X_train[0])
        node_labels = [f"Node: {i}" for i in range(NODES)]
        visualizer.plot_states(states, node_labels, legend=False)
        return

    # Perform classification
    metrics = classifier.classify(
        processes=0, 
        save_states=True, 
        load_states=True)

    # Log classification metrics
    classifier.log_metrics(metrics)

if __name__ == "__main__":
    main(use_oscillators=True, plot_states=True)

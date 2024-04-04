from enum import Enum

from utils.preprocessing import DataLoader
from utils.classification import Classifier
from utils.visualisation import Visualizer
from reservoir.reservoir import OscillatorReservoir
from reservoirpy.nodes import Reservoir, Ridge

class LogLevel(Enum):
    NONE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

class ReservoirType(Enum):
    OSCILLATOR = "oscillator"
    NEURONS = "neurons"

reservoir_config = {
    "reservoir_type": ReservoirType.OSCILLATOR,
    "nodes": 10,
}

data_loader_config = {
    "rows": 2500,
    "test_ratio": 0.2,
    "encode_labels": True,
    "normalize": True,
    "repeat_targets": False,
    "data_dir": 'data/ecg'
}

classifier_config = {
    "ridge": 1e-5,
    "use_multiprocessing": True,
    "save_states": True,
    "load_states": True,
}

visualizer_config = {
    "results_path": "results/",
    "dpi": 500,
    "plot_states": False,
    "plot_distribution": False,
}

global_config = {
    "seed": 1337,
    "log_level": LogLevel.DEBUG.value,
    "log_file": None
}

def main():
    # Load data
    data_loader = DataLoader(
        log_level=global_config['log_level'],
        log_file=global_config['log_file'],
        seed=global_config['seed']
    )

    X_train, Y_train, X_test, Y_test = data_loader.load_ecg_data(
        rows=data_loader_config['rows'],
        test_ratio=data_loader_config['test_ratio'],
        normalize=data_loader_config['normalize'],
        encode_labels=data_loader_config['encode_labels'],
        repeat_targets=data_loader_config['repeat_targets']
    )

    # Log dataset information
    data_loader.log_dataset_info(X_train, Y_train, X_test, Y_test)

    # Initialize Reservoir
    if reservoir_config['reservoir_type'] == ReservoirType.OSCILLATOR:
        reservoir = OscillatorReservoir(
            units=reservoir_config['nodes'], 
            timesteps=X_train[0].shape[0]
        )
    else:
        reservoir = Reservoir(
            units=reservoir_config['nodes'], 
            sr=0.9, lr=0.1
        )
    
    # Initialize readout node
    readout = Ridge(ridge=classifier_config['ridge'])

    # Initialize classifier
    classifier = Classifier(
        reservoir=reservoir,
        readout=readout,
        train_set=(X_train, Y_train), 
        test_set=(X_test, Y_test),
        log_level=global_config['log_level'],
        log_file=global_config['log_file'],
        seed=global_config['seed']
    )

    # Initialize visualizer object
    visualizer = Visualizer(
        results_path=visualizer_config['results_path'],
        dpi=visualizer_config['dpi']
    )

    # Plot data distribution
    if visualizer_config['plot_distribution']:
        labels = ['Normal', 'Unknown', 'Ventricular ectopic', 'Supraventricular ectopic', 'Fusion']
        train_counts = data_loader._get_label_counts(Y_train).values()
        test_counts = data_loader._get_label_counts(Y_test).values()
        visualizer.plot_data_distribution(counts=train_counts, labels=labels, filename="train_distribution", show=False)
        visualizer.plot_data_distribution(counts=test_counts, labels=labels, filename="test_distribution", show=False)

    # Plot states if set
    if visualizer_config['plot_states']:
        states = reservoir.run(X_train[0])
        node_labels = [f"Node: {i}" for i in range(reservoir_config['nodes'])]
        visualizer.plot_states(states, node_labels, legend=False)

    # Perform classification
    metrics = classifier.classify(
        use_multiprocessing=classifier_config['use_multiprocessing'], 
        save_states=classifier_config['save_states'], 
        load_states=classifier_config['load_states']
    )

    # Log classification metrics
    classifier.log_metrics(metrics)

if __name__ == "__main__":
    main()

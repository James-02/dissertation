from utils.preprocessing import DataLoader
from utils.classification import Classifier
from utils.visualisation import Visualizer
from reservoir.reservoir import OscillatorReservoir
from reservoirpy.nodes import Reservoir, Ridge

from config import GLOBAL_CONFIG, RESERVOIR_CONFIG, DATA_LOADER_CONFIG, CLASSIFIER_CONFIG, VISUALIZER_CONFIG

def main():
    data_loader = DataLoader(
        log_level=GLOBAL_CONFIG.log_level,
        log_file=GLOBAL_CONFIG.log_file,
        seed=GLOBAL_CONFIG.seed
    )

    X_train, Y_train, X_test, Y_test = data_loader.load_ecg_data(
        rows=DATA_LOADER_CONFIG.rows,
        test_ratio=DATA_LOADER_CONFIG.test_ratio,
        normalize=DATA_LOADER_CONFIG.normalize,
        encode_labels=DATA_LOADER_CONFIG.encode_labels,
        repeat_targets=DATA_LOADER_CONFIG.repeat_targets
    )

    # Log dataset information
    data_loader.log_dataset_info(X_train, Y_train, X_test, Y_test)

    # Initialize Reservoir
    if RESERVOIR_CONFIG.reservoir_type == "oscillator":
        reservoir = OscillatorReservoir(
            units=RESERVOIR_CONFIG.nodes, 
            timesteps=X_train[0].shape[0]
        )
    else:
        reservoir = Reservoir(
            units=RESERVOIR_CONFIG.nodes, 
            sr=0.9, lr=0.1
        )
    
    # Initialize readout node
    readout = Ridge(ridge=CLASSIFIER_CONFIG.ridge)

    # Initialize classifier
    classifier = Classifier(
        reservoir=reservoir,
        readout=readout,
        train_set=(X_train, Y_train), 
        test_set=(X_test, Y_test),
        log_level=GLOBAL_CONFIG.log_level,
        log_file=GLOBAL_CONFIG.log_file,
        seed=GLOBAL_CONFIG.seed
    )

    # Initialize visualizer object
    visualizer = Visualizer(
        results_path=VISUALIZER_CONFIG.results_path,
        dpi=VISUALIZER_CONFIG.dpi
    )

    # Plot data distribution
    if VISUALIZER_CONFIG.plot_distribution:
        labels = ['Normal', 'Unknown', 'Ventricular ectopic', 'Supraventricular ectopic', 'Fusion']
        train_counts = data_loader._get_label_counts(Y_train).values()
        test_counts = data_loader._get_label_counts(Y_test).values()
        visualizer.plot_data_distribution(counts=train_counts, labels=labels, filename="train_distribution", show=False)
        visualizer.plot_data_distribution(counts=test_counts, labels=labels, filename="test_distribution", show=False)

    # Plot states if set
    if VISUALIZER_CONFIG.plot_states:
        states = reservoir.run(X_train[0])
        node_labels = [f"Node: {i}" for i in range(RESERVOIR_CONFIG.nodes)]
        visualizer.plot_states(states, node_labels, legend=True)

    # Perform classification
    metrics = classifier.classify(
        use_multiprocessing=CLASSIFIER_CONFIG.use_multiprocessing, 
        save_states=CLASSIFIER_CONFIG.save_states, 
        load_states=CLASSIFIER_CONFIG.load_states
    )

    # Log classification metrics
    classifier.log_metrics(metrics)

if __name__ == "__main__":
    main()

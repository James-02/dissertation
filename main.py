import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
import reservoirpy as rpy

from utils.logger import Logger
from utils.preprocessing import load_ecg_data, load_npz, load_ecg_forecast
from utils.classification import classify
from utils.visualisation import plot_states, plot_data_distribution, \
    plot_dataset_info, plot_average_instance, plot_class_std, \
    plot_class_mean, plot_confusion_matrix, plot_metrics_across_folds, \
    plot_metrics_table, plot_class_metrics, plot_tsne_clustering, plot_forecast_data, plot_forecast_results
from reservoir.reservoir import OscillatorReservoir, Oscillator

# Define global constants
SEED = 1337
VERBOSITY = 1
LOG_LEVEL = 1

def set_global_config():
    """Set global configurations."""
    rpy.set_seed(SEED)
    rpy.verbosity(VERBOSITY)
    np.random.seed(SEED)

def plot_full_ecg():
    """Plot full ECG waveform."""
    timesteps = 2000
    X_train, _, _, _ = load_ecg_forecast(timesteps=timesteps)
    sampling_freq = 125
    num_samples = len(X_train)
    time_ms = np.arange(num_samples) / sampling_freq * 1000
    
    # Plot the ECG waveform
    plt.figure(figsize=(14, 8))
    plt.plot(time_ms, X_train, label='ECG waveform')

    # Calculate box width in ms
    box_width = 187 * 1000 / sampling_freq
    start_x = 4500

    # Plot the box
    plt.plot([start_x, start_x + box_width], [min(X_train), min(X_train)], linestyle=':', color='red', label="Sampled Segment")
    plt.plot([start_x, start_x + box_width], [max(X_train), max(X_train)], linestyle=':', color='red')
    plt.plot([start_x, start_x], [min(X_train), max(X_train)], linestyle=':', color='red')
    plt.plot([start_x + box_width, start_x + box_width], [min(X_train), max(X_train)], linestyle=':', color='red')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.show()

def plot_dde_states():
    """Plot DDE states."""
    timesteps = np.linspace(0, 1000, 1000)
    sine_wave = np.sin(timesteps)
    oscillator = Oscillator(timesteps=1000, delay=10, initial_values=[0, 100, 0, 0])
    states = []
    for i in range(1000):
        states.append(oscillator.forward(sine_wave[i] * 1e-5))
    plot_states(np.array(states), labels=["A", "I", "Hi", "He"], ylabel="Concentration", title="", legend=True, show=True)
    
def plot_reservoir_states(reservoir, X_train, iterations=1):
    """Plot reservoir states."""
    labels = [f"Node: {i}" for i in range(reservoir.units)]
    for x in X_train[:iterations]:
        plot_states(reservoir.run(x), labels, legend=False)

def analyse_dataset(X_train, Y_train, X_test, Y_test):
    """Analyse dataset."""
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)

    plot_data_distribution(Y, filename="binary-data-distribution.png", show=True)
    plot_dataset_info(X_train, Y_train, X_test, Y_test, filename="binary-dataset-info.png", show=True)
    plot_average_instance(X, Y, filename="binary-average-instance", show=True)
    plot_class_std(X, Y, filename="binary-std.png", show=True)
    plot_class_mean(X, Y, filename="binary-means.png", show=True)

def initialize_reservoir(timesteps: int, use_oscillators: bool = True, **kwargs):
    """Initialize reservoir."""

    if use_oscillators:
        oscillator_kwargs = {'delay': kwargs.get('delay'), 'initial_conditions': kwargs.get('initial_conditions')}

        return OscillatorReservoir(
            units=kwargs.get('nodes'),
            timesteps=timesteps,
            sr=kwargs.get('sr'),
            W=W,
            noise_rc=kwargs.get('noise_rc'),
            rc_scaling=kwargs.get('rc_scaling'),
            seed=SEED,
            node_kwargs=oscillator_kwargs)

    return Reservoir(units=kwargs.get('nodes'), sr=kwargs.get('sr'))

def initialize_readout(**kwargs):
    """Initialize readout."""
    return Ridge(ridge=kwargs.get('ridge', 1e-7))

def classification(use_oscillators: bool = True, analyse_data: bool = False, plot_states: bool = False, **kwargs):
    """Perform classification."""
    X_train, Y_train, X_test, Y_test = load_ecg_data(
        rows=kwargs.get('instances', 500),
        test_ratio=kwargs.get('test_ratio', 0.2),
        normalize=kwargs.get('normalize', True),
        shuffle=kwargs.get('shuffle', True),
        encode_labels=kwargs.get('encode_labels', True),
        repeat_targets=kwargs.get('repeat_targets', False),
        binary=kwargs.get('binary', False))

    if analyse_data:
        analyse_dataset(X_train, Y_train, X_test, Y_test)
        return

    reservoir = initialize_reservoir(X_train[0].shape[0], use_oscillators, **kwargs)
    readout = initialize_readout(**kwargs)

    if plot_states:
        plot_reservoir_states(reservoir, X_train)
        return

    # Perform classification
    binary_tag = "binary" if kwargs.get('binary', False) else ""
    filename = f"results/runs/{reservoir.name}-{kwargs.get('nodes', 2)}-{len(X_train)}-{binary_tag}"
    classify(reservoir, readout, X_train, Y_train, X_test, Y_test, folds=kwargs.get('folds', None), save_file=filename)

    if not kwargs.get('folds'):
        data = load_npz(filename + ".npz", allow_pickle=True)
        if data is not None:
            metrics = data['metrics'].item()
            plot_tsne_clustering(data['Y_predicted'], show=True)
            plot_class_metrics(metrics['class_metrics'])
            plot_confusion_matrix(metrics['confusion_matrix'])
        else:
            logger.error(f"Model data does not exist at: {filename}")
    else:
        metrics = []
        for fold in range(kwargs.get('folds')):
            filename = filename + f"-fold-{str(fold)}.npz"
            fold_data = load_npz(filename, allow_pickle=True)
            fold_metrics = fold_data['metrics'].item()
            if fold_data is not None:
                metrics.append(fold_metrics)
            else:
                logger.error(f"Fold data does not exist at: {filename}")
        plot_metrics_across_folds(metrics)

def forecasting(use_oscillators: bool = True, **kwargs):
    """Perform forecasting."""
    timesteps = kwargs.get('timesteps', 2510)
    X_train, Y_train, X_test, Y_test = load_ecg_forecast(
        timesteps=timesteps, 
        forecast=kwargs.get("forecast"), 
        test_ratio=kwargs.get("test_ratio"))

    reservoir = initialize_reservoir(X_train[0].shape[0], use_oscillators, **kwargs)
    readout = initialize_readout(**kwargs)

    plot_forecast_data(X_train, Y_train, X_test, Y_test)

    esn = reservoir >> readout
    esn.fit(X_train, Y_train)
    Y_pred = esn.run(X_test)

    mse = np.mean((Y_test - Y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"MSE {mse}, RMSE: {rmse}")

    plot_forecast_results(Y_pred, Y_test)


if __name__ == "__main__":
    set_global_config()

    model_params = {
        'nodes': 100,
        'sr': 0.9,
        'delay': 10,
        'initial_conditions': [200, 200, 0, 0],
        'coupling': 1e-4,
        'rc_scaling': 4e-7,
        'input_connectivity': 0.1,
        'rc_connectivity': 0.1,
        'noise_rc': 0.1,
        'ridge': 1e-5,
        'folds': None,
    }

    data_params = {
        'instances': 500,
        'normalize': True,
        'shuffle': True,
        'encode_labels': True,
        'repeat_targets': False,
        'binary': False,
        'test_ratio': 0.2,
        'forecast': 1,
    }

    log_file = f"logs/{model_params['nodes']}_nodes.log"
    logger = Logger(level=LOG_LEVEL, log_file=log_file)

    # perform classification
    classification(plot_states=False, **{**data_params, **model_params})

    # perform forecasting
    # forecasting(**{**data_params, **model_params})

    # plot the system of delay differential equations
    # plot_dde_states()

    # plot an example of the dataset
    # plot_full_ecg()

from multiprocessing import Pool

import time
import numpy as np
import reservoirpy as rpy

from reservoirpy.nodes import Reservoir, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from reservoir.reservoir import OscillatorReservoir
from utils.preprocessing import DataLoader
from utils.visualisation import plot_states
from utils.logger import Logger

SEED = 1337
rpy.set_seed(SEED)
rpy.verbosity(1)

def evaluation(Y_true, Y_pred):
    """
    Evaluate the classification performance.

    Args:
        Y_true (numpy.ndarray): True labels.
        Y_pred (numpy.ndarray): Predicted labels.

    Returns:
        dict: Dictionary containing performance metrics.
    """
    accuracy = accuracy_score(Y_true, Y_pred) * 100
    f1 = f1_score(Y_true, Y_pred, average='weighted')
    recall = recall_score(Y_true, Y_pred, average='weighted')
    precision = precision_score(Y_true, Y_pred, average='weighted')
    mse = np.mean((Y_true - Y_pred) ** 2)
    conf_matrix = confusion_matrix(Y_true, Y_pred)

    return {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision,
            "mse": mse, "confusion_matrix": conf_matrix}

def __train_reservoir(args):
    """
    Train the reservoir with input data.

    Args:
        x_reservoir_tuple (tuple): Tuple containing input data and reservoir instance.

    Returns:
        numpy.ndarray: Trained states.
    """
    x, reservoir = args
    return reservoir.run(x)

def __predict_reservoir(args):
    """
    Predict using the reservoir and readout.

    Args:
        x_reservoir_readout_tuple (tuple): Tuple containing input data, reservoir instance, and readout instance.

    Returns:
        numpy.ndarray: Predicted states.
    """
    x, reservoir, readout = args
    states = reservoir.run(x)
    return readout.run(states[-1, np.newaxis])

def train(X_train, reservoir, use_multiprocessing):
    """
    Train the reservoir on the training data.

    Args:
        X_train (list): List of training data.
        reservoir (Reservoir): Reservoir instance.
        use_multiprocessing (bool): Flag indicating whether to use multiprocessing.

    Returns:
        list: List of trained states.
    """
    if use_multiprocessing:
        with Pool() as pool:
            trained_states = pool.map(__train_reservoir, [(x, reservoir) for x in X_train])
    else:
        trained_states = [__train_reservoir((x, reservoir)) for x in X_train]
    return [state[-1, np.newaxis] for state in trained_states]

def predict(X_test, reservoir, readout, use_multiprocessing):
    """
    Predict using the trained reservoir and readout on test data.

    Args:
        X_test (list): List of test data.
        reservoir (Reservoir): Trained reservoir instance.
        readout (Ridge): Readout instance.
        use_multiprocessing (bool): Flag indicating whether to use multiprocessing.

    Returns:
        list: List of predicted states.
    """
    if use_multiprocessing:
        with Pool() as pool:
            predicted_states = pool.map(__predict_reservoir, [(x, reservoir, readout) for x in X_test])
        return predicted_states
    else:
        return [__predict_reservoir((x, reservoir, readout)) for x in X_test]

def log_metrics(metrics: dict):
    """Log performance metrics as a classification report"""
    logger.info("----- Classification Report -----")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}%")
    logger.info(f"MSE: {metrics['mse']:.3f}")
    logger.info(f"Recall: {metrics['recall']:.3f}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"F1: {metrics['f1']:.3f}")
    logger.info("---------------------------------")

def classification(use_oscillator=True, use_multiprocessing=True, plot=False):
    """
    Perform classification using reservoir computing.

    Args:
        use_oscillator (bool): Flag indicating whether to use oscillator reservoir.
        use_multiprocessing (bool): Flag indicating whether to use multiprocessing.
        plot (bool): Flag indicating whether to plot a single instance's states and return.

    Returns:
        dict: Dictionary containing performance metrics.
    """
    # Load dataset
    data_loader = DataLoader()
    X_train, Y_train, X_test, Y_test = data_loader.load_ecg_data(rows=100, test_ratio=0.3, normalize=True, encode_labels=True)

    data_loader.log_dataset_info(X_train, Y_train, X_test, Y_test)

    # Use the oscillator node as the reservoir if the flag is set
    nodes = 10
    timesteps = X_train[0].shape[0]
    reservoir = OscillatorReservoir(units=nodes, timesteps=timesteps) if not use_oscillator else Reservoir(100, sr=0.9, lr=0.1)

    # Initialize reservoir and readout
    readout = Ridge(ridge=1e-5)

    # Training
    if plot:
        logger.info("Plotting states of X_train[0]")
        plot_states(reservoir.run(X_train[0]))
        return

    logger.info(f"Training Reservoir of {nodes} nodes with {len(X_train)} instances")
    start = time.time()
    states_train = train(X_train, reservoir, use_multiprocessing)
    end = time.time()
    logger.debug(f"Training Time Elapsed: {str(round(end - start, 4))}s")

    # Fitting
    logger.info(f"Fitting readout layer with {len(states_train)} states")
    readout.fit(states_train, Y_train)

    # Predicting
    logger.info(f"Predicting with {len(X_test)} instances.")
    start = time.time()
    Y_pred = predict(X_test, reservoir, readout, use_multiprocessing)
    end = time.time()
    logger.debug(f"Prediction Time Elapsed: {str(round(end - start, 4))}s")

    # Calculate performance metrics
    logger.info("Calculating model performance metrics")
    Y_test_class = np.array([np.argmax(y_t) for y_t in Y_test])
    Y_pred_class = np.array([np.argmax(y_p) for y_p in Y_pred])
    metrics = evaluation(Y_test_class, Y_pred_class)

    # produce classification report
    log_metrics(metrics)

if __name__ == "__main__":
    logger = Logger(name="classification", level=1)
    classification(use_oscillator=True, use_multiprocessing=True, plot=False)

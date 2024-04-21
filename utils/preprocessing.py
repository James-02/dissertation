from typing import Tuple, List
import os

import numpy as np
import pandas as pd

from utils.logger import Logger
from utils.analysis import count_labels, log_params

from reservoirpy.datasets import to_forecasting, mackey_glass

DEFAULT_LOG_LEVEL = 1

logger = Logger()

def save_npz(filename: str, **kwargs) -> None:
    np.savez(filename, **kwargs)

def load_npz(filename: str, allow_pickle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return np.load(filename, allow_pickle=allow_pickle)
    except Exception as e:
        logger.error(f"Error loading data from: {filename}\n{e}")
        return None
    
def load_mackey_glass(timesteps=2510, test_ratio=0.2, tau=17):
    X = mackey_glass(timesteps, tau=tau)

    train_size = int(len(X) * (1 - test_ratio))

    # Rescale between -1 and 1
    X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

    # Prepare forecasting data
    X, Y = to_forecasting(X, forecast=10)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]
    return X_train, Y_train, X_test, Y_test

def load_ecg_forecast(timesteps=1000, forecast=10, test_ratio=0.2):
    # Read the CSV file and extract the 'MLII' column
    df = pd.read_csv("data/ecg/mit-bih-100.csv")
    data = df[['MLII']].values

    # Use the to_forecasting function to create input-output pairs for forecasting
    X, Y = to_forecasting(data[:timesteps], forecast=forecast)

    # rescale between 1 and -1
    X, Y = _normalize(X), _normalize(Y)

    train_size = int(len(X) * (1 - test_ratio))

    # Split the data into training and testing sets
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    return X_train, Y_train, X_test, Y_test


def load_ecg_data(rows: int = None, test_ratio: float = 0.2, encode_labels: bool = True, normalize: bool = True, 
                    repeat_targets: bool = False, shuffle: bool = True, binary: bool = False,
                    data_dir: str = "data/ecg", train_file: str = "ecg_train.csv",
                    test_file: str = "ecg_test.csv", save_file: str = "ecg_data.npz",
                    binary_save_file: str = "binary_ecg_data.npz") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECG dataset, preprocess, balance classes, and split into training and testing subsets.

    Args:
        rows (int, optional): Number of rows to limit the dataset to, defaults to all rows.
        test_ratio (float, optional): Ratio of testing data to total data. Defaults to 0.2.
        encode_labels (bool, optional): Whether to one-hot encode labels. Defaults to True.
        normalize (bool, optional): Whether to normalize the input values. Defaults to True.
        repeat_targets (bool, optional): Whether to repeat targets. Defaults to False.
        shuffle (bool, optional): Whether to shuffle instances. Defaults to True.
        binary (bool, optional): Whether to load dataset as binary targets. Defaults to False.
        data_dir (str, optional): Directory containing ECG data files. Defaults to "ecg".
        train_file (str, optional): Filename of the training data file. Defaults to "ecg_train.csv".
        test_file (str, optional): Filename of the testing data file. Defaults to "ecg_test.csv".
        save_file (str, optional): Filename to save preprocessed data. Defaults to "ecg_data.npz".
        binary_save_file (str, optional): Filename to save preprocessed data with binary targets. Defaults to "binary_ecg_data.npz".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing X_train, Y_train, X_test, Y_test.
    """
    train_file_path = os.path.join(data_dir, train_file)
    test_file_path = os.path.join(data_dir, test_file)
    save_file_path = os.path.join(data_dir, save_file)
    save_file_path_binary = os.path.join(data_dir, binary_save_file)

    logger.info("Loading ECG Dataset")
    X_loaded, Y_loaded = _load_ecg_dataset(binary, train_file_path, test_file_path, save_file_path, save_file_path_binary)

    # If rows is not given, set to all rows
    if rows is None:
        rows = len(X_loaded)
        logger.warning("Number of rows not specified, implicitly using all rows.")

    if normalize:
        logger.debug("Normalizing inputs")
        X_loaded = _normalize(X_loaded)

    logger.debug("Balancing classes")
    X_balanced, Y_balanced = _balance_classes(X_loaded, Y_loaded)

    logger.debug(f"Limiting instances to {rows} rows")
    X_limited, Y_limited = _limit_instances(X_balanced, Y_balanced, rows)

    num_rows = X_limited.shape[0]
    if num_rows < rows:
        logger.warning(f"The {rows} rows requested were capped at {num_rows} rows to keep classes balanced.")

    logger.debug("Splitting dataset into training and testing sets")
    X_train, Y_train, X_test, Y_test = _train_test_split(X_limited, Y_limited, test_size=test_ratio, shuffle=shuffle)

    if encode_labels:
        num_classes = len(np.unique(Y_limited))
        logger.debug("One-hot encoding labels.")
        Y_train = _one_hot_encode(Y_train, num_classes)
        Y_test = _one_hot_encode(Y_test, num_classes)

    logger.debug("Reshaping data into time-series.")
    X_train, Y_train = _reshape_data(X_train, Y_train)
    X_test, Y_test = _reshape_data(X_test, Y_test)

    # Repeat targets if specified
    if repeat_targets:
        logger.debug(f"Repeating train and test targets to size: {X_train[0].shape[0]}")
        Y_train = [np.repeat(Y_instance, X_instance.shape[0], axis=0) for X_instance, Y_instance in zip(X_train, Y_train)]
        Y_test = [np.repeat(Y_instance, X_instance.shape[0], axis=0) for X_instance, Y_instance in zip(X_test, Y_test)]

    train_shapes = (X_train[0].shape, Y_train[0].shape)
    test_shapes = (X_test[0].shape, Y_test[0].shape)
    params = {"instances": num_rows,
              "encode_labels": encode_labels,
              "repeat_targets": repeat_targets,
              "normalize": normalize,
              "test_ratio": test_ratio,
              "shuffle": shuffle,
              "binary": binary,
              "train_instances": len(X_train),
              "test_instances": len(X_test),
              "train_shapes": train_shapes,
              "test_shapes": test_shapes,
              "train_labels": count_labels(Y_train),
              "test_labels": count_labels(Y_test),
            }

    log_params(params, title="Dataset Parameters")

    return X_train, Y_train, X_test, Y_test

def _normalize(X: np.ndarray) -> np.ndarray:
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return (X - min_vals) / (max_vals - min_vals)

def _balance_classes(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Y = Y.astype(int)
    classes, counts = np.unique(Y, return_counts=True)
    min_instances_per_class = min(counts)
    balanced_indices = [np.random.choice(np.where(Y == c)[0], min_instances_per_class, replace=False) for c in classes]
    return X[np.concatenate(balanced_indices)], Y[np.concatenate(balanced_indices)]

def _preprocess_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, binary: bool) -> Tuple[np.ndarray, np.ndarray]:
    # Concatenate train and test datasets
    X_combined = pd.concat([train_df.iloc[:, :-1], test_df.iloc[:, :-1]], axis=0)
    Y_combined = pd.concat([train_df.iloc[:, -1], test_df.iloc[:, -1]], axis=0)

    # If binary flag is enabled, merge arrhythmic classes into one
    if binary:
        logger.debug("Merging all arrhythmia variations into one target for binary classification")
        Y_combined[Y_combined != 0] = 1

    return X_combined.values, Y_combined.values

def _load_ecg_dataset(binary: bool, train_file_path: str, test_file_path: str, save_file_path: str, save_file_path_binary: str) -> Tuple[np.ndarray, np.ndarray]:
    file_path = save_file_path_binary if binary else save_file_path

    if os.path.exists(file_path):
        logger.debug(f"Loading preprocessed dataset from: {file_path}")
        data = load_npz(file_path)
        if data is None:
            logger.warning(f"Could not load dataset at: {file_path}")
        else:
            return data["X"], data["Y"]

    logger.debug(f"Preprocessing datasets from: {train_file_path} and {test_file_path}")
    train_df = pd.read_csv(train_file_path, header=None)
    test_df = pd.read_csv(test_file_path, header=None)

    X_loaded, Y_loaded = _preprocess_dataset(train_df, test_df, binary)
    save_npz(file_path, **{"X": X_loaded, "Y": Y_loaded})
    logger.debug(f"Saving preprocessed dataset to: {file_path}")

    return X_loaded, Y_loaded

def _train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float, shuffle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.debug(f"Train:Test dataset ratio: [{int((1 - test_size) * 100)}:{int(test_size * 100)}]")
    Y = Y.astype(int)
    classes, counts = np.unique(Y, return_counts=True)
    test_class_counts = np.floor(counts * test_size).astype(int)
    train_indices, test_indices = [], []

    for c, count in zip(classes, test_class_counts):
        class_indices = np.where(Y == c)[0]
        selected_indices = np.random.choice(class_indices, count, replace=False)
        test_indices.extend(selected_indices)
        train_indices.extend(np.setdiff1d(class_indices, selected_indices))

    if shuffle:
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    return X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]

def _one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[labels.astype(int)]

def _reshape_data(X: np.ndarray, Y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Reshape X to have shape (num_instances, timesteps, features)
    X_reshaped = [instance.T.reshape(-1, 1) for instance in X]

    # If Y is one-hot encoded, keep it as it is
    if len(Y.shape) > 1:
        Y_reshaped = [label.reshape(-1, 1) for label in Y]

    # Create a list of arrays for Y
    Y_reshaped = [np.array([label]) for label in Y]

    return X_reshaped, Y_reshaped

def _limit_instances(X: np.ndarray, Y: np.ndarray, rows: int) -> Tuple[np.ndarray, np.ndarray]:
    Y = Y.astype(int)
    classes, counts = np.unique(Y, return_counts=True)
    max_instances_per_class = int(rows / len(classes))
    limited_indices = [np.random.choice(np.where(Y == c)[0], min(count, max_instances_per_class), replace=False) for c, count in zip(classes, counts)]
    return X[np.concatenate(limited_indices)], Y[np.concatenate(limited_indices)]

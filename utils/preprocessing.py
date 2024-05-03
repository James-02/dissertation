from typing import Tuple, List, Optional
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.logger import Logger
from utils.analysis import count_labels, log_params
from reservoirpy.datasets import to_forecasting, mackey_glass

DEFAULT_LOG_LEVEL = 1

logger = Logger()

def save_npz(filename: str, **kwargs) -> None:
    """
    Save numpy arrays to a compressed file.

    Args:
        filename (str): Name of the file.
        **kwargs: Dictionary containing arrays to save.
    """
    np.savez(filename, **kwargs)

def load_npz(filename: str, allow_pickle: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load numpy arrays from a compressed file.

    Args:
        filename (str): Name of the file.
        allow_pickle (bool, optional): Allow loading pickled objects (default is True).

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Tuple containing loaded arrays if successful, None otherwise.
    """
    try:
        return np.load(filename, allow_pickle=allow_pickle)
    except Exception as e:
        logger.error(f"Error loading data from: {filename}\n{e}")
        return None

def load_mackey_glass(timesteps: int = 2510, test_ratio: float = 0.2, tau: int = 17) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess Mackey-Glass dataset for reservoir computing.

    Args:
        timesteps (int, optional): Number of timesteps in the dataset (default is 2510).
        test_ratio (float, optional): Ratio of test data to total data (default is 0.2).
        tau (int, optional): Time delay parameter for the dataset (default is 17).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing X_train, Y_train, X_test, Y_test.
    """
    X = mackey_glass(timesteps, tau=tau)

    train_size = int(len(X) * (1 - test_ratio))

    # Rescale between -1 and 1
    X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

    # Prepare forecasting data
    X, Y = to_forecasting(X, forecast=10)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]
    return X_train, Y_train, X_test, Y_test

def load_ecg_forecast(timesteps: int = 2000, forecast: int = 10, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess ECG (Electrocardiogram) dataset for forecasting.

    Args:
        timesteps (int, optional): Number of timesteps in the dataset (default is 1000).
        forecast (int, optional): Number of timesteps to forecast (default is 10).
        test_ratio (float, optional): Ratio of test data to total data (default is 0.2).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing X_train, Y_train, X_test, Y_test.
    """
    # Read the CSV file and extract the 'MLII' column
    df = pd.read_csv("data/ecg/mit-bih-100.csv")
    X = df[['MLII']].values[:timesteps]

    # Rescale between -1 and 1
    X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

    # Use the to_forecasting function to create input-output pairs for forecasting
    X, Y = to_forecasting(X, forecast=forecast)

    # Split the data into training and testing sets
    train_size = int(len(X) * (1 - test_ratio))
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]
    return X_train, Y_train, X_test, Y_test

def load_ecg_data(rows: Optional[int] = None, test_ratio: float = 0.2, encode_labels: bool = True, standardize: bool = True,
                    repeat_targets: bool = False, shuffle: bool = True, binary: bool = False, noise_rate: float = 0,
                    noise_ratio: float = 0, data_dir: str = "data/ecg", train_file: str = "ecg_train.csv",
                    test_file: str = "ecg_test.csv", save_file: str = "ecg_data.npz",
                    binary_save_file: str = "binary_ecg_data.npz") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECG dataset from file, encode labels, shuffle, augment, scale, balance classes, and split into training and testing subsets.

    Args:
        rows (int, optional): Number of rows to limit the dataset to, defaults to the maximum.
        test_ratio (float, optional): Ratio of testing data to total data. Defaults to 0.2.
        encode_labels (bool, optional): Whether to one-hot encode labels. Defaults to True.
        standardize (bool, optional): Whether to standardize the input values. Defaults to True.
        repeat_targets (bool, optional): Whether to repeat targets. Defaults to False.
        shuffle (bool, optional): Whether to shuffle instances. Defaults to True.
        binary (bool, optional): Whether to load dataset as binary targets. Defaults to False.
        noise_rate (float, optional): Rate of noise augmentation. Defaults to 0.
        noise_ratio (float, optional): Ratio of noise augmentation. Defaults to 0.
        data_dir (str, optional): Directory containing ECG data files. Defaults to "data/ecg".
        train_file (str, optional): Filename of the training data file. Defaults to "ecg_train.csv".
        test_file (str, optional): Filename of the testing data file. Defaults to "ecg_test.csv".
        save_file (str, optional): Filename to save the preprocessed data. Defaults to "ecg_data.npz".
        binary_save_file (str, optional): Filename to save the preprocessed data with binary targets. Defaults to "binary_ecg_data.npz".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing X_train, Y_train, X_test, Y_test.
    """
    train_file_path = os.path.join(data_dir, train_file)
    test_file_path = os.path.join(data_dir, test_file)
    save_file_path = os.path.join(data_dir, save_file)
    save_file_path_binary = os.path.join(data_dir, binary_save_file)

    logger.info("Loading ECG Dataset")
    X_loaded, Y_loaded = _load_raw_ecg_dataset(binary, train_file_path, test_file_path, save_file_path, save_file_path_binary)

    # If rows is not given, set to all rows
    if rows is None:
        rows = len(X_loaded)
        logger.warning("Number of rows not specified, implicitly using all rows.")

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
        num_classes = len(np.unique(Y_loaded))
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

    # add augmentation (noise) to training set
    if noise_rate and noise_ratio:
        X_train, Y_train = augment_data(np.array(X_train), np.array(Y_train), noise_rate, noise_ratio)

    if standardize:
        X_train, X_test = standardize_data(X_train, X_test)

    train_shapes = (X_train[0].shape, Y_train[0].shape)
    test_shapes = (X_test[0].shape, Y_test[0].shape)

    params = {
        "instances": num_rows,
        "encode_labels": encode_labels,
        "repeat_targets": repeat_targets,
        "standardize": standardize,
        "test_ratio": test_ratio,
        "shuffle": shuffle,
        "binary": binary,
        "noise_rate": noise_rate,
        "noise_ratio": noise_ratio,
        "train_instances": len(X_train),
        "test_instances": len(X_test),
        "train_shapes": train_shapes,
        "test_shapes": test_shapes,
        "train_labels": count_labels(Y_train),
        "test_labels": count_labels(Y_test),
    }

    log_params(params, title="Dataset Parameters")

    return X_train, Y_train, X_test, Y_test

def standardize_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize input data using Min-Max scaling.

    Args:
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Testing data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Standardized training and testing data.
    """
    # Convert into correct shape
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Initialize scaler and fit on training data
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # Transform training and testing data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape back to original shape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test

def _balance_classes(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance classes by undersampling.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Balanced input data and target labels.
    """
    Y = Y.astype(int)
    classes, counts = np.unique(Y, return_counts=True)
    min_instances_per_class = min(counts)
    balanced_indices = [np.random.choice(np.where(Y == c)[0], min_instances_per_class, replace=False) for c in classes]
    return X[np.concatenate(balanced_indices)], Y[np.concatenate(balanced_indices)]

def augment_data(X: np.ndarray, Y: np.ndarray, noise_rate: float, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with Gaussian distributed noise, at a specific rate, for a specific ratio of the training data.

    Args:
        X (np.ndarray): Training data.
        Y (np.ndarray): Training target labels.
        noise_rate (float): Rate of noise augmentation, where an increasing value from 0 - 1 indicates more noise.
        noise_ratio (float): Ratio of data augmentation defining the amount of augmentated instances to add to each class.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented input data and target labels.
    """
    if not noise_rate or not noise_ratio:
        logger.debug("Noise rate or ratio of 0, no data was augmented.")
        return X, Y

    # Flatten Y to handle both one-hot encoded and categorical labels
    Y_flat = np.argmax([y[0] for y in Y], axis=1) if len(Y.shape) > 1 else Y

    num_classes = len(np.unique(Y_flat))
    num_instances = int(noise_ratio * len(X))
    num_instances_per_class = num_instances // num_classes

    X_augmented = np.empty((0, X.shape[1], X.shape[2]))
    Y_augmented = np.empty(0)

    for class_label in np.unique(Y_flat):
        class_instances = X[Y_flat == class_label]

        if len(class_instances) < num_instances_per_class:
            raise ValueError(f"Not enough instances in class {class_label} for augmentation.")

        indices = np.random.choice(len(class_instances), num_instances_per_class, replace=True)
        instances = class_instances[indices]

        # Generate noise using a normal distribution
        noise = np.random.normal(0, noise_rate, instances.shape)
        
        # Scale the noise to have similar range as the instances
        noise_min = np.min(noise, axis=1, keepdims=True)
        noise_max = np.max(noise, axis=1, keepdims=True)
        noise_range = noise_max - noise_min
        noise = (noise - noise_min) / noise_range * (noise_rate)
        noisy_instances = instances + noise

        X_augmented = np.concatenate((X_augmented, noisy_instances), axis=0)

        augmented_labels = np.full(num_instances_per_class, class_label)
        Y_augmented = np.concatenate((Y_augmented, augmented_labels))

    shuffle_indices = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[shuffle_indices]
    Y_augmented = Y_augmented[shuffle_indices]

    if len(Y.shape) > 1:
        Y_augmented = _one_hot_encode(Y_augmented.reshape(-1, 1), num_classes)

    X_augmented = np.concatenate((X, X_augmented), axis=0)
    Y_augmented = np.concatenate((Y, Y_augmented))

    return X_augmented, Y_augmented

def _merge_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, binary: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the dataset by concatenating train and test datasets.

    Args:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
        binary (bool): Whether to merge arrhythmic classes into one.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed input data and target labels.
    """
    # Concatenate train and test datasets
    X_combined = pd.concat([train_df.iloc[:, :-1], test_df.iloc[:, :-1]], axis=0)
    Y_combined = pd.concat([train_df.iloc[:, -1], test_df.iloc[:, -1]], axis=0)

    # If binary flag is enabled, merge arrhythmic classes into one
    if binary:
        logger.debug("Merging all arrhythmia variations into one target for binary classification")
        Y_combined[Y_combined != 0] = 1

    return X_combined.values, Y_combined.values

def _load_raw_ecg_dataset(binary: bool, train_file_path: str, test_file_path: str, save_file_path: str, save_file_path_binary: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load or preprocess ECG dataset.

    Args:
        binary (bool): Whether to load dataset with binary targets.
        train_file_path (str): File path for training data.
        test_file_path (str): File path for testing data.
        save_file_path (str): File path to save preprocessed data.
        save_file_path_binary (str): File path to save preprocessed data with binary targets.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed input data and target labels.
    """
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

    X_loaded, Y_loaded = _merge_dataset(train_df, test_df, binary)
    save_npz(file_path, **{"X": X_loaded, "Y": Y_loaded})
    logger.debug(f"Saving preprocessed dataset to: {file_path}")

    return X_loaded, Y_loaded

def _train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float, shuffle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.
        test_size (float): Ratio of testing data to total data.
        shuffle (bool): Whether to shuffle instances.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split training and testing data.
    """
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
    """
    Perform one-hot encoding of target labels.

    Args:
        labels (np.ndarray): Target labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: One-hot encoded target labels.
    """
    return np.eye(num_classes)[labels.astype(int)]

def _reshape_data(X: np.ndarray, Y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Reshape input and target data for reservoir processing.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Reshaped input and target data.
    """
    # Reshape X to have shape (num_instances, timesteps, features)
    X_reshaped = [instance.T.reshape(-1, 1) for instance in X]

    # Create a list of arrays for Y
    Y_reshaped = [np.array([label]) for label in Y]

    return X_reshaped, Y_reshaped

def _limit_instances(X: np.ndarray, Y: np.ndarray, rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limit the number of instances to balance classes.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target labels.
        rows (int): Number of rows to limit the dataset to.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Limited input data and target labels.
    """
    Y = Y.astype(int)
    classes, counts = np.unique(Y, return_counts=True)
    max_instances_per_class = int(rows / len(classes))
    limited_indices = [np.random.choice(np.where(Y == c)[0], min(count, max_instances_per_class), replace=False) for c, count in zip(classes, counts)]
    return X[np.concatenate(limited_indices)], Y[np.concatenate(limited_indices)]

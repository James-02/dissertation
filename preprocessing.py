import numpy as np
import pandas as pd
import os

DATA_DIR = 'data/ecg/'

def load_ecg_data(repeat_targets=False, encode_labels=True, normalise=True, class_size=None):
    """
    Load ECG data and preprocess it.

    Parameters:
        repeat_targets (bool): Whether to repeat targets.
        encode_labels (bool): Whether to encode labels.
        normalise (bool): Whether to normalise data.
        class_size (int): Size of each class in the subset.

    Returns:
        tuple: A tuple containing preprocessed training and testing data (X_train, Y_train, X_test, Y_test).
    """
    if not all(os.path.exists(os.path.join(DATA_DIR, file)) for file in ["X_train.npy", "Y_train.npy", "X_test.npy", "Y_test.npy"]):
        X_train, Y_train, X_test, Y_test = _load_ecg_data()

    X_train, Y_train = _preprocess_data(X_train, Y_train, encode_labels, normalise, repeat_targets, class_size)
    X_test, Y_test = _preprocess_data(X_test, Y_test, encode_labels, normalise, repeat_targets, class_size)

    return X_train, Y_train, X_test, Y_test

def _save_data(X_train, Y_train, X_test, Y_test):
    """
    Save the training and testing data to disk.

    Parameters:
        X_train (numpy.ndarray): Training data features.
        Y_train (numpy.ndarray): Training data labels.
        X_test (numpy.ndarray): Testing data features.
        Y_test (numpy.ndarray): Testing data labels.
    """
    np.save(os.path.join(DATA_DIR, "ecg_X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "ecg_Y_train.npy"), Y_train)
    np.save(os.path.join(DATA_DIR, "ecg_X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "ecg_Y_test.npy"), Y_test)

def _preprocess_data(X, Y, encode_labels=True, normalise=True, repeat_targets=False, class_size=None):
    """
    Preprocess input features and labels.

    Parameters:
        X (numpy.ndarray): Input features.
        Y (numpy.ndarray): Input labels.
        encode_labels (bool): Whether to encode labels.
        normalise (bool): Whether to normalise data.
        repeat_targets (bool): Whether to repeat targets.
        class_size (int): Size of each class in the subset.

    Returns:
        tuple: A tuple containing preprocessed input features and labels (X, Y).
    """
    if normalise:
        X = _normalise(X)

    if encode_labels:
        Y = _encode_labels(Y)

    if repeat_targets:
        Y = _repeat_targets(X, Y)

    if class_size:
        X, Y = _get_subset(X, Y, class_size)
    
    Y = _reshape_Y(Y)
    X = _reshape_X(X)
    
    return X, Y

def _load_ecg_data():
    """
    Load ECG data from CSV files.

    Returns:
        tuple: A tuple containing raw training and testing data (X_train, Y_train, X_test, Y_test).
    """
    train_df = pd.read_csv(os.path.join(DATA_DIR, "ecg_train.csv"), header=None)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "ecg_test.csv"), header=None)
    
    train_df = _balance_dataset(train_df)
    test_df = _balance_dataset(test_df)

    X_train = train_df.iloc[:, :-1].values
    Y_train = train_df.iloc[:, -1].values.astype(int)
    X_test = test_df.iloc[:, :-1].values
    Y_test = test_df.iloc[:, -1].values.astype(int)

    _save_data(X_train, Y_train, X_test, Y_test)

    return X_train, Y_train, X_test, Y_test

def _get_subset(X, Y, class_size):
    """
    Get a subset of data with balanced classes.

    Parameters:
        X (numpy.ndarray): Input features.
        Y (numpy.ndarray): Input labels.
        class_size (int): Size of each class in the subset.

    Returns:
        tuple: A tuple containing subset of input features and labels (X_subset, Y_subset).
    """
    class_counts = pd.Series(Y.flatten()).value_counts()
    min_class_count = class_counts.min()
    class_size = min(class_size, min_class_count)
    
    sampled_indices = []
    
    for label, _ in class_counts.items():
        indices = np.where(Y == label)[0]
        np.random.shuffle(indices)
        sampled_indices.extend(indices[:class_size])
    
    X_subset = X[sampled_indices]
    Y_subset = Y[sampled_indices]
    
    return X_subset, Y_subset

def _balance_dataset(data):
    """
    Sample data to balanced classes.

    Parameters:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Sampled data.
    """
    class_size = data.iloc[:, -1].value_counts().min()
    return data.groupby(data.columns[-1]).apply(lambda x: x.sample(n=class_size))

def _encode_labels(Y):
    """
    Encode labels using one-hot encoding.

    Parameters:
        Y (numpy.ndarray): Input labels.

    Returns:
        numpy.ndarray: Encoded labels.
    """
    return np.eye(np.max(Y) + 1)[Y]

def _normalise(X):
    """
    Normalize input features.

    Parameters:
        X (numpy.ndarray): Input features.

    Returns:
        numpy.ndarray: Normalized input features.
    """
    min_values = X.min()
    max_values = X.max()
    return (X - min_values) / (max_values - min_values)

def _reshape_X(X):
    """
    Reshape input features array.

    Parameters:
        X (numpy.ndarray): Input features.

    Returns:
        list: List of reshaped input features arrays.
    """
    return [array.reshape(-1, 1) for array in X]

def _reshape_Y(Y):
    """
    Reshape input labels array.

    Parameters:
        Y (numpy.ndarray): Input labels.

    Returns:
        list: List of reshaped input labels arrays.
    """
    return [array.reshape(1, len(array)) for array in Y]

def _repeat_targets(X, Y):
    """
    Repeat each Y value for the dataset to the size of the relative X.

    Parameters:
        X (numpy.ndarray): Input features.
        Y (numpy.ndarray): Input labels.

    Returns:
        list: List of repeated input labels arrays.
    """
    return [np.repeat(array, X[0].shape[0], axis=0) for array in Y]

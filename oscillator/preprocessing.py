import numpy as np
import pandas as pd

# Define global constants
DATA_DIR = 'data/'
DEFAULT_ROWS = 1000

def load_ecg(rows=DEFAULT_ROWS, repeat_targets=False, encode_labels=True, normalise=True, class_size=None):
    """
    Load the ECG dataset with optional row limitation.

    Parameters:
        rows (int, optional): Number of rows to load. Defaults to DEFAULT_ROWS.
        repeat_targets (bool, optional): Whether to repeat each Y value for the training and testing set 
                                         to the size of the relative X. Defaults to False.
        encode_labels (bool, optional): Whether to encode labels using one-hot encoding. Defaults to True.
        normalise (bool, optional): Whether to normalize inputs to phases. Defaults to True.

    Returns:
        tuple: Tuple containing X_train, Y_train, X_test, Y_test.
    """
    # Load the dataset
    train_df = pd.read_csv(DATA_DIR + "mit-bih/mitbih_train.csv", header=None)
    test_df = pd.read_csv(DATA_DIR + "mit-bih/mitbih_test.csv", header=None)

    # Sample data to balance dataset
    train_df = _sample_data(train_df, class_size)
    test_df = _sample_data(test_df, class_size)

    # Extract features and labels
    X_train = train_df.iloc[:, :-1].values
    Y_train = train_df.iloc[:, -1].values.astype(int)
    X_test = test_df.iloc[:, :-1].values
    Y_test = test_df.iloc[:, -1].values.astype(int)

    # Preprocess data
    X_train, Y_train = preprocess_data(X_train, Y_train, encode_labels, normalise, repeat_targets)
    X_test, Y_test = preprocess_data(X_test, Y_test, encode_labels, normalise, repeat_targets)

    return X_train, Y_train, X_test, Y_test

def preprocess_data(X, Y, encode_labels=True, normalise=True, repeat_targets=False):
    """
    Preprocess data for training/testing.

    Parameters:
        X (numpy.ndarray): Features array.
        Y (numpy.ndarray): Labels array.
        encode_labels (bool, optional): Whether to encode labels using one-hot encoding. Defaults to True.
        normalise (bool, optional): Whether to normalize inputs. Defaults to True.
        repeat_targets (bool, optional): Whether to repeat each Y value to match the size of the relative X. Defaults to False.

    Returns:
        tuple: Tuple containing preprocessed features and labels.
    """

    # Normalise inputs into phases
    if normalise:
        X = _normalise(X)

    # Encode labels
    if encode_labels:
        Y = _encode_labels(Y)

    Y = _reshape_Y(Y)
    X = _reshape_X(X)

    # Repeat targets if necessary
    if repeat_targets:
        Y = _repeat_targets(X, Y)
    
    return X, Y

def _repeat_targets(X, Y):
    """
    Repeat each Y value for the dataset to the size of the relative X.

    Parameters:
        X (numpy.ndarray): Features array.
        Y (numpy.ndarray): Targets array.

    Returns:
        numpy.ndarray: Repeated targets array.
    """
    return [np.repeat(array, X[0].shape[0], axis=0) for array in Y]    

def _sample_data(data, class_size=None):
    """
    Sample data to balance classes.
    if the class_size is not set, the size of the smallest class is used.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        class_size (int): Size of classes

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    # Determine the size of the smallest class
    class_size = data.iloc[:, -1].value_counts().min() if not class_size else class_size

    # Sample each class to balance data
    sampled_dfs = [group.sample(n=class_size) for _, group in data.groupby(data.columns[-1])]

    # Concatenate the sampled DataFrames into a single DataFrame
    return pd.concat(sampled_dfs)

def _encode_labels(Y):
    """
    Encode labels using one-hot encoding.

    Parameters:
        Y (numpy.ndarray): Labels array.

    Returns:
        numpy.ndarray: One-hot encoded labels array.
    """
    return np.eye(np.max(Y) + 1)[Y]

def _normalise(X):
    """
    Normalize values between 0-1.

    Parameters:
        X (numpy.ndarray): Array of values.

    Returns:
        numpy.ndarray: Normalized array.
    """
    # Find the minimum and maximum values along each row
    min_values = np.min(X)
    max_values = np.max(X)

    # Normalize values to be between 0 and 1
    return (X - min_values) / (max_values - min_values)

def _reshape_X(X):
    """
    Reshapes each array in the input list of arrays to have a shape of (timesteps, features).
    
    Parameters:
    -----------
    X : list of numpy.ndarray
        The list of input arrays to be reshaped.
        
    Returns:
    --------
    list of numpy.ndarray
        The list of reshaped arrays with shape (timesteps, features).
    """
    return [array.reshape(-1, 1) for array in X]


def _reshape_Y(Y):
    """
    Reshapes each array in the input list of arrays to have a shape of (timesteps, classes)
    
    Parameters:
    -----------
    Y : list of numpy.ndarray
        The list of labels arrays to be reshaped
        
    Returns:
    --------
    list of numpy.ndarray
        The list of reshaped arrays with shape (timesteps, classes).
    """
    return [array.reshape(1, len(array)) for array in Y]

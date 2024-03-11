import numpy as np
import pandas as pd

DATA_PATH = 'data/'

def load_mnist():
    data = np.load(DATA_PATH + 'mnist.npz')
    # Split out testing and training sets into inputs and targets
    return data['x_train'][:5000], data['y_train'][:5000], data['x_test'][:5000], data['y_test'][:5000]

def load_ecg(rows=100000):
    train_df = pd.read_csv(DATA_PATH + "mit-bih/mitbih_train.csv", nrows=rows, header=None)
    test_df = pd.read_csv(DATA_PATH + "mit-bih/mitbih_test.csv", nrows=rows, header=None)

    train_df = _sample_data(train_df)
    test_df = _sample_data(test_df)

    X_train = [np.array(row) for row in train_df.iloc[:, :-1].values]
    Y_train = [np.array([label]) for label in train_df.iloc[:, -1].values.astype(int)]

    X_test = [np.array(row) for row in test_df.iloc[:, :-1].values]
    Y_test = [np.array([label]) for label in test_df.iloc[:, -1].values.astype(int)]

    X_train = [array.reshape(-1, 1) for array in _normalise_to_phases(X_train)]
    X_test = [array.reshape(-1, 1) for array in _normalise_to_phases(X_test)]

    num_classes = max(max(targets) for targets in Y_train) + 1

    # Perform normalization for each target array using list comprehension
    Y_train = [(np.eye(num_classes)[targets] / np.max(np.eye(num_classes)[targets], axis=1, keepdims=True)) for targets in Y_train]
    Y_test = [(np.eye(num_classes)[targets] / np.max(np.eye(num_classes)[targets], axis=1, keepdims=True)) for targets in Y_test]

    return X_train, Y_train, X_test, Y_test

def _sample_data(data):
    # Determine the size of the smallest class
    min_class_size = data.iloc[:, -1].value_counts().min()

    # Group the DataFrame by the target column
    grouped = data.groupby(data.columns[-1])

    # Initialize an empty list to store the sampled DataFrames
    sampled_dfs = []

    # Iterate over each group, and sample min_class_size records from each group
    for _, group in grouped:
        sampled_dfs.append(group.sample(n=min_class_size))

    # Concatenate the sampled DataFrames into a single DataFrame
    subset_df = pd.concat(sampled_dfs)

    # Reset the index of the subset DataFrame
    subset_df.reset_index(drop=True, inplace=True)

    return subset_df

def _normalise_to_phases(values):
    # Find the minimum and maximum values along each row
    min_values = np.min(values, axis=1)[:, np.newaxis]
    max_values = np.max(values, axis=1)[:, np.newaxis]

    # Normalize each row of values to be between 0 and 1
    normalized_values = (values - min_values) / (max_values - min_values)

    # Map normalized values to the range [0, 180] (representing degrees)
    return normalized_values * 180

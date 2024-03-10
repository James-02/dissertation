import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data/'

def load_mnist():
    data = np.load(DATA_PATH + 'mnist.npz')
    # Split out testing and training sets into inputs and targets
    return data['x_train'][:5000], data['y_train'][:5000], data['x_test'][:5000], data['y_test'][:5000]

def load_ecg():
    train_df = pd.read_csv(DATA_PATH + "mit-bih/mitbih_train.csv", nrows=100, header=None)
    test_df = pd.read_csv(DATA_PATH + "mit-bih/mitbih_test.csv", nrows=100, header=None)

    label_encoder = LabelEncoder()

    X_train = [np.array(row) for row in train_df.iloc[:, :-1].values]
    Y_train = [np.array([label]) for label in label_encoder.fit_transform(train_df.iloc[:, -1].values.astype(int))]

    X_test = [np.array(row) for row in test_df.iloc[:, :-1].values]
    Y_test = [np.array([label]) for label in label_encoder.fit_transform(test_df.iloc[:, -1].values.astype(int))]

    return X_train, Y_train, X_test, Y_test

def normalise_to_phases(values):
    # Find the minimum and maximum values along each row
    min_values = np.min(values, axis=1)[:, np.newaxis]
    max_values = np.max(values, axis=1)[:, np.newaxis]

    # Normalize each row of values to be between 0 and 1
    normalized_values = (values - min_values) / (max_values - min_values)

    # Map normalized values to the range [0, 180] (representing degrees)
    return normalized_values * 180

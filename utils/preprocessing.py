from typing import Tuple, List
import os

import numpy as np
import pandas as pd

from utils.logger import Logger

DEFAULT_LOG_LEVEL = 1

class DataLoader:
    """
    Class for loading and preprocessing ECG dataset.
    """

    def __init__(self, log_level: int = DEFAULT_LOG_LEVEL, log_file: str = None, seed: int = None,
                 data_dir: str = "data/ecg", train_file: str = "ecg_train.csv", 
                 test_file: str = "ecg_test.csv", save_file: str = "ecg_data.npz"):
        """
        Initialize the Data Loader with directory paths and filenames.

        Args:
            log_level (int, optional): Logging level for the class. Defaults to 1.
            log_file (str, optional): File path to save logs to.
            seed (int, optional): Random state seed.
            data_dir (str, optional): Directory containing ECG data files. Defaults to "ecg".
            train_file (str, optional): Filename of the training data file. Defaults to "ecg_train.csv".
            test_file (str, optional): Filename of the testing data file. Defaults to "ecg_test.csv".
            save_file (str, optional): Filename to save preprocessed data. Defaults to "ecg_data.npz".
        """
        self.logger = Logger(name=__name__, level=log_level, log_file=log_file)

        self.data_path = data_dir
        self.train_file_path = os.path.join(self.data_path, train_file)
        self.test_file_path = os.path.join(self.data_path, test_file)
        self.save_file_path = os.path.join(self.data_path, save_file)

        np.random.seed(seed)

    def load_ecg_data(self, rows: int = None, test_ratio: float = 0.2, encode_labels: bool = True, 
                      normalize: bool = True, repeat_targets: bool = False, shuffle: bool = True) \
                        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load ECG dataset, preprocess, balance classes, and split into training and testing subsets.

        Args:
            rows (int): Number of rows to limit the dataset to, defaults to all rows.
            test_ratio (float, optional): Ratio of testing data to total data. Defaults to 0.2.
            encode_labels (bool, optional): Whether to one-hot encode labels. Defaults to True.
            normalize (bool, optional): Whether to normalize the input values. Defaults to True.
            repeat_targets (bool, optional): Whether to repeat targets. Defaults to False.
            shuffle (bool, optional): Whether to shuffle instances. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing X_train, Y_train, X_test, Y_test.
        """
        self.logger.info("Loading ECG Dataset")
        X_loaded, Y_loaded = self._load_and_preprocess_data()

        # If rows is not given, set to all rows
        if rows is None:
            rows = len(X_loaded)
            self.logger.warning("Number of rows not specified, implicitly using all rows.")

        self.logger.debug(f"Loading ECG data with parameters: [rows={rows}, test_ratio={test_ratio}, encode_labels={encode_labels}, normalise={normalize}, repeat_targets={repeat_targets}]")

        if normalize:
            self.logger.debug("Normalizing inputs")
            X_loaded = self._normalize(X_loaded)

        self.logger.debug("Balancing classes")
        X_balanced, Y_balanced = self._balance_classes(X_loaded, Y_loaded)

        self.logger.debug(f"Limiting instances to {rows} rows")
        X_limited, Y_limited = self._limit_instances(X_balanced, Y_balanced, rows)

        num_rows = X_limited.shape[0]
        if num_rows < rows:
            self.logger.warning(f"The {rows} rows requested were capped at {num_rows} rows to keep classes balanced.")

        self.logger.debug("Splitting dataset into training and testing sets")
        X_train, Y_train, X_test, Y_test = self._train_test_split(X_limited, Y_limited, test_size=test_ratio, shuffle=shuffle)

        if encode_labels:
            num_classes = len(np.unique(Y_loaded))
            self.logger.debug("One-hot encoding labels.")
            Y_train = self._one_hot_encode(Y_train, num_classes)
            Y_test = self._one_hot_encode(Y_test, num_classes)

        self.logger.debug("Reshaping data into time-series.")
        X_train, Y_train = self._reshape_data(X_train, Y_train)
        X_test, Y_test = self._reshape_data(X_test, Y_test)

        # Repeat targets if specified
        if repeat_targets:
            self.logger.debug(f"Repeating train and test targets to size: {X_train[0].shape[0]}")
            Y_train = [np.repeat(Y_instance, X_instance.shape[0], axis=0) for X_instance, Y_instance in zip(X_train, Y_train)]
            Y_test = [np.repeat(Y_instance, X_instance.shape[0], axis=0) for X_instance, Y_instance in zip(X_test, Y_test)]

        return X_train, Y_train, X_test, Y_test

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize the input values.

        Args:
            X (np.ndarray): Input feature data.

        Returns:
            np.ndarray: Normalized feature data.
        """
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        return (X - min_vals) / (max_vals - min_vals)

    def _save_dataset(self, X: np.ndarray, Y: np.ndarray, filename: str) -> None:
        """
        Save dataset X and corresponding labels Y to a file.

        Args:
            X (np.ndarray): Feature data.
            Y (np.ndarray): Label data.
            filename (str): Name of the file to save.
        """
        np.savez(filename, X=X, Y=Y)

    def _load_dataset(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset X and corresponding labels Y from a file.

        Args:
            filename (str): Name of the file to load.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Loaded feature data and label data.
        """
        data = np.load(filename)
        return data['X'], data['Y']

    def _balance_classes(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes in dataset X with labels Y.

        Args:
            X (np.ndarray): Feature data.
            Y (np.ndarray): Label data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Balanced feature data and label data.
        """
        Y = Y.astype(int)
        classes, counts = np.unique(Y, return_counts=True)
        min_instances_per_class = min(counts)
        balanced_indices = [np.random.choice(np.where(Y == c)[0], min_instances_per_class, replace=False) for c in classes]
        return X[np.concatenate(balanced_indices)], Y[np.concatenate(balanced_indices)]

    def _preprocess_dataset(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess train and test DataFrames and return balanced datasets.

        Args:
            train_df (pd.DataFrame): Training DataFrame.
            test_df (pd.DataFrame): Testing DataFrame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Balanced feature data and label data.
        """
        # Concatenate train and test datasets
        X_combined = pd.concat([train_df.iloc[:, :-1], test_df.iloc[:, :-1]], axis=0)
        Y_combined = pd.concat([train_df.iloc[:, -1], test_df.iloc[:, -1]], axis=0)

        return self._balance_classes(X_combined.values, Y_combined.values)

    def _load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess datasets, returning balanced X and Y.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Balanced feature data and label data.
        """
        if os.path.exists(self.save_file_path):
            self.logger.debug(f"Loading preprocessed dataset from: {self.save_file_path}")
            return self._load_dataset(self.save_file_path)
        else:
            self.logger.warning(f"Could not load dataset at: {self.save_file_path}")
            self.logger.debug(f"Preprocessing datasets from: {self.train_file_path} and {self.test_file_path}")

            train_df = pd.read_csv(self.train_file_path, header=None)
            test_df = pd.read_csv(self.test_file_path, header=None)

            X_loaded, Y_loaded = self._preprocess_dataset(train_df, test_df)
            self._save_dataset(X_loaded, Y_loaded, self.save_file_path)
            self.logger.debug(f"Saving preprocessed dataset to: {self.save_file_path}")
            return X_loaded, Y_loaded

    def _train_test_split(self, X: np.ndarray, Y: np.ndarray, test_size: float, shuffle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset X and corresponding labels Y into training and testing sets with balanced classes.

        Args:
            X (np.ndarray): Feature data.
            Y (np.ndarray): Label data.
            test_size (float): Ratio of testing data to total data.
            shuffle (bool): Flag indicating whether to shuffle the data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split feature data and label data.
        """
        self.logger.debug(f"Train:Test dataset ratio: [{int((1 - test_size) * 100)}:{int(test_size * 100)}]")
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

    def _one_hot_encode(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """
        One-hot encode labels.

        Args:
            labels (np.ndarray): Label data.
            num_classes (int): Number of classes.

        Returns:
            np.ndarray: One-hot encoded label data.
        """
        return np.eye(num_classes)[labels.astype(int)]

    def _reshape_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Reshape time-series data into a list of time-series arrays where each array is an instance [(timesteps, features)].

        Args:
            X (np.ndarray): Input feature data with shape (num_instances, timesteps, features).
            Y (np.ndarray): Input label data with shape (num_instances, num_classes) or (num_instances,).

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Reshaped feature data and label data as lists of arrays.
        """
        # Reshape X to have shape (num_instances, timesteps, features)
        X_reshaped = [instance.T.reshape(-1, 1) for instance in X]

        # If Y is one-hot encoded, keep it as it is
        if len(Y.shape) > 1:
            Y_reshaped = [label.reshape(-1, 1) for label in Y]

        # Create a list of arrays for Y
        Y_reshaped = [np.array([label]) for label in Y]

        return X_reshaped, Y_reshaped

    def _limit_instances(self, X: np.ndarray, Y: np.ndarray, rows: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Limit the number of instances to rows while maintaining balanced classes.

        Args:
            X (np.ndarray): Feature data.
            Y (np.ndarray): Label data.
            rows (int): Number of rows to limit the dataset to.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Limited feature data and label data.
        """
        Y = Y.astype(int)
        classes, counts = np.unique(Y, return_counts=True)
        max_instances_per_class = int(rows / len(classes))
        limited_indices = [np.random.choice(np.where(Y == c)[0], min(count, max_instances_per_class), replace=False) for c, count in zip(classes, counts)]
        return X[np.concatenate(limited_indices)], Y[np.concatenate(limited_indices)]

    def log_dataset_info(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray) -> None:
        """
        Log information about the training and testing datasets.

        Args:
            X_train (np.ndarray): Training feature data.
            Y_train (np.ndarray): Training label data.
            X_test (np.ndarray): Testing feature data.
            Y_test (np.ndarray): Testing label data.
        """
        train_shapes = (X_train[0].shape, Y_train[0].shape)
        test_shapes = (X_test[0].shape, Y_test[0].shape)

        self.logger.info("----- ECG Dataset Report -----")

        self.logger.info(f"Training Instances: {len(X_train)}")
        self.logger.info(f"Testing Instances: {len(X_test)}")

        self.logger.info(f"Training Instances Shape: {train_shapes}")
        self.logger.info(f"Testing Instances Shape: {test_shapes}")

        self.logger.info(f"Training label counts: {self._get_label_counts(Y_train)}")
        self.logger.info(f"Testing label counts: {self._get_label_counts(Y_test)}")
        self.logger.info("------------------------------")

    def _get_label_counts(self, Y: List) -> dict:
        """
        Calculate and return label counts.

        Args:
            Y (List): List of label data.

        Returns:
            dict: Dictionary containing label counts.
        """
        # Always use the first row each target, in case the targets are repeated
        Y = np.array([targets[0] for targets in Y])

        # If Y is encoded, revert the encoding
        if len(Y.shape) != 1:
            Y = np.argmax(Y, axis=1)

        return {label: count for label, count in zip(*np.unique(Y, return_counts=True))}

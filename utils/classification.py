from multiprocessing import Pool
from typing import Optional

import os
import time
import numpy as np
import reservoirpy

from reservoirpy import Node
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from utils.logger import Logger

DEFAULT_LOG_LEVEL = 1
DEFAULT_VERBOSITY = 1

class Classifier:
    """
    Class for time-series training and classification using a reservoir computing model.
    """
    
    def __init__(self, reservoir: Node, readout: Node, train_set: tuple, test_set: tuple, log_level: int = DEFAULT_LOG_LEVEL, 
                 log_file: str = None, seed: int = None, verbosity: int = DEFAULT_VERBOSITY):
        """
        Initialize the Classifier object.

        Args:
            reservoir (Node): Reservoir node for processing input data.
            readout (Node): Readout node for classification.
            train_set (tuple): Tuple containing training data features and labels.
            test_set (tuple): Tuple containing test data features and labels.
            log_level (int): Log level for logging messages.
            log_file (str): File path to save log messages.
            seed (int): Random seed for reproducibility.
            verbosity (int): Verbosity level for reservoirpy library.
        """
        self.logger = Logger(name=__name__, level=log_level, log_file=log_file)
        self.reservoir = reservoir
        self.readout = readout
        self.X_train, self.Y_train = train_set
        self.X_test, self.Y_test = test_set
        self.results_path = "results/training"

        np.random.seed(seed)
        reservoirpy.set_seed(seed)
        reservoirpy.verbosity(verbosity)

    def _train_reservoir(self, x: np.ndarray):
        """
        Train the reservoir node.

        Args:
            x (np.ndarray): Input data for training.

        Returns:
            numpy.ndarray: Trained states.
        """
        return self.reservoir.run(x)

    def _predict_reservoir(self, x: np.ndarray):
        """
        Predict using the reservoir node.

        Args:
            x (np.ndarray): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted states.
        """
        states = self.reservoir.run(x)
        return self.readout.run(states[-1, np.newaxis])

    def train(self, X_train: np.ndarray, processes: int):
        """
        Train the reservoir using the training instances.

        Args:
            X_train (np.ndarray): Training data features.
            processes (int): Processes to use. (A value of 0 will use the maximum)

        Returns:
            list: List of trained states.
        """
        if processes is None or processes > 1:
            with Pool(processes=processes) as pool:
                trained_states = pool.map(self._train_reservoir, [x for x in X_train])
        else:
            trained_states = [self._train_reservoir(x) for x in X_train]
        return [state[-1, np.newaxis] for state in trained_states]

    def predict(self, X_test: np.ndarray, processes: int):
        """
        Perform prediction by training the reservoir on the test instances and then training the readout on the states produced.

        Args:
            X_test (np.ndarray): Test data features.
            processes (int): Processes to use. (A value of 0 will use the maximum)

        Returns:
            list: List of predicted states.
        """
        if processes is None or processes > 1:
            with Pool(processes=processes) as pool:
                predicted_states = pool.map(self._predict_reservoir, [x for x in X_test])
            return predicted_states
        else:
            return [self._predict_reservoir(x) for x in X_test]

    def log_metrics(self, metrics: dict):
        """
        Log performance metrics.

        Args:
            metrics (dict): Dictionary containing performance metrics.
        """
        self.logger.info("----- Classification Report -----")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.3f}%")
        self.logger.info(f"MSE: {metrics['mse']:.3f}")
        self.logger.info(f"Recall: {metrics['recall']:.3f}")
        self.logger.info(f"Precision: {metrics['precision']:.3f}")
        self.logger.info(f"F1: {metrics['f1']:.3f}")
        self.logger.info("---------------------------------")

    def save_states_to_file(self, file_path: str, states: np.ndarray):
        """
        Save states to a file.

        Args:
            file_path (str): Path to save the file.
            states (np.ndarray): States to be saved.
        """
        try:
            np.save(file=os.path.join(self.results_path, file_path), arr=states)
            self.logger.info(f"Saved {len(states)} states to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving states to file: {e}")

    def load_states_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load states from a file.

        Args:
            file_path (str): Path to the file.

        Returns:
            numpy.ndarray or None: Loaded states if successful, else None.
        """
        try:
            return np.load(file=os.path.join(self.results_path, file_path))
        except Exception as e:
            self.logger.error(f"Error loading states from file: {e}")
            return None

    def classify(self, processes: int = 0, save_states: bool = False, load_states: bool = False):
        """
        Perform classification by training the reservoir, and then predicting testing instances' classes using the readout layer.

        Args:
            processes (int): Processes to use. (A value of 0 will use the maximum allowed by the system)
            save_states (bool): Whether to save states.
            load_states (bool): Whether to load states.

        Returns:
            dict: Performance metrics.
        """
        training_instances = len(self.X_train)
        testing_instances = len(self.X_test)

        # try to load states if they exist
        trained_states = []
        if load_states:
            file = f"states-{self.reservoir.name}-{self.reservoir.units}-{training_instances}.npy"
            self.logger.debug(f"Attempting to load states from: {file}")
            trained_states = self.load_states_from_file(file)
            if trained_states is not None:
                self.logger.info(f"Loaded {len(trained_states)} states from: {file}")

        # train states if could not be loaded
        if not trained_states:
            self.logger.info(f"Training Reservoir of {self.reservoir.units} nodes with {training_instances} instances")
            start = time.time()
            trained_states = self.train(self.X_train, processes)
            end = time.time()
            self.logger.debug(f"Training Time Elapsed: {str(round(end - start, 4))}s")

            if save_states:
                self.save_states_to_file(f"states-{self.reservoir.name}-{self.reservoir.units}-{training_instances}", trained_states)

        # Fitting
        self.logger.info(f"Fitting readout layer with {len(trained_states)} states")
        self.readout.fit(trained_states, self.Y_train)

        # Predicting
        self.logger.info(f"Predicting with {testing_instances} instances.")
        start = time.time()
        Y_pred = self.predict(self.X_test, processes)
        end = time.time()
        self.logger.debug(f"Prediction Time Elapsed: {str(round(end - start, 4))}s")

        # Calculate performance metrics
        self.logger.info("Calculating model performance metrics")
        Y_test_class = np.array([np.argmax(y_t) for y_t in self.Y_test])
        Y_pred_class = np.array([np.argmax(y_p) for y_p in Y_pred])
        metrics = self.evaluation(Y_test_class, Y_pred_class)

        return metrics
        
    def evaluation(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        """
        Evaluate the classification performance.

        Args:
            Y_true (np.ndarray): True labels.
            Y_pred (np.ndarray): Predicted labels.

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

from typing import Optional

import os
import time
import numpy as np

from reservoirpy import Node
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score

from utils.logger import Logger

DEFAULT_LOG_LEVEL = 1
DEFAULT_VERBOSITY = 1

class Classifier:
    def __init__(self, reservoir: Node, readout: Node, train_set: tuple, test_set: tuple, log_level: int = DEFAULT_LOG_LEVEL, log_file: str = None):
        self.logger = Logger(name=__name__, level=log_level, log_file=log_file)
        self.reservoir = reservoir
        self.readout = readout
        self.X_train, self.Y_train = train_set
        self.X_test, self.Y_test = test_set
        self.results_path = "results/training"

    def train(self, X_train: np.ndarray):
        trained_states = [self.reservoir.run(x) for x in X_train]
        return [state[-1, np.newaxis] for state in trained_states]

    def predict(self, X_test: np.ndarray):
        return [self.readout.run(self.reservoir.run(x) [-1, np.newaxis]) for x in X_test]

    def log_metrics(self, metrics: dict):
        self.logger.info("----- Classification Report -----")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.3f}%")
        self.logger.info(f"MSE: {metrics['mse']:.3f}")
        self.logger.info(f"RMSE: {metrics['rmse']:.3f}")
        self.logger.info(f"R^2: {metrics['r_squared']:.3f}")
        self.logger.info(f"Recall: {metrics['recall']:.3f}")
        self.logger.info(f"Precision: {metrics['precision']:.3f}")
        self.logger.info(f"F1: {metrics['f1']:.3f}")
        self.logger.info("---------------------------------")

    def log_params(self):
        params = self.reservoir.hypers
        self.logger.info("----- Reservoir Parameters -----")
        for k, v in params.items():
            self.logger.debug(f"{k}: {v}")
        self.logger.info("--------------------------------")

        params = self.readout.hypers
        self.logger.info("----- Readout Parameters -----")
        for k, v in params.items():
            self.logger.debug(f"{k}: {v}")
        self.logger.info("--------------------------------")

    def save_states_to_file(self, file_path: str, states: np.ndarray):
        try:
            np.save(file=os.path.join(self.results_path, file_path), arr=states)
            self.logger.info(f"Saved {len(states)} states to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving states to file: {e}")

    def load_states_from_file(self, file_path: str) -> Optional[np.ndarray]:
        try:
            return np.load(file=os.path.join(self.results_path, file_path))
        except Exception as e:
            self.logger.error(f"Error loading states from file: {e}")
            return []

    def classify(self, save_states: bool = False, load_states: bool = False):
        training_instances = len(self.X_train)
        testing_instances = len(self.X_test)

        # try to load states if they exist
        trained_states = []
        if load_states:
            file = f"states-{self.reservoir.name}-{self.reservoir.units}-{training_instances}.npy"
            self.logger.debug(f"Attempting to load states from: {file}")
            trained_states = self.load_states_from_file(file)

            if len(trained_states) != 0:
                self.logger.info(f"Loaded {len(trained_states)} states from: {file}")

        # train states if could not be loaded
        if len(trained_states) == 0:
            self.logger.info(f"Training Reservoir of {self.reservoir.units} nodes with {training_instances} instances")
            start = time.time()
            trained_states = self.train(self.X_train)
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
        Y_pred = self.predict(self.X_test)
        end = time.time()
        self.logger.debug(f"Prediction Time Elapsed: {str(round(end - start, 4))}s")

        # Calculate performance metrics
        self.logger.info("Calculating model performance metrics")
        Y_test_class = np.array([np.argmax(y_t) for y_t in self.Y_test])
        Y_pred_class = np.array([np.argmax(y_p) for y_p in Y_pred])
        
        return self.evaluation(Y_test_class, Y_pred_class)
        
    def evaluation(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        accuracy = accuracy_score(Y_true, Y_pred) * 100
        f1 = f1_score(Y_true, Y_pred, average='weighted')
        recall = recall_score(Y_true, Y_pred, average='weighted')
        precision = precision_score(Y_true, Y_pred, average='weighted')
        mse = np.mean((Y_true - Y_pred) ** 2)
        rmse = np.sqrt(mse)
        r_squared = r2_score(Y_true, Y_pred)
        conf_matrix = confusion_matrix(Y_true, Y_pred)

        return {
            "accuracy": accuracy, "f1": f1, "recall": recall, 
            "precision": precision, "mse": mse, "rmse": rmse, 
            "r_squared": r_squared, "confusion_matrix": conf_matrix
        }

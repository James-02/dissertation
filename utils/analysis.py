from typing import List, Dict, Tuple, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, classification_report

from utils.logger import Logger

logger = Logger()

def log_metrics(metrics: Dict[str, Any]) -> None:
    """
    Log classification metrics.

    Args:
        metrics (dict): Dictionary containing classification metrics.
    """
    logger.info("----- Classification Report -----")
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            logger.info(f"{key}: {value:.3f}")
    logger.info("---------------------------------")

def log_params(params: Dict[str, Any], title: str = "Hyperparameters") -> None:
    """
    Log model hyperparameters.

    Args:
        params (dict): Dictionary containing hyperparameters.
        title (str): Title for the hyperparameters section. Default is "Hyperparameters".
    """
    logger.debug(f"----- {title} -----")
    for key, value in params.items():
        logger.debug(f"{key}: {value}")
    logger.debug("--------------------------------")

def evaluate_performance(Y_true: np.ndarray, Y_pred: np.ndarray, time: float = 0) -> Dict[str, any]:
    """
    Evaluate the performance of a classification model using different accuracy and loss based metrics.

    Also produce F1 score metrics for each class, and the corresponding confusion matrix.

    Args:
        Y_true (np.ndarray): True labels.
        Y_pred (np.ndarray): Predicted labels.
        time (float): Runtime of the model (default is 0).

    Returns:
        dict: Dictionary containing various performance metrics.
    """
    logger.info("Calculating model performance metrics")
    Y_true = np.array([np.argmax(y_t) for y_t in Y_true])
    Y_pred = np.array([np.argmax(y_p) for y_p in Y_pred])

    return {
        "runtime": time,
        "accuracy": accuracy_score(Y_true, Y_pred),
        "f1": f1_score(Y_true, Y_pred, average='weighted'),
        "recall": recall_score(Y_true, Y_pred, average='weighted'),
        "precision": precision_score(Y_true, Y_pred, average='weighted'),
        "mse": np.mean((Y_true - Y_pred) ** 2),
        "rmse": np.sqrt(np.mean((Y_true - Y_pred) ** 2)),
        "r2_score": r2_score(Y_true, Y_pred),
        "confusion_matrix": confusion_matrix(Y_true, Y_pred),
        "class_metrics": classification_report(Y_true, Y_pred, output_dict=True)
    }

def compute_mean_dicts(dicts_list: List[Dict]) -> Dict[str, Dict]:
    """
    Compute the mean of dictionaries for integer based keys, representing each class in the dataset.

    Args:
        dicts_list (list): List of dictionaries to compute mean from.

    Returns:
        dict: Mean values for each key across dictionaries.
    """
    # check if the dictionary key is an integer, extracting just the metrics for each individual class
    valid_keys = [str(i) for i in range(100)]
    keys = list(filter(None, [str(d_key) if str(d_key) in valid_keys else None for d in dicts_list for d_key in d.keys()]))
    sum_values = {key: {inner_key: 0 for inner_key in keys} for key in keys}
    count = len(dicts_list)

    # compute the average of each metric across each class
    for d in dicts_list:
        for key, value in d.items():
            if key in keys and isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    if inner_key in sum_values[key].keys():
                        sum_values[key][inner_key] += inner_value
                    else:
                        sum_values[key][inner_key] = inner_value
    return {key: {inner_key: sum_value / count for inner_key, sum_value in inner_sum_values.items()} for key, inner_sum_values in sum_values.items()}

def compute_mean_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the mean of classification metrics, recursively averages the fold and individual class metrics.

    Args:
        metrics (dict): Dictionary containing classification metrics.

    Returns:
        dict: Mean values for each metric.
    """
    class_metrics = [d.pop("class_metrics") for d in metrics]
    sum_metrics = {key: 0 for key in metrics[0].keys()}
    count = len(metrics)

    for metric in metrics:
        for key, value in metric.items():
            sum_metrics[key] += value

    mean_metrics = {key: sum_value / count for key, sum_value in sum_metrics.items()}
    mean_metrics["class_metrics"] = compute_mean_dicts(class_metrics)
    return mean_metrics

def count_labels(Y: List[np.ndarray]) -> Dict:
    """
    Count the occurrences of each class label, converts one-hot encoding if given.

    Args:
        Y (list of numpy.ndarray): List of target arrays.

    Returns:
        dict: Count of occurrences for each class label.
    """
    Y = np.array([targets[0] for targets in Y])
    if len(Y.shape) != 1:
        Y = np.argmax(Y, axis=1)
    return {label: count for label, count in zip(*np.unique(Y, return_counts=True))}

def measure_dataset_deviation(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure the mean and standard deviation of features in a dataset.

    Args:
        X (numpy.ndarray): Input data.

    Returns:
        tuple: Tuple containing mean and standard deviation arrays.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

def measure_class_deviation(X: List[np.ndarray], Y: List[np.ndarray]) -> Tuple[Dict, Dict]:
    """
    Measure the mean and standard deviation of features for each class in a dataset.

    Args:
        X (list of numpy.ndarray): List of feature arrays.
        Y (list of numpy.ndarray): List of target arrays.

    Returns:
        tuple: Tuple containing dictionaries of class means and standard deviations.
    """
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    class_means = {}
    class_stds = {}

    for label in np.unique(Y):
        indices = np.where(Y == label)[0]
        class_means[label] = round(np.mean(X[indices], axis=0).tolist()[0], 4)
        class_stds[label] = round(np.std(X[indices], axis=0).tolist()[0], 4)

    return class_means, class_stds

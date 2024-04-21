from typing import List

import numpy as np

from utils.logger import Logger

logger = Logger()

def log_metrics(metrics: dict):
    logger.info("----- Classification Report -----")
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            logger.info(f"{key}: {value:.3f}")
    logger.info("---------------------------------")

def log_params(params, title="Hyper-parameters"):
    logger.debug(f"----- {title} -----")
    for key, value in params.items():
        logger.debug(f"{key}: {value}")
    logger.debug("--------------------------------")

def aggregate_dicts(dicts_list, keys):
    sum_values = {key: {inner_key: 0 for inner_key in keys} for key in keys}
    count = len(dicts_list)

    for d in dicts_list:
        for key, value in d.items():
            if key in keys and isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    if inner_key in sum_values[key].keys():
                        sum_values[key][inner_key] += inner_value
                    else:
                        sum_values[key][inner_key] = inner_value
    mean_values = {key: {inner_key: sum_value / count for inner_key, sum_value in inner_sum_values.items()} for key, inner_sum_values in sum_values.items()}
    
    return mean_values

def compute_mean_metrics(metrics):
    class_metrics = [d.pop("class_metrics") for d in metrics]
    sum_metrics = {key: 0 for key in metrics[0].keys()}
    count = len(metrics)

    for metric in metrics:
        for key, value in metric.items():
            sum_metrics[key] += value

    mean_metrics = {key: sum_value / count for key, sum_value in sum_metrics.items()}
    mean_metrics['class_metrics'] = aggregate_dicts(class_metrics, ['0', '1', '2', '3', '4'])
    return mean_metrics

def count_labels(Y: List) -> dict:
    # Always use the first row each target, in case the targets are repeated
    Y = np.array([targets[0] for targets in Y])

    # If Y is encoded, revert the encoding
    if len(Y.shape) != 1:
        Y = np.argmax(Y, axis=1)

    return {label: count for label, count in zip(*np.unique(Y, return_counts=True))}


def measure_dataset_deviation(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

def measure_class_deviation(X, Y):
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    class_means = {}
    class_stds = {}
    
    # Calculate mean and std for each class
    for label in np.unique(Y):
        indices = np.where(Y == label)[0]
        class_means[label] = round(np.mean(X[indices], axis=0).tolist()[0], 4)
        class_stds[label] = round(np.std(X[indices], axis=0).tolist()[0], 4)

    return class_means, class_stds

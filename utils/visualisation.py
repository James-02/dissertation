from typing import List
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from utils.analysis import count_labels, measure_class_deviation

# set global visualisation config
sns.set_style("ticks")

plt.rcParams.update({'font.size': 15})

DPI = 800
RESULTS_DIR = "results/"

binary_classes = ['Normal', 'Arrhythmia']
binary_palette = sns.color_palette("husl", n_colors=len(binary_classes))[::-1]

classes = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion', 'Unknown']
categorical_palette = sns.color_palette("husl", n_colors=len(classes))[::-1]

def _save_figure(filename: str):
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches="tight", dpi=DPI)

def plot_states(states: np.ndarray, labels: List[str] = None, title: str = "", xlabel: str = "Time", 
         ylabel: str = "State", filename: str = "states.png", legend: bool = True, show: bool = True):
    """
    Plot a generic line graph.

    Args:
        states (np.ndarray): Time-series states to plot.
        labels (List[str]): Labels for the data.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str): Name of the file to save.
        legend (bool): Whether to show the legend.
        show (bool): Whether to display the plot.
    """
    timesteps = len(states)
    time = np.linspace(0, timesteps, timesteps)

    plt.figure(figsize=(10, 6))
    for i in range(0, states.shape[1]):
        plt.plot(time, states[:, i], label=labels[i] if labels else None)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if legend:
        plt.legend()
    _save_figure(filename=os.path.join("states/", filename))
    
    if show:
        plt.show()

def plot_data_distribution(Y: list, filename: str = "data-distribution.png", show: bool = True):
    label_counts = count_labels(Y)
    if len(label_counts) > 2:
        class_labels = [classes[label] for label in label_counts.keys()]
        colors = categorical_palette
    else:
        class_labels = [binary_classes[label] for label in label_counts.keys()]
        colors = binary_palette

    _, ax = plt.subplots(figsize=(12, 6))
    _, _, autotexts = ax.pie(label_counts.values(), labels=class_labels, autopct='%1.1f%%', colors=colors, textprops=dict(color="black"))
    for autotext in autotexts:
        autotext.set_color('black')
    plt.tight_layout()

    _save_figure(filename=os.path.join("preprocessing/", filename))

    if show:
        plt.show()

def plot_dataset_info(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, show=True, filename="dataset-table.png") -> None:
    train_label_counts = count_labels(Y_train)
    test_label_counts = count_labels(Y_test)

    # Create figure and axis
    _, ax = plt.subplots(figsize=(10, 3))

    # Create table
    table_data = [
        ["", "Training", "Testing"],
        ["Instances", len(X_train), len(X_test)],
        ["Targets Shape", str(X_train[0].shape), str(Y_train[0].shape)],
        ["Labels Shape", str(X_test[0].shape), str(Y_test[0].shape)],
        ["Class Size", train_label_counts[0], test_label_counts[0]]
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2])

    # Hide axes
    ax.axis('off')

    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    _save_figure(filename=os.path.join("preprocessing/", filename))
    if show:
        plt.show()

def plot_class_std(X, Y, show=True, filename="class_std.png"):
    # if one-hot encoded, reverse encoding
    if len(Y.shape) > 2:
        Y = np.array([np.argmax(y, axis=1) for y in Y])

    _, class_std = measure_class_deviation(X, Y)
    mean_class_std = {label: np.mean(std) for label, std in class_std.items()}

    if len(class_std) > 2:
        class_labels = [classes[label] for label in class_std.keys()]
        colors = categorical_palette
    else:
        class_labels = [binary_classes[label] for label in class_std.keys()]
        colors = binary_palette

    _, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.5
    ax.bar(np.arange(len(class_labels)), mean_class_std.values(), color=colors, width=bar_width)
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=20)
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Standard Deviation')
    plt.tight_layout()

    _save_figure(filename=os.path.join("preprocessing/", filename))
    if show:
        plt.show()

def plot_class_mean(X, Y, show=True, filename="class_means.png"):
    # if one-hot encoded, reverse encoding
    if len(Y.shape) > 2:
        Y = np.array([np.argmax(y, axis=1) for y in Y])

    class_means, _ = measure_class_deviation(X, Y)
    mean_class_means = {label: np.mean(means) for label, means in class_means.items()}

    if len(class_means) > 2:
        class_labels = [classes[label] for label in class_means.keys()]
        colors = categorical_palette
    else:
        class_labels = [binary_classes[label] for label in class_means.keys()]
        colors = binary_palette

    _, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.5
    ax.bar(np.arange(len(class_labels)), mean_class_means.values(), color=colors, width=bar_width)
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=20)
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Value')
    plt.tight_layout()

    _save_figure(filename=os.path.join("preprocessing/", filename))
    if show:
        plt.show()

def plot_average_instance(X, Y, show=True, filename="average_instances.png"):
    # if one-hot encoded, reverse encoding
    if len(Y.shape) > 2:
        Y = np.array([np.argmax(y, axis=1) for y in Y])

    # Group instances by class
    class_instances = {}
    for x, label in zip(X, Y):
        label = int(label)
        if label not in class_instances:
            class_instances[label] = []
        class_instances[label].append(x.flatten())

    # Calculate mean for each timestep separately across all instances within each class
    class_means = {label: np.mean(instances, axis=0).flatten() for label, instances in class_instances.items()}

    # Sort class_means dictionary by keys
    class_means = dict(sorted(class_means.items()))

    if len(class_means) > 2:
        class_labels = [classes[label] for label in class_means.keys()]
        colors = categorical_palette
    else:
        class_labels = [binary_classes[label] for label in class_means.keys()]
        colors = binary_palette

    # Plot mean values for each timestep
    plt.figure(figsize=(12, 6))
    for i, (label, mean_values) in enumerate(class_means.items()):
        plt.plot(mean_values, label=class_labels[i], color=colors[i])
    plt.xlabel('Timestep')
    plt.ylabel('Mean Value')
    plt.tight_layout()
    plt.legend()

    _save_figure(filename=os.path.join("preprocessing/", filename))
    if show:
        plt.show()

def plot_confusion_matrix(confusion_matrix, cmap=plt.cm.Blues, show=True, filename="confusion_matrix.png"):
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    labels = confusion_matrix.shape[0]
    if labels > 2:
        class_labels = [classes[i] for i in range(labels)]
    else:
        class_labels = [binary_classes[i] for i in range(labels)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap=cmap, xticklabels=class_labels, yticklabels=class_labels)

    # Adjust font size and rotation for better readability
    plt.xticks(rotation=30, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)

    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()

def plot_metrics_across_folds(metrics, metric_names=["accuracy", "mse", "rmse", "f1"], show=True, filename="metrics_folds.png"):
    num_folds = len(metrics)
    num_metrics = len(metric_names)
    bar_width = 0.2
    index = np.arange(num_folds)

    plt.figure(figsize=(12, 6))
    for i in range(num_metrics):
        metric_values = [metrics[j][metric_names[i]] for j in range(num_folds)]
        plt.bar(index + i * bar_width, metric_values, bar_width, label=metric_names[i])

    plt.xlabel('Fold')
    plt.ylabel('Value')
    plt.xticks(index + bar_width * (num_metrics - 1) / 2, range(1, num_folds + 1))
    plt.legend()
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()

def plot_class_metrics(metrics, filename="class-metrics.png", show=True):
    # Extract metrics for each class
    class_metrics = {}
    for key in metrics.keys():
        # Check if the key represents an individual class (numeric string)
        if key.isdigit():
            class_metrics[key] = metrics[key]
    precisions = [metrics[i]['precision'] for i in class_metrics.keys()]
    recalls = [metrics[i]['recall'] for i in class_metrics.keys()]
    f1_scores = [metrics[i]['f1-score'] for i in class_metrics.keys()]

    class_labels = classes if len(class_metrics.keys()) > 2 else binary_classes

    # Create x-axis ticks and labels
    x_ticks = range(len(class_labels))
    x_labels = [class_labels[int(label)] for label in class_metrics.keys()]

    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.bar(x_ticks, precisions, width=0.2, label='Precision', align='center')
    plt.bar([x + 0.2 for x in x_ticks], recalls, width=0.2, label='Recall', align='center')
    plt.bar([x + 0.4 for x in x_ticks], f1_scores, width=0.2, label='F1-score', align='center')

    # Add labels and legend
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks([x + 0.2 for x in x_ticks], x_labels, rotation=45, ha='right')
    plt.legend()

    # Show plot
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))

    if show:
        plt.show()

def plot_noise(X, noise, noisy_X, show=True):
    sampling_freq = 125
    num_samples = len(X)
    time_ms = np.arange(num_samples) / sampling_freq * 1000

    plt.figure(figsize=(10, 6))
    plt.plot(time_ms, X, label="Training Data", linestyle="solid", color="orange")
    plt.plot(time_ms, noise, linestyle="dotted", label="Noise", color="green")
    plt.plot(time_ms, noisy_X, linestyle="dashed", label="Augmented Data", color="blue")
    plt.ylabel('Normalized Value')
    plt.xlabel('Time (ms)')
    plt.xticks()
    plt.yticks()
    plt.legend()

    if show:
        plt.show()


def plot_tsne_clustering(Y_pred, show=True):
    Y_pred_flat = np.array(Y_pred).reshape(len(Y_pred), -1)
    
    # Perform t-SNE embedding
    tsne = TSNE(n_components=2)
    tsne_embeddings = tsne.fit_transform(Y_pred_flat)

    Y_pred = np.array([np.argmax(y_p) for y_p in Y_pred]).reshape(-1, 1)

    ticks = np.unique(Y_pred)

    class_names = classes
    colors = ListedColormap(categorical_palette)
    if len(ticks) == 2:
        class_names = binary_classes
        colors = ListedColormap(binary_palette)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], c=Y_pred, cmap=colors)
    cbar = plt.colorbar(scatter, ticks=ticks)
    cbar.set_ticklabels([class_names[t] for t in ticks])
    cbar.set_label('Class')

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    _save_figure(filename=os.path.join("metrics/", "test.png"))
    if show:
        plt.show()

def plot_forecast_data(X_train, Y_train, X_test, Y_test, sample=500, show=True, filename="forecast_data.png"):
    plt.plot(np.arange(sample), X_train[-sample:], label="Training data")
    plt.plot(np.arange(sample), Y_train[-sample:], label="Training ground truth")
    plt.plot(np.arange(sample, sample + len(X_test)), X_test, label="Testing data")
    plt.plot(np.arange(sample, sample + len(Y_test)), Y_test, label="Testing ground truth")
    plt.legend()

    _save_figure(filename=os.path.join("forecasting/", filename))
    if show:
        plt.show()

def plot_forecast_results(y_pred, y_test, sample=500, show=True, filename="forecast_prediction.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN Prediction")
    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True Value")
    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Forecasting")

    _save_figure(filename=os.path.join("forecasting/", filename))
    if show:
        plt.show()

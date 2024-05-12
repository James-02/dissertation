from typing import List, Dict
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from utils.analysis import count_labels, measure_class_deviation

# Set global visualization config
sns.set_style("ticks")

plt.rcParams.update({'font.size': 15})

DPI = 800
RESULTS_DIR = "results/"

# Define class labels and palettes
binary_classes = ['Normal', 'Arrhythmia']
binary_palette = sns.color_palette("husl", n_colors=len(binary_classes))[::-1]

classes = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
categorical_palette = sns.color_palette("husl", n_colors=len(classes))[::-1]

def _save_figure(filename: str) -> None:
    """
    Save the current figure with a given filename.

    Args:
        filename (str): The name of the file to save.
    """
    file_path = os.path.join(RESULTS_DIR, filename)
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(file_path, bbox_inches="tight", dpi=DPI)

def plot_states(states: np.ndarray, labels: List[str] = None, xlabel: str = "Time", 
         ylabel: str = "State", filename: str = "states.png", legend: bool = True, show: bool = True) -> None:
    """
    Plot a generic line graph of the evolution of system states over time.

    Args:
        states (np.ndarray): Time-series states to plot.
        labels (List[str], optional): Labels for the data.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        filename (str, optional): Name of the file to save.
        legend (bool, optional): Whether to show the legend.
        show (bool, optional): Whether to display the plot.
    """
    timesteps = len(states)
    time = np.linspace(0, timesteps, timesteps)

    plt.figure(figsize=(10, 6))
    for i in range(states.shape[1]):
        plt.plot(time, states[:, i], label=labels[i] if labels else None)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if legend:
        plt.legend()
    _save_figure(filename=os.path.join("states/", filename))
    
    if show:
        plt.show()

def plot_data_distribution(Y: list, filename: str = "data-distribution.png", show: bool = True):
    """
    Generate a pie chart to visualize the distribution of data labels.

    Args:
        Y (list): List of data labels.
        filename (str, optional): Filename to save the generated plot. Default is "data-distribution.png".
        show (bool, optional): Whether to display the plot. Default is True.
    """
    label_counts = count_labels(Y)
    if len(label_counts) > 2:
        class_labels = [classes[label] for label in label_counts.keys()]
        colors = categorical_palette
    else:
        class_labels = [binary_classes[label] for label in label_counts.keys()]
        colors = binary_palette

    plt.rcParams.update({'font.size': 14})
    _, ax = plt.subplots(figsize=(14, 8))
    _, _, autotexts = ax.pie(label_counts.values(), autopct='%1.1f%%', colors=colors)

    for autotext in autotexts:
        autotext.set_color('black')

    plt.legend(class_labels, loc="best", fontsize='large')
    plt.tight_layout()

    _save_figure(filename=os.path.join("preprocessing/", filename))

    if show:
        plt.show()

def plot_dataset_info(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, 
                      show=True, filename="dataset-table.png") -> None:
    """
    Create a table to display information about the dataset.

    Args:
        X_train (np.ndarray): Training data.
        Y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing data.
        Y_test (np.ndarray): Testing labels.
        show (bool, optional): Whether to display the table. Default is True.
        filename (str, optional): Filename to save the generated table. Default is "dataset-table.png".
    """
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

def plot_class_std(X: np.ndarray, Y: np.ndarray, show: bool = True, filename: str = "class_std.png"):
    """
    Plot the mean standard deviation of each class.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Labels.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "class_std.png".
    """
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

def plot_class_mean(X: np.ndarray, Y: np.ndarray, show: bool = True, filename: str = "class_means.png") -> None:
    """
    Plot the mean value of each class.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Labels.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "class_means.png".
    """
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

def plot_average_instance(X: np.ndarray, Y: np.ndarray, show: bool = True, filename: str = "average_instances.png") -> None:
    """
    Plot the average instance for each class.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Labels.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "average_instances.png".
    """
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

def plot_confusion_matrix(confusion_matrix: np.ndarray, cmap=plt.cm.Blues, 
                          show: bool = True, filename: str = "confusion_matrix.png") -> None:
    """
    Plot the confusion matrix.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        cmap: Colormap.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "confusion_matrix.png".
    """
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

    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()

def plot_metrics_across_folds(metrics: List[Dict[str, any]], metric_names: List[str] = ["accuracy", "mse", "rmse", "f1"], 
                              show: bool = True, filename: str = "metrics_folds.png") -> None:
    """
    Plot the metrics across different folds.

    Args:
        metrics (List[Dict[str, any]]): List of dictionaries containing metric values for each fold.
        metric_names (List[str], optional): List of metric names to plot. Default is ["accuracy", "mse", "rmse", "f1"].
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "metrics_folds.png".
    """
    num_folds = len(metrics)
    num_metrics = len(metric_names)
    bar_width = 0.2
    index = np.arange(num_folds)

    plt.figure(figsize=(12, 6))
    for i in range(num_metrics):
        metric_values = [metric[metric_names[i]] for metric in metrics]
        plt.bar(index + i * bar_width, metric_values, bar_width, label=metric_names[i])

    plt.xlabel('Fold')
    plt.ylabel('Value')
    plt.xticks(index + bar_width * (num_metrics - 1) / 2, range(1, num_folds + 1))
    plt.legend()
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()

def plot_class_metrics(metrics: Dict[str, Dict[str, float]], show: bool = True, filename: str = "class_metrics.png") -> None:
    """
    Plot the metrics for each class.

    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary containing metrics for each class.
        filename (str, optional): Filename to save figure to. Default is "class_metrics.png".
        show (bool, optional): Whether to display the plot. Default is True.
    """
    # Extract metrics for each class
    class_metrics = {key: value for key, value in metrics.items() if key.isdigit()}
    precisions = [metric['precision'] for metric in class_metrics.values()]
    recalls = [metric['recall'] for metric in class_metrics.values()]
    f1_scores = [metric['f1-score'] for metric in class_metrics.values()]

    class_labels = classes if len(class_metrics) > 2 else binary_classes

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

def plot_readout_weights(Wout: np.ndarray, show: bool = True, filename: str = "readout_weights.png") -> None:
    """
    Plot the readout weight coefficients for each class.

    Args:
        Wout (np.ndarray): Readout weights.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save figure to. Default is "readout_weights.png".
    """
    scaler = StandardScaler()
    Wout_scaled = scaler.fit_transform(Wout)
    num_classes = Wout_scaled.shape[1]

    # Plotting
    plt.figure(figsize=(12, 6))

    # Use a violin plot to visualize the distribution of weights for each class
    sns.violinplot(data=Wout_scaled, inner=None, linewidth=2, palette=categorical_palette)

    # Plot mean and standard deviation as horizontal lines
    means = np.mean(Wout_scaled, axis=0)
    stds = np.std(Wout_scaled, axis=0)
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.plot([i - 0.2, i + 0.2], [mean, mean], color='black', linewidth=3)  # Mean line
        plt.plot([i, i], [mean - std, mean + std], color='black', linewidth=1, linestyle="dotted")  # Standard deviation lines

    # Create custom legend with unique labels
    plt.plot([], [], color='black', linewidth=3, label="Mean")
    plt.plot([], [], color='black', linewidth=1, linestyle="dotted", label="Std. Deviation")

    # Set labels and title
    plt.xlabel('Class')
    plt.ylabel('Scaled Wout')

    # Set x-axis ticks and labels
    plt.xticks(np.arange(num_classes), [classes[i] for i in range(num_classes)])
    plt.legend()
    plt.tight_layout()

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()

def plot_noise(X: np.ndarray, noise: np.ndarray, noisy_X: np.ndarray, show: bool = True, filename: str = "noise.png") -> None:
    """
    Plot the noise in training data.

    Args:
        X (np.ndarray): Training data.
        noise (np.ndarray): Noise data.
        noisy_X (np.ndarray): Augmented data.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save figure to. Default is "noise.png".
    """
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

    _save_figure(filename=os.path.join("preprocessing/", filename))

    if show:
        plt.show()

def plot_tsne_clustering(Y_pred: np.ndarray, Y_true: np.ndarray, show: bool = True, filename: str = "clustering.png") -> None:
    """
    Plot the t-SNE clustering of predicted and true labels.

    Args:
        Y_pred (Union[List[np.ndarray], np.ndarray]): Predicted labels.
        Y_true (Union[List[np.ndarray], np.ndarray]): True labels.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "clustering.png".
    """
    Y_pred_flat = np.array(Y_pred).reshape(len(Y_pred), -1)
    
    # Perform t-SNE embedding
    tsne = TSNE(n_components=2)
    tsne_embeddings = tsne.fit_transform(Y_pred_flat)

    Y_pred = np.array([np.argmax(y_p) for y_p in Y_pred]).reshape(-1, 1)
    Y_true = np.array([np.argmax(y_t) for y_t in Y_true]).reshape(-1, 1)

    ticks = np.unique(Y_pred)

    class_names = classes
    colors = ListedColormap(categorical_palette)
    if len(ticks) == 2:
        class_names = binary_classes
        colors = ListedColormap(binary_palette)

    # Identify correct and incorrect predictions
    correct_preds = (Y_pred == Y_true)
    incorrect_preds = ~correct_preds

    plt.figure(figsize=(10, 8))

    # Create empty scatter plots with black markers for legend
    scatter_correct = plt.scatter([], [], marker='o', label='Correct Predictions', color='k')
    scatter_incorrect = plt.scatter([], [], marker='x', label='Incorrect Predictions', color='k')

    # Plot correct predictions
    plt.scatter(x=tsne_embeddings[correct_preds[:, 0], 0], 
                y=tsne_embeddings[correct_preds[:, 0], 1], 
                c=Y_true[correct_preds[:, 0]], cmap=colors, marker='o')

    # Plot incorrect predictions
    plt.scatter(x=tsne_embeddings[incorrect_preds[:, 0], 0], 
                y=tsne_embeddings[incorrect_preds[:, 0], 1], 
                c=Y_true[incorrect_preds[:, 0]], cmap=colors, marker='x')

    plt.legend(handles=[scatter_correct, scatter_incorrect])

    # Add color bar representing class labels
    cbar = plt.colorbar(ticks=ticks)
    cbar.set_ticklabels([class_names[int(t)] for t in ticks])
    cbar.set_label('Class')

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    _save_figure(filename=os.path.join("metrics/", filename))
    if show:
        plt.show()

def plot_forecast_data(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, 
                       sample: int = 500, show: bool = True, filename: str = "forecast_data.png") -> None:
    """
    Plot the forecast data.

    Args:
        X_train (np.ndarray): Training data.
        Y_train (np.ndarray): Training ground truth.
        X_test (np.ndarray): Testing data.
        Y_test (np.ndarray): Testing ground truth.
        sample (int, optional): Number of samples to plot. Default is 500.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "forecast_data.png".
    """
    plt.plot(np.arange(sample), X_train[-sample:], label="Training data")
    plt.plot(np.arange(sample), Y_train[-sample:], label="Training ground truth")
    plt.plot(np.arange(sample, sample + len(X_test)), X_test, label="Testing data")
    plt.plot(np.arange(sample, sample + len(Y_test)), Y_test, label="Testing ground truth")
    plt.legend()

    _save_figure(filename=os.path.join("forecasting/", filename))
    if show:
        plt.show()

def plot_forecast_results(y_pred: np.ndarray, y_test: np.ndarray, sample: int = 500,
                          show: bool = True, filename: str = "forecast_prediction.png") -> None:
    """
    Plot the forecast results.

    Args:
        y_pred (np.ndarray): Predicted values.
        y_test (np.ndarray): True values.
        sample (int, optional): Number of samples to plot. Default is 500.
        show (bool, optional): Whether to display the plot. Default is True.
        filename (str, optional): Filename to save the plot. Default is "forecast_prediction.png".
    """
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

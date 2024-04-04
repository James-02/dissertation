from typing import List

import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self, results_path: str = "results/", style: str = "whitegrid", dpi: int = 800):
        """
        Initialize the Visualizer.

        Args:
            results_path (str): Path to store the visualizations.
            style (str): Seaborn style to apply.
            dpi (int): Dots per inch for figure resolution.
        """
        self.results_path = results_path
        self.dpi = dpi

        sns.set_style(style)

    def _save_figure(self, filename: str):
        """
        Save the current figure to a file.

        Args:
            filename (str): Name of the file to save.
        """
        plt.savefig(os.path.join(self.results_path, filename), bbox_inches="tight", dpi=self.dpi)

    def plot_states(self, states: np.ndarray, labels: List[str], title: str = "States", xlabel: str = "Time", 
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
            show
        """
        timesteps = len(states)
        time = np.linspace(0, timesteps, timesteps)
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            plt.plot(time, states[:, i], label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()

        if legend:
            plt.legend()
        self._save_figure(filename=os.path.join("states/", filename))
    
        if show:
            plt.show()

    def plot_data_distribution(self, counts: list, labels: list, filename: str = "data-distribution.png", show: bool = True):
        """
        Plot the distribution of data classes.

        Args:
            counts (list): List of counts for each class.
            labels (list): List of labels, must align with counts.
            filename (str): Filename to save figure to.
            show (bool): Flag to show figure
        """
        _, ax = plt.subplots(figsize=(16, 8))
        ax.set_title('ECG Dataset Class Distribution')
        ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['red', 'orange', 'blue', 'magenta', 'cyan'])
        self._save_figure(filename=os.path.join("preprocessing/", filename))
    
        if show:
            plt.show()

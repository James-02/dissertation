import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

def plot_ecg(time, signal, idx):
    classes = ['Normal','Unknown','Ventricular ectopic','Supraventricular ectopic', 'Fusion']
    plt.figure(figsize=(10, 6))

    plt.plot(time, signal)

    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('ECG Signal Evolution')
    plt.legend([classes[idx // len(classes)]])

    plt.tight_layout()
    plt.savefig(f"results/ecg-signal-{str(idx)}.png", bbox_inches="tight", dpi=500)


def plot_ecg_gif(time, signal, idx):
    classes = ['Normal','Unknown','Ventricular ectopic','Supraventricular ectopic', 'Fusion']
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set labels, title, and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('ECG Signal')
    ax.set_title('ECG Signal Over Time')

    # Plot initial states and store lines in variables
    signal_, = ax.plot(time[0], signal[0, 0], label='A')
    ax.legend(["Normal"])

    # Set x-axis and y-axis limits
    ax.set_xlim(0, time[-1])
    ax.set_ylim(np.min(signal), np.max(signal))

    fig.tight_layout()

    def update(frame):
        # Update data of the lines representing the states
        signal_.set_data(time[:frame], signal[:frame])
        return [signal_]

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(signal), repeat=False, interval=800)

    # Save animation as GIF
    ani.save(f'results/ecg-animation-{idx}.gif', writer="pillow", fps=60)    

def plot_states_gif(time, states):
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set labels, title, and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('Oscillator States Evolution')

    # Plot initial states and store lines in variables
    a, = ax.plot(time[0], states[0, 0], label='A')
    i, = ax.plot(time[0], states[0, 1], label='I')
    hi, = ax.plot(time[0], states[0, 2], label='Hi')
    he, = ax.plot(time[0], states[0, 3], label='He')
    ax.legend(["A", "I", "Hi", "He"])

    # Set x-axis and y-axis limits
    ax.set_xlim(0, time[-1])
    ax.set_ylim(np.min(states), np.max(states))

    fig.tight_layout()

    def update(frame):
        # Update data of the lines representing the states
        a.set_data(time[:frame], states[:frame, 0])
        i.set_data(time[:frame], states[:frame, 1])
        hi.set_data(time[:frame], states[:frame, 2])
        he.set_data(time[:frame], states[:frame, 3])
        return [a, i, hi, he]

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(states), repeat=False, interval=800)

    # Save animation as GIF
    ani.save('results/states-animation.gif', writer="pillow", fps=60)

def plot_states(time, states):
    plt.figure(figsize=(6, 6))

    plt.plot(time, states[:, 0], label='A')
    plt.plot(time, states[:, 1], label='I')
    plt.plot(time, states[:, 2], label='Hi')
    plt.plot(time, states[:, 3], label='He')

    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('Oscillator States Evolution')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/states.png", bbox_inches="tight", dpi=800)

def plot_data_distribution(data):
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot training data distribution
    ax.set_title('ECG Dataset Class Distribution')
    ax.pie(data.iloc[:, -1].value_counts(), 
           labels=['Normal', 'Unknown', 'Ventricular ectopic', 'Supraventricular ectopic', 'Fusion'], 
           autopct='%1.1f%%', 
           colors=['red', 'orange', 'blue', 'magenta', 'cyan'])

    plt.savefig("results/original-class-distribution.png", dpi=800, bbox_inches="tight")

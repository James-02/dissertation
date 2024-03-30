import numpy as np

from reservoirpy.node import Node

from .node import Oscillator

def weight_coupling(reservoir: Node) -> np.ndarray:
    coupling_matrix = np.full((len(reservoir.nodes), 1), reservoir.hypers['coupling'])
    min_percent = 0.1
    max_percent = 1000

    # calculate the range of coupling values
    min_value = reservoir.hypers['coupling'] * min_percent / 100
    max_value = reservoir.hypers['coupling'] * max_percent / 100

    # apply random noise to the connection weights
    reservoir.W += reservoir.hypers['noise_generator'](dist=reservoir.hypers['noise_type'], 
                                                       shape=reservoir.W.shape, gain=reservoir.hypers['noise_rc'])

    # calculate each node's total neighbourhood weight
    weighted_coupling = reservoir.W @ coupling_matrix

    # normalise weights within the coupling range
    return normalize_array(weighted_coupling, (min_value, max_value))

def initialize_nodes(reservoir: Node):
    return [Oscillator(timesteps=reservoir.timesteps) for _ in range(reservoir.hypers['units'])]

def compute_weight_matrix(units: int, decay_rate: float) -> np.ndarray:
    node_indices = np.arange(units)
    distances = np.abs(node_indices[:, None] - node_indices[None, :])
    return np.exp(-decay_rate * distances)

def weight_input(reservoir: Node, x: np.ndarray) -> np.ndarray:
    u = x.reshape(-1, 1)

    # influence the input with the weights and noise
    weighted_input = reservoir.params['Win'] @ (u + reservoir.hypers['noise_generator'](
                            dist=reservoir.hypers['noise_type'], shape=u.shape, gain=reservoir.hypers['noise_in'])) + reservoir.bias,

    # normalise input between [0-1]    
    return normalize_array(weighted_input, (0, 1))

def normalize_array(array: np.ndarray, range: tuple) -> np.ndarray:
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val) * (range[1] - range[0]) + range[0]


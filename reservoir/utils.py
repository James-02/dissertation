import numpy as np

from reservoirpy.node import Node

def weight_input(reservoir: Node, x: np.ndarray) -> np.ndarray:
    u = x.reshape(-1, 1)

    # Apply input weights and noise
    noise = reservoir.noise_generator(dist=reservoir.noise_type, shape=u.shape, gain=reservoir.noise_in)
    weighted_input = reservoir.Win @ (u + noise) + reservoir.bias

    # Apply coupling
    return weighted_input * reservoir.coupling

def weight_previous_state(reservoir: Node):
    r = reservoir.state().T
    noise = reservoir.noise_generator(dist=reservoir.noise_type, shape=r.shape, gain=reservoir.noise_rc)
    return (reservoir.W @ (r + noise)) * reservoir.rc_scaling


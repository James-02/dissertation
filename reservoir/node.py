from functools import partial

import numpy as np
import math

from .dde import ddeint, dde_system, interpolate_history

from reservoirpy.node import Node

# Default hyperparameters for the oscillator
DEFAULT_HYPERS = {
    'CA': 1,
    'CI': 4,
    'del_': 1e-3, 
    'alpha': 2500, 
    'k': 1, 
    'k1': 0.1, 
    'b': 0.06, 
    'gammaA': 15, 
    'gammaI': 24, 
    'gammaH': 0.01, 
    'f': 0.3, 
    'g': 0.01, 
    'd': 0.7, 
    'd0': 0.88, 
    'D': 2.5, 
    'mu': 0.6,
    'delay': 10,
    'time': np.linspace(0, 1, 2),
    'initial_conditions': [0, 100, 0, 0]
}

def forward_oscillator(node: Node, x: np.ndarray, **kwargs) -> np.ndarray:
    # Reset states if we have completed a full timeseries (+1 for initial conditions)
    if node.current_timestep == node.timesteps:
        node.reset_states()
        node.current_timestep = 0

    # Update the parameters to add the input signal
    node.hypers.update({'input': x})

    # Solve the delayed differential equations and extract the final row as the state
    state = ddeint(dde_system, node._history, node.hypers['time'], args=(node.hypers,))[-1]

    # update the history of states
    node._update_history(state)

    # increment timestep
    node.current_timestep += 1

    # return state as results from single timestep run
    return state


def initialize_oscillator(node: Node, x=None, y=None, initial_values=None, *args, **kwargs):
    if node.input_dim is not None:
        dim = node.input_dim
    else:
        # infer data dimensions
        dim = x.shape[1] if x is not None else 1

    # set input dimensions
    node.set_input_dim(dim)

    # set output dimension to be the 4 dde system variables
    node.set_output_dim(len(node.hypers['initial_conditions']))

    # Set the node's initial values
    if initial_values:
        if len(initial_values) == 4 and all(isinstance(val, int) for val in initial_values):
            node.initial_values = initial_values
        else:
            raise RuntimeError("Initial values must be an array of integers of length 4.")
    else:
        node.initial_values = DEFAULT_HYPERS['initial_conditions']

class Oscillator(Node):
    def __init__(
        self,
        timesteps,
        input_dim=None,
        initial_values=None,
        name=None,
        **kwargs,
    ):
        super(Oscillator, self).__init__(
            hypers=DEFAULT_HYPERS,
            params={},
            forward=forward_oscillator,
            initializer=partial(initialize_oscillator, initial_values=initial_values),
            input_dim=input_dim,
            name=name,
            **kwargs,
        )

        # initialize node parameters
        self.timesteps = timesteps
        self.max_states = math.ceil(self.hypers['delay'])
        self.current_timestep = 0

        # initialize node's states
        self.reset_states()

    def reset_states(self):
        self.states = np.array(self.hypers['initial_conditions']).reshape(1, -1)

    def _update_history(self, state):
        self.states = np.vstack((self.states[-(self.max_states - 1):], state))

    def _history(self, t):
        if abs(t) > self.states.shape[0]:
            return self.states[0]
        return interpolate_history(t, self.states)

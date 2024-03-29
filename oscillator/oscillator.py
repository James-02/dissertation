# Author: James Newsome on 26/02/2024 <james.newsome02@gmail.com>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial

import numpy as np

from .dde import ddeint, dde_system

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
    'coupling': 5e-4,
    'time': np.linspace(0, 1, 2),
    'initial_conditions': [0, 100, 0, 0]
}

def forward(node: Node, x: np.ndarray, **kwargs) -> np.ndarray:
    # Reset states if we have completed a full timeseries (+1 for initial conditions)
    if len(node.states) - 1 == node.timesteps:
        node.reset_states()

    input_val = x[0][0]  # first feature (ECG only has one feature)

    # Update the parameters to add the input signal
    node.hypers.update({'input': input_val})

    # Solve the delayed differential equations to get the derivatives of the system variables
    results = ddeint(dde_system, node.history, node.hypers['time'], node.hypers['delay'], args=(node.hypers,))

    # update history with results
    node.states = np.vstack((node.states, results[-1]))

    # return state as results from single timestep run
    return results[-1]


def initialize(node: Node, x=None, y=None, initial_values=None, *args, **kwargs):
    print("Initialised Node: " + node.name)
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
            forward=forward,
            initializer=partial(initialize, initial_values=initial_values),
            input_dim=input_dim,
            name=name,
            **kwargs,
        )

        # initialise node's states
        self.timesteps = timesteps
        self.reset_states()
    
    def reset_states(self):
        self.states = np.array(self.hypers['initial_conditions']).reshape(1, -1)
    
    def history(self, t):
        num_timesteps, num_features = self.states.shape

        if abs(t) > num_timesteps:
            return self.states[0]

        indices = np.arange(-num_timesteps + 1, 1)
        return np.array([np.interp(t, indices, self.states[:, i]) for i in range(num_features)])

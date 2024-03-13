# Author: James Newsome on 26/02/2024 <james.newsome02@gmail.com>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial

import numpy as np

from .dde import ddeint, dde_system
from .utils import compute_period, compute_phase, prof_cos

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
    'n': 1, 
    'period': 67,
    'coupling': 0.007,
    'time': np.linspace(0, 240, 1000),
    'initial_conditions': [0, 100, 0, 0]
}

def forward(node: Node, x: np.ndarray, **kwargs) -> np.ndarray:
    # Normalise data to 0-180 degree phases and use as input phase
    input_phase = x[0][0]  # first feature (ECG only has one feature)

    # Save original coupling parameter for later use (will be overwritten)
    coupling = node.hypers['coupling']

    # If period is not set, calculate it with 0 coupling
    if node.hypers['period'] == 0:
        node.hypers.update({'period': 1, 'phase': 0, 'coupling': 0})
        # Initial run with no coupling to compute the period
        results = ddeint(dde_system, lambda t: node.initial_values, node.hypers['time'], node.hypers['delay'], args=(node.hypers,))
        A_signal = [row[0] for row in results]

        # Calculate the period
        node.hypers['period'] = compute_period(A_signal, np.mean(np.diff(node.hypers['time'])))

    # Update the parameters to add coupling and input phase
    node.hypers.update({'phase': input_phase, 'coupling': coupling})

    # Generate a reference signal at phase 0
    reference_signal = prof_cos(node.hypers['time'], node.hypers['period'], 0)

    # Solve the delayed differential equations to get the derivatives of the system variables
    results = ddeint(dde_system, lambda t: node.initial_values, node.hypers['time'], node.hypers['delay'], args=(node.hypers,))
    A_signal = [row[0] for row in results]

    # Calculate the phase between the reference signal and the A signal
    phase = compute_phase(A_signal, reference_signal, node.hypers['period'], np.mean(np.diff(node.hypers['time'])))
    # print("Output Phase: " + str(phase))

    # Update state with phase
    return np.array([phase])


def initialize(node: Node, x=None, y=None, initial_values=None, *args, **kwargs):
    print("Node: " + node.name + " initialised")
    if node.input_dim is not None:
        dim = node.input_dim
    else:
        # infer data dimensions
        dim = x.shape[1] if x is not None else 1

    # set input dimensions
    node.set_input_dim(dim)
    node.set_output_dim(dim)

    # Set the node's initial values
    if initial_values:
        if len(initial_values) == 4 and all(isinstance(val, int) for val in initial_values):
            node.initial_values = initial_values
        else:
            raise RuntimeError("Initial values must be an array of integers of length 4.")
    else:
        node.initial_values = DEFAULT_HYPERS['initial_conditions']

    print("Num. Input Features: " + str(node.input_dim))
    print("Initial Values: " + str(node.initial_values))
    print("\n")

class Oscillator(Node):
    """
    Genetic oscillator node defined by a delayed differential equation.

    :param initial_values: array of length 4, defaults to [0, 100, 0, 0].
        Initial conditions for the dde system's variables
    :param input_dim: int, optional
        Input dimension. Can be inferred at first call.
    :param dtype: Numpy dtype, defaults to `None`.
        Numerical type for node parameters.
    :param **kwargs: Additional keyword arguments to pass to the parent class.

    Attributes
    ----------
    hypers : dict
        Default immutable hyperparameters for the oscillator.
    params : dict
        Additional mutable parameters for the oscillator.
    forward : function
        Function to propagate node state using input.
    initializer : function
        Function to initialize the node and it's parameters.
    input_dim : int
        Input dimension of the data.
    output_dim : int
        Output dimension of the data.
    """

    def __init__(
        self,
        input_dim=None,
        initial_values=None,
        dtype=None,
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

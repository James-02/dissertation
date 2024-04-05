# Author: James Newsome on 26/02/2024 <james.newsome02@gmail.com>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from functools import partial
from typing import Optional, Union, Dict, Callable, Sequence

from reservoirpy.mat_gen import bernoulli, normal
from reservoirpy.node import Node
from reservoirpy.type import Weights
from reservoirpy.utils.random import noise, rand_generator
from reservoirpy.nodes.reservoirs.base import initialize, initialize_feedback
from reservoirpy.utils.validation import is_array

from .utils import weight_input, weight_previous_state
from .node import Oscillator

import numpy as np

class OscillatorReservoir(Node):
    def __init__(
        self,
        units: int = None,
        timesteps: int = None,
        delay: float = 10,
        initial_values: list = [0, 100, 0, 0],
        coupling: float = 1e-3,
        sr: Optional[float] = None,
        input_bias: bool = True,
        noise_rc: float = 0.1,
        noise_in: float = 0.1,
        noise_fb: float = 0.1,
        noise_type: str = "normal",
        noise_kwargs: Dict = None,
        rc_scaling: float = 1e-6,
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        fb_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        fb_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = normal,
        Wfb: Union[Weights, Callable] = bernoulli,
        bias: Union[Weights, Callable] = bernoulli,
        input_dim: Optional[int] = None,
        feedback_dim: Optional[int] = None,
        seed=None,
        **kwargs,
    ):
        """
        Initialize an oscillator reservoir.

        Args:
            units (int, optional): Number of oscillator units in the reservoir. Defaults to None.
            timesteps (int, optional): Number of timesteps in the reservoir. Defaults to None.
            delay (float, optional): Delay parameter of the oscillator nodes. Defaults to 10.
            initial_values (list, optional): Initial states for each oscillator node. Defaults to [0, 100, 0, 0].
            coupling (float, optional): Coupling strength of the oscillator nodes. Defaults to 1e-3.
            sr (Optional[float], optional): Spectral radius parameter for reservoir weights initialization. Defaults to None.
            input_bias (bool, optional): Flag indicating whether to include input bias. Defaults to True.
            noise_rc (float, optional): RC noise parameter for reservoir initialization. Defaults to 0.3.
            noise_in (float, optional): Input noise parameter for reservoir initialization. Defaults to 0.3.
            noise_fb (float, optional): Feedback noise parameter for reservoir initialization. Defaults to 0.3.
            noise_type (str, optional): Type of noise to be applied. Defaults to "normal".
            noise_kwargs (Dict, optional): Additional keyword arguments for noise generation. Defaults to None.
            input_scaling (Union[float, Sequence], optional): Scaling factor for input weights. Defaults to 1.0.
            bias_scaling (float, optional): Scaling factor for bias weights. Defaults to 1.0.
            fb_scaling (Union[float, Sequence], optional): Scaling factor for feedback weights. Defaults to 1.0.
            input_connectivity (float, optional): Connectivity parameter for input weights. Defaults to 0.5.
            rc_connectivity (float, optional): Connectivity parameter for reservoir weights. Defaults to 0.1.
            fb_connectivity (float, optional): Connectivity parameter for feedback weights. Defaults to 0.1.
            Win (Union[Weights, Callable], optional): Initializer function for input weights. Defaults to normal.
            W (Union[Weights, Callable], optional): Initializer function for reservoir weights. Defaults to normal.
            Wfb (Union[Weights, Callable], optional): Initializer function for feedback weights. Defaults to bernoulli.
            bias (Union[Weights, Callable], optional): Initializer function for bias weights. Defaults to bernoulli.
            input_dim (Optional[int], optional): Dimensionality of the input data. Defaults to None.
            feedback_dim (Optional[int], optional): Dimensionality of the feedback data. Defaults to None.
            seed ([type], optional): Seed for random number generation. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If 'units' parameter is None and 'W' parameter is not a matrix.
        """
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not "
                "a matrix."
            )

        rng = rand_generator(seed)

        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

        super(OscillatorReservoir, self).__init__(
            fb_initializer=partial(
                initialize_feedback,
                Wfb_init=Wfb,
                fb_scaling=fb_scaling,
                fb_connectivity=fb_connectivity,
                seed=seed,
            ),
            params={
                "W": None,
                "Win": None,
                "Wfb": None,
                "bias": None,
                "internal_state": None,
            },
            hypers={
                "timesteps": timesteps,
                "coupling": coupling,
                "delay": delay,
                "initial_values": initial_values,
                "sr": sr,
                "rc_scaling": rc_scaling,
                "input_scaling": input_scaling,
                "bias_scaling": bias_scaling,
                "fb_scaling": fb_scaling,
                "rc_connectivity": rc_connectivity,
                "input_connectivity": input_connectivity,
                "fb_connectivity": fb_connectivity,
                "noise_in": noise_in,
                "noise_rc": noise_rc,
                "noise_out": noise_fb,
                "noise_type": noise_type,
                "units": units,
                "noise_generator": partial(noise, rng=rng, **noise_kwargs),
            },
            forward=forward_reservoir,
            initializer=partial(
                initialize,
                sr=sr,
                rc_scaling=rc_scaling,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                W_init=W,
                Win_init=Win,
                bias_init=bias,
                input_bias=input_bias,
                seed=seed,
            ),
            output_dim=units,
            feedback_dim=feedback_dim,
            input_dim=input_dim,
            **kwargs,
        )

        self.nodes = initialize_nodes(self)
    
def initialize_nodes(reservoir: Node):
    """
    Initialize oscillator nodes for the reservoir.

    Args:
        reservoir (Node): The reservoir node.

    Returns:
        List: List of initialized oscillator nodes.
    """
    return [Oscillator(reservoir.timesteps, reservoir.delay, reservoir.initial_values) for _ in range(reservoir.units)]

def forward_reservoir(reservoir: Node, x: np.ndarray) -> np.ndarray:
    """
    Compute the next state's of each oscillator node based on the previous state and the input with weighting and noise.

    Args:
        reservoir (Node): The reservoir node.
        x (np.ndarray): Input data.

    Returns:
        np.ndarray: States of each node for the timestep.
    """
    states = np.zeros((len(reservoir.nodes), x.shape[0]))

    # Calculate pre_state
    pre_state = weight_previous_state(reservoir)

    # Calculate weighted input
    weighted_input = weight_input(reservoir, x)

    for i, node in enumerate(reservoir.nodes):
        coupled_input = pre_state[i, 0] + weighted_input[i, 0]

        # Run forward function with weighted input
        state = node.forward(coupled_input)

        # Update the reservoir state with the state of gene I for this timestep
        states[i] = state.flatten()[1]

    return states.T
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

from .utils import weight_input, initialize_nodes, weight_coupling, compute_weight_matrix
from .node import forward_oscillator

import numpy as np

class OscillatorReservoir(Node):
    def __init__(
        self,
        units: int = None,
        timesteps: int = None,
        coupling: float = 1e-4,
        sr: Optional[float] = None,
        input_bias: bool = True,
        noise_rc: float = 0.3,
        noise_in: float = 0.3,
        noise_fb: float = 0.3,
        noise_type: str = "normal",
        noise_kwargs: Dict = None,
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        fb_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.5,
        rc_connectivity: float = 0.1,
        fb_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = normal,
        W: Union[Weights, Callable] = normal,
        Wfb: Union[Weights, Callable] = bernoulli,
        bias: Union[Weights, Callable] = bernoulli,
        input_dim: Optional[int] = None,
        feedback_dim: Optional[int] = None,
        seed=None,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not "
                "a matrix."
            )

        rng = rand_generator(seed)
        weight_matrix = compute_weight_matrix(units, rc_connectivity)

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
                "coupling": coupling,
                "sr": sr,
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
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                W_init=weight_matrix,
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

        self.W = weight_matrix
        self.timesteps = timesteps

        self.nodes = initialize_nodes(self)
        self.coupling_matrix = weight_coupling(self)


def forward_reservoir(reservoir: Node, x: np.ndarray) -> np.ndarray:
    states = np.zeros((len(reservoir.nodes), x.shape[0]))
    weighted_input = weight_input(reservoir, x)

    for i, node in enumerate(reservoir.nodes):
        coupled_input = reservoir.coupling_matrix[i, 0] * np.sum(weighted_input)

        # Run forward function with weighted input
        state_next = forward_oscillator(node, coupled_input)

        # Update the reservoir state for that node at the timestep
        states[i] = state_next.flatten().sum()

    return states.T
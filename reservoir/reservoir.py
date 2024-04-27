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
from reservoirpy.activationsfunc import identity

from .node import Oscillator

import numpy as np

class OscillatorReservoir(Node):
    def __init__(
        self,
        units: int = None,
        timesteps: int = None,
        warmup: int = 40,
        coupling: float = 1e-3,
        sr: Optional[float] = None,
        input_bias: bool = True,
        noise_rc: float = 0.0,
        noise_in: float = 0.0,
        noise_fb: float = 0.0,
        noise_type: str = "normal",
        noise_kwargs: Dict = None,
        rc_scaling: float = 1e-6,
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        fb_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        fb_connectivity: float = 0.1,
        fb_activation: Callable = identity,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = normal,
        Wfb: Union[Weights, Callable] = bernoulli,
        bias: Union[Weights, Callable] = bernoulli,
        input_dim: Optional[int] = None,
        feedback_dim: Optional[int] = None,
        seed=None,
        node_kwargs: Dict = None,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not "
                "a matrix."
            )

        rng = rand_generator(seed)

        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs
        node_kwargs = dict() if node_kwargs is None else node_kwargs

        super(OscillatorReservoir, self).__init__(
            fb_initializer=partial(
                initialize_feedback,
                Wfb_init=Wfb,
                fb_activation=fb_activation,
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
                "warmup": warmup,
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
        self.nodes = _initialize_nodes(self, node_kwargs)

def _initialize_nodes(reservoir: Node, node_kwargs):
    return [Oscillator(reservoir.timesteps, warmup=reservoir.warmup, **node_kwargs) for _ in range(reservoir.units)]

def _compute_input(reservoir, x):
    u = x.reshape(-1, 1)
    r = reservoir.state().T

    # Apply input weights and noise
    noise = reservoir.noise_generator(dist=reservoir.noise_type, shape=u.shape, gain=reservoir.noise_in)
    weighted_input = (reservoir.Win @ (u + noise) + reservoir.bias) * reservoir.coupling

    noise = reservoir.noise_generator(dist=reservoir.noise_type, shape=r.shape, gain=reservoir.noise_rc)
    recurrent_state = (reservoir.W @ (r + noise)) * reservoir.rc_scaling

    pre_state = weighted_input + recurrent_state

    if reservoir.has_feedback:
        y = reservoir.feedback().reshape(-1, 1)
        y = reservoir.fb_activation(y) + reservoir.noise_generator(dist=reservoir.noise_type, shape=y.shape, gain=reservoir.noise_fb)

        pre_state += reservoir.Wfb @ y

    return np.array(pre_state)

def forward_reservoir(reservoir: Node, x: np.ndarray) -> np.ndarray:
    states = np.zeros((len(reservoir.nodes), x.shape[0]))

    # Calculate pre_state
    pre_state = _compute_input(reservoir, x)

    for i, node in enumerate(reservoir.nodes):
        # Run forward function with weighted input
        state = node.forward(pre_state[i, 0])

        # Update the reservoir state with the state of gene I for this timestep
        states[i] = state.flatten()[1]

    return states.T

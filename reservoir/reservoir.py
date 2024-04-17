# Author: James Newsome on 26/02/2024 <james.newsome02@gmail.com>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from functools import partial
from typing import Optional, Union, Dict, Callable, Sequence

from reservoirpy.mat_gen import bernoulli, normal
from reservoirpy.node import Node
from reservoirpy.type import Weights
from reservoirpy.utils.random import noise, rand_generator
from reservoirpy.nodes.reservoirs.base import initialize_feedback
from reservoirpy.utils.validation import is_array
from reservoirpy.activationsfunc import identity
from reservoirpy.mat_gen import zeros


from .node import Oscillator

import numpy as np

class OscillatorReservoir(Node):
    def __init__(
        self,
        units: int = None,
        timesteps: int = None,
        delay: float = 7,
        initial_values: list = [300, 300, 0, 0],
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
        fb_activation: Callable = identity,
        Win: Union[Weights, Callable] = bernoulli,
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

        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

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
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                rc_scaling=rc_scaling,
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
        self.nodes = _initialize_nodes(self)

def initialize(
    reservoir,
    x=None,
    y=None,
    sr=None,
    input_scaling=None,
    bias_scaling=None,
    rc_scaling=None,
    input_connectivity=None,
    rc_connectivity=None,
    W_init=None,
    Win_init=None,
    bias_init=None,
    input_bias=None,
    seed=None,
):
    if x is not None:
        reservoir.set_input_dim(x.shape[1])

        dtype = reservoir.dtype
        dtype_msg = (
            "Data type {} not understood in {}. {} should be an array or a "
            "callable returning an array."
        )

        if is_array(W_init):
            W = W_init
            if W.shape[0] != W.shape[1]:
                raise ValueError(
                    "Dimension mismatch inside W: "
                    f"W is {W.shape} but should be "
                    f"a square matrix."
                )

            if W.shape[0] != reservoir.output_dim:
                reservoir._output_dim = W.shape[0]
                reservoir.hypers["units"] = W.shape[0]

        elif callable(W_init):
            W = W_init(
                reservoir.output_dim,
                reservoir.output_dim,
                sr=sr,
                input_scaling=rc_scaling,
                connectivity=rc_connectivity,
                dtype=dtype,
                seed=seed,
            )
        else:
            raise ValueError(dtype_msg.format(str(type(W_init)), reservoir.name, "W"))

        reservoir.set_param("units", W.shape[0])
        reservoir.set_param("W", W.astype(dtype))

        out_dim = reservoir.output_dim

        Win_has_bias = False
        if is_array(Win_init):
            Win = Win_init

            msg = (
                f"Dimension mismatch in {reservoir.name}: Win input dimension is "
                f"{Win.shape[1]} but input dimension is {x.shape[1]}."
            )

            # is bias vector inside Win ?
            if Win.shape[1] == x.shape[1] + 1:
                if input_bias:
                    Win_has_bias = True
                else:
                    bias_msg = (
                        " It seems Win has a bias column, but 'input_bias' is False."
                    )
                    raise ValueError(msg + bias_msg)
            elif Win.shape[1] != x.shape[1]:
                raise ValueError(msg)

            if Win.shape[0] != out_dim:
                raise ValueError(
                    f"Dimension mismatch in {reservoir.name}: Win internal dimension "
                    f"is {Win.shape[0]} but reservoir dimension is {out_dim}"
                )

        elif callable(Win_init):
            Win = Win_init(
                reservoir.output_dim,
                x.shape[1],
                input_scaling=input_scaling,
                connectivity=input_connectivity,
                dtype=dtype,
                seed=seed,
            )
        else:
            raise ValueError(
                dtype_msg.format(str(type(Win_init)), reservoir.name, "Win")
            )

        if input_bias:
            if not Win_has_bias:
                if callable(bias_init):
                    bias = bias_init(
                        reservoir.output_dim,
                        1,
                        input_scaling=bias_scaling,
                        connectivity=input_connectivity,
                        dtype=dtype,
                        seed=seed,
                    )
                elif is_array(bias_init):
                    bias = bias_init
                    if bias.shape[0] != reservoir.output_dim or (
                        bias.ndim > 1 and bias.shape[1] != 1
                    ):
                        raise ValueError(
                            f"Dimension mismatch in {reservoir.name}: bias shape is "
                            f"{bias.shape} but should be {(reservoir.output_dim, 1)}"
                        )
                else:
                    raise ValueError(
                        dtype_msg.format(str(type(bias_init)), reservoir.name, "bias")
                    )
            else:
                bias = Win[:, :1]
                Win = Win[:, 1:]
        else:
            bias = zeros(reservoir.output_dim, 1, dtype=dtype)

        reservoir.set_param("Win", Win.astype(dtype))
        reservoir.set_param("bias", bias.astype(dtype))
        reservoir.set_param("internal_state", reservoir.zero_state())

def _initialize_nodes(reservoir: Node):
    return [Oscillator(reservoir.timesteps, reservoir.delay, reservoir.initial_values) for _ in range(reservoir.units)]

def _compute_input(reservoir, x):
    u = x.reshape(-1, 1)
    r = reservoir.state().T

    # Apply input weights and noise
    noise = reservoir.noise_generator(dist=reservoir.noise_type, shape=u.shape, gain=reservoir.noise_in)
    weighted_input = (reservoir.Win @ (u + noise) + reservoir.bias) * reservoir.coupling

    noise = reservoir.noise_generator(dist=reservoir.noise_type, shape=r.shape, gain=reservoir.noise_rc)
    recurrent_state = (reservoir.W @ (r + noise))

    pre_state = weighted_input + recurrent_state

    if reservoir.has_feedback:
        y = reservoir.feedback().reshape(-1, 1)
        y = reservoir.fb_activation(y) + reservoir.noise_generator(dist=reservoir.noise_type, shape=y.shape, gain=reservoir.noise_fb)

        pre_state += reservoir.Wfb @ y

    return np.array(pre_state)

def forward_reservoir(reservoir: Node, x: np.ndarray) -> np.ndarray:
    states = np.zeros((len(reservoir.nodes), x.shape[0]))

    # Normalize input between 0-1
    x = (x - 0) / (1 - 0)

    # Calculate pre_state
    pre_state = _compute_input(reservoir, x)

    for i, node in enumerate(reservoir.nodes):
        # Run forward function with weighted input
        state = node.forward(pre_state[i, 0])

        # Update the reservoir state with the state of gene I for this timestep
        states[i] = state.flatten()[1]

    return states.T

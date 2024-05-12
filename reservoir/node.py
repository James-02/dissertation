import numpy as np
import math

from .dde import solve_dde, dde_system, interpolate_history

# Default hyperparameters for the oscillator
DEFAULT_PARAMS = {
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
    'time': np.linspace(0, 1)
}

class Oscillator:
    """
    Oscillator node which simulates a system of two coupled genes.

    The genes interact through the quorum sensing molecule AHL, represented by an internal signal Hi, and an external signal, He.
    """
    def __init__(self, timesteps: int, warmup: int = 40, **kwargs):
        """
        Initialize a genetic oscillator node.

        Args:
            timesteps (int): The number of timesteps for which to simulate the oscillator.
            warmup (int, optional): The number of warmup iterations to stabilize the oscillator before using it.
            **kwargs: Additional parameters for the oscillator.
        """
        self.hypers = {**DEFAULT_PARAMS, **kwargs}
        self.timesteps = timesteps
        self.warmup = warmup
        self.warmup_states = []

        if self.hypers['delay'] <= 0:
            raise ValueError("Delay must be specified and > 0")

        self._max_states = math.ceil(self.hypers['delay'])
        self._initialize_initial_conditions()
        self._current_timestep = 0
        self._initial_states = self._warmup_states()
        self._states = self._initial_states

    def _initialize_initial_conditions(self) -> None:
        """
        Initializes the initial values of the A and I genes randomly between the range of [30 - 60].
        """
        self.hypers['initial_conditions'] = np.zeros(4)
        self.hypers['initial_conditions'][0] = np.random.randint(30, 60)
        self.hypers['initial_conditions'][1] = np.random.randint(30, 60)

    def _warmup_states(self) -> np.ndarray:
        """
        Performs warmup iterations with an input of 0 to stabilize the oscillatons before use.

        Returns:
            np.ndarray: Array of warmup states.
        """
        self.hypers['input'] = 0
        self._states = np.array(self.hypers['initial_conditions']).reshape(1, -1)
        for _ in range(self.warmup):
            state = solve_dde(dde_system, self._history, self.hypers['time'], args=(self.hypers,))[:, -1]
            self.warmup_states.append(state)
            self._update_history(state)
        return self._states

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Processes a single timestep of continuous data using a system of delay differential equations and the history of previous timesteps.

        The states produced for each timestep are collated in the node's history, which is interpolated to compute the next state.

        Args:
            x (np.ndarray): The timestep of input.

        Returns:
            np.ndarray: The state of the dde system.
        """
        if self._current_timestep == self.timesteps:
            self._states = self._initial_states
            self._current_timestep = 0

        self.hypers.update({'input': x})
        state = solve_dde(dde_system, self._history, self.hypers['time'], args=(self.hypers,))[:, -1]
        self._update_history(state)
        self._current_timestep += 1
        return state

    def _update_history(self, state: np.ndarray) -> None:
        """
        Updates the history of states, keeping a rolling buffer to the size of the "delay" parameter.

        Args:
            state (np.ndarray): The state of the oscillator.
        """
        if len(self._states) == self._max_states:
            self._states = np.vstack((self._states[1:], state))
        else:
            self._states = np.vstack((self._states, state))

    def _history(self, t: float, idx: int = None) -> np.ndarray:
        """
        Interpolate the node's state at 't' timesteps in the past using the history of states.

        Args:
            t (float): The time index.
            idx (int, optional): The index of the state.

        Returns:
            np.ndarray: The state at the given time index.
        """
        if abs(t) == 0:
            return self._states[-1]

        if abs(t) > self._states.shape[0]:
            if idx is None:
                return self._states[0]
            return self._states[0, idx]

        return interpolate_history(t, self._states, idx)

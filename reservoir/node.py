import numpy as np
import math

from .dde import solve_dde, dde_system, interpolate_history

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
    'time': np.linspace(0, 1),
    'initial_conditions': [0, 100, 0, 0]
}

class Oscillator():
    """
    A class representing an oscillator system.

    Attributes:
        timesteps (int): Number of timesteps to run the oscillator.
        hypers (dict): Dictionary containing hyperparameters of the oscillator.
        max_states (int): Maximum number of states to keep in history.
        current_timestep (int): Current timestep of the oscillator.
    """

    def __init__(self, timesteps: int, hypers: dict = DEFAULT_HYPERS):
        """
        Initialize an oscillator node.

        Args:
            timesteps (int): Number of timesteps to run the oscillator.
            hypers (dict, optional): Dictionary containing hyperparameters of the oscillator. 
                Defaults to DEFAULT_HYPERS.
        """
        self.timesteps = timesteps
        self.hypers = hypers

        self._max_states = math.ceil(self.hypers['delay'])
        self._current_timestep = 0
        self._states = self._reset_states()

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run the oscillator for one timestep with input x.

        Args:
            x (np.ndarray): Input signal.

        Returns:
            np.ndarray: Current state of the oscillator (A, I, Hi, He).
        """
        # Reset states if we have completed a full timeseries
        if self._current_timestep == self.timesteps:
            self._states = self._reset_states()
            self._current_timestep = 0

        # Update the parameters to add the input signal
        self.hypers.update({'input': x})

        # Solve the delayed differential equations and extract the final row as the state
        state = solve_dde(dde_system, self._history, self.hypers['time'], args=(self.hypers,))[-1]

        # update the history of states
        self._update_history(state)

        # increment timestep
        self._current_timestep += 1

        # return system states at this timestep
        return state

    def _reset_states(self):
        """Reset the states of the oscillator to the initial conditions."""
        return np.array(self.hypers['initial_conditions']).reshape(1, -1)

    def _update_history(self, state):
        """
        Append a new state to the states history vector.

        Args:
            state (np.ndarray): New state to be added to the history.
        """
        self._states = np.vstack((self._states[-(self._max_states - 1):], state))

    def _history(self, t):
        """
        Return the interpolated history of states at time t.

        Args:
            t (float): Time at which the history is requested.

        Returns:
            np.ndarray: Interpolated history of states at time t.
        """
        if abs(t) > self._states.shape[0]:
            return self._states[0]
        return interpolate_history(t, self._states)
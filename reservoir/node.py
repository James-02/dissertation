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

class Oscillator():
    def __init__(self, timesteps: int, warmup: int = 40, **kwargs):
        self.hypers = {**DEFAULT_PARAMS, **kwargs}
        self.timesteps = timesteps
        self.warmup = warmup
        self._max_states = math.ceil(self.hypers['delay'])

        if self.hypers['delay'] <= 0:
            raise ValueError("Delay must be specified and > 0")
        
        # initialize random initial conditions
        self.hypers['initial_conditions'] = np.zeros(4)

        # Randomise initial state of I gene
        self.hypers['initial_conditions'][1] = np.random.randint(50, 100)

        # control history of states
        self._current_timestep = 0
        self._max_states = math.ceil(self.hypers['delay'])

        # warmup genes to start oscillating without input
        self._initial_states = self._warmup_states()
        self._states = self._initial_states

    def _warmup_states(self):
        self.hypers['input'] = 0
        self._states = np.array(self.hypers['initial_conditions']).reshape(1, -1)
        for _ in range(self.warmup):
            state = solve_dde(dde_system, self._history, self.hypers['time'], args=(self.hypers,))[:, -1]
            self._update_history(state)
        return self._states

    def forward(self, x: np.ndarray) -> np.ndarray:       
        # Reset states if we have completed a full timeseries
        if self._current_timestep == self.timesteps:
            self._states = self._initial_states
            self._current_timestep = 0

        # Update the parameters to add the input signal
        self.hypers.update({'input': x})

        # Solve the delayed differential equations and extract the final row as the state
        state = solve_dde(dde_system, self._history, self.hypers['time'], args=(self.hypers,))[:, -1]
        # update the history of states
        self._update_history(state)

        # increment timestep
        self._current_timestep += 1

        # return system states at this timestep
        return state

    def _update_history(self, state: np.ndarray):
        if len(self._states) == self._max_states:
            # If the size limit is reached, roll buffer to remove oldest state and append newest
            self._states = np.vstack((self._states[1:], state))
        else:
            # Otherwise, simply append the new state
            self._states = np.vstack((self._states, state))

    def _history(self, t: float, idx: int = None):
        # if delay is 0, return current state
        if abs(t) == 0:
            return self._states[-1]

        # if delay is greater than existing history, return oldest state (initial conditions)
        if abs(t) > self._states.shape[0]:
            if idx is None:
                return self._states[0]
            return self._states[0, idx]
        
        # interpolate state at historical timepoint from history of states
        return interpolate_history(t, self._states, idx)

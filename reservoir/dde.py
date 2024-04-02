import numpy as np

from scipy.integrate import solve_ivp
from typing import Callable, Tuple

def solve_dde(func: Callable, history: Callable, t: np.ndarray, args: Tuple = ()) -> np.ndarray:
    """Wrapper function which solves Delay Differential Equations using scipy's solve_ivp.

    Args:
        func (callable): Function representing the system of delay differential equations.
        history (callable): Function providing history values for the system.
        t (np.ndarray): Array of time points for integration.
        args (tuple, optional): Additional positional arguments to pass to `func`.

    Returns:
        np.ndarray: Solution of the delay differential equations at the specified time points.
    """
    return solve_ivp(lambda t, Y, args: func(Y, t, history, args), [t[0], t[-1]], history(t[0]), t_eval=t, args=args).y.T

def dde_system(Y: np.ndarray, t: float, history: Callable, params: dict) -> np.ndarray:
    """
    Define the delayed differential equations (DDE) system.

    This function represents the system of delay differential equations,
    the equations represent two coupled genes (A, I) and their behaviour during expression.
    Hi, He represent the internal and external signals of the cell during expression.

    The system uses a delayed differential equation solver to compute the derivatives of each
    variable in the system at a given time point `t`, based on the current state `Y`,
    the past state provided by the `history` function, and the parameters `params`.

    Args:
        Y (np.ndarray): Current state of the system at time `t`.
        t (float): Current time point.
        history (callable): Function to interpolate the historical values of the system's variables.
        params (dict): Dictionary containing parameters required for the system dynamics.

    Returns:
        List: List containing the derivatives of each variable in the system
        at the given time point `t`.
    """
    # dde system variables representing two coupled genes (A, I) and the internal/external signals of the cell (Hi, He)
    A, I, Hi, He = Y

    # Value of Hi at 'delay' timesteps in the past
    Hlag = history(t - params['delay'])[2]

    P = (params['del_'] + params['alpha'] * Hlag**2) / (1 + params['k1'] * Hlag**2)

    # external input signal
    Hetot = He + params['input']

    # equations to calculate the state of the genetic oscillator for the timepoint t
    dAdt = params['CA'] * (1 - (params['d']/params['d0'])**4) * P - params['gammaA'] * A / (1 + params['f'] * (A + I))
    dIdt = params['CI'] * (1 - (params['d']/params['d0'])**4) * P - params['gammaI'] * I / (1 + params['f'] * (A + I))
    dHidt = params['b'] * I / (1 + params['k'] * I) - params['gammaH'] * A * Hi / (1 + params['g'] * A) + params['D'] * (Hetot - Hi)
    dHedt = -params['d'] / (1 - params['d']) * params['D'] * (He - Hi) - params['mu'] * He
    
    return [dAdt, dIdt, dHidt, dHedt]

def interpolate_history(t: float, states: np.ndarray) -> np.ndarray:
    """Interpolates history values at given time points.

    Args:
        t (float): Time point to interpolate history at.
        states (np.ndarray): Array of historical states.

    Returns:
        List: Interpolated history values.
    """
    indices = np.arange(-states.shape[0] + 1, 1)
    return [np.interp(t, indices, states[:, i]) for i in range(states.shape[1])]

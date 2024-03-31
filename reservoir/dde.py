import numpy as np
import scipy

def ddeint(func, y0, t, tau, args=(), y0_args=(), n_time_points_per_step=None):
    """
    Integrate a system of delay differential equations.

    Args:
        func (callable): Function representing the system of delay differential equations.
        y0 (callable): Function providing initial conditions for the system.
        t (array-like): Array of time points for integration.
        tau (float or array-like): Delay time(s).
        args (tuple, optional): Additional arguments to pass to `func`.
        y0_args (tuple, optional): Additional arguments to pass to `y0`.
        n_time_points_per_step (int, optional): Number of time points per integration step.

    Returns:
        array-like: Solution of the delay differential equations at the specified time points.
    """

    # Convert tau to a numpy array for consistent handling.
    tau = np.atleast_1d(tau)

    # Ensure that all delay times are positive.
    if (tau <= 0).any():
        raise RuntimeError("All tau's must be greater than zero.")

    # Determine the shortest and longest delay times.
    tau_short = np.min(tau)
    tau_long = np.max(tau)

    # If the number of time points per step isn't specified,
    # calculate it based on the total time range and the longest delay.
    if n_time_points_per_step is None:
        n_time_points_per_step = max(int(1 + len(t) / (t.max() - t.min()) * tau_long), 20)

    t0 = t[0]

    # Define the past function for the first step.
    y_past = lambda time_point: y0(time_point, *y0_args)

    # Integrate the system over the first step.
    t_step = np.linspace(t0, t0 + tau_short, n_time_points_per_step)
    y = scipy.integrate.odeint(func, y_past(t0), t_step, args=(y_past,) + args)

    # Store the solution from the first step.
    y_dense = y.copy()
    t_dense = t_step.copy()

    # Get the dimension of the system (number of equations).
    n = y.shape[1]

    # Integrate the system over subsequent steps.
    j = 1
    while t_step[-1] < t[-1]:
        # Determine the starting time for interpolation.
        t_start = max(t0, t_step[-1] - tau_long)
        i = np.searchsorted(t_dense, t_start, side="left")
        t_interp = t_dense[i:]
        y_interp = y_dense[i:, :]

        # Create B-spline representations of the solution for interpolation.
        tck = [scipy.interpolate.splrep(t_interp, y_interp[:, i]) for i in range(n)]

        # Define the past function for this step.
        y_past = (
            lambda time_point: np.array([scipy.interpolate.splev(time_point, tck[i]) for i in range(n)])
            if time_point > t0
            else y0(time_point, *y0_args)
        )

        # Integrate the system over this step.
        t_step = np.linspace(t0 + j * tau_short, t0 + (j + 1) * tau_short, n_time_points_per_step)
        y = scipy.integrate.odeint(func, y[-1, :], t_step, args=(y_past,) + args)

        # Append the solution from this step to the stored solution.
        y_dense = np.append(y_dense, y[1:, :], axis=0)
        t_dense = np.append(t_dense, t_step[1:])

        j += 1

    # Interpolate the dense solution to get values at the desired time points.
    y_return = np.empty((len(t), n))
    for i in range(n):
        tck = scipy.interpolate.splrep(t_dense, y_dense[:, i])
        y_return[:, i] = scipy.interpolate.splev(t, tck)

    return y_return

def dde_system(Y, t, Y_past, params):
    """
    Define the delayed differential equations (DDE) system.
    """
    A, I, Hi, He = Y
    Hlag = Y_past(t - params['delay'])[2]  # Delayed value of Hi
    P = (params['del_'] + params['alpha'] * Hlag**2) / (1 + params['k1'] * Hlag**2)

    # external input signal
    Hetot = He + params['input']

    dAdt = params['CA'] * (1 - (params['d']/params['d0'])**4) * P - params['gammaA'] * A / (1 + params['f'] * (A + I))
    dIdt = params['CI'] * (1 - (params['d']/params['d0'])**4) * P - params['gammaI'] * I / (1 + params['f'] * (A + I))
    dHidt = params['b'] * I / (1 + params['k'] * I) - params['gammaH'] * A * Hi / (1 + params['g'] * A) + params['D'] * (Hetot - Hi)
    dHedt = -params['d'] / (1 - params['d']) * params['D'] * (He - Hi) - params['mu'] * He
    
    return [dAdt, dIdt, dHidt, dHedt]

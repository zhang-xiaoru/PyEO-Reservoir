from numba import jit
import numpy as np



@jit
def jit_interp(
    x: list[float | int], xp: float | list[float], yp: float | list[float]
) -> float:
    """JIT complied 1D interpolation"""
    return np.interp(x, xp, yp)


@jit
def jit_sin(x: list[float | int] | float) -> list[float] | float:
    """JIT complied numpy sine function"""
    return np.sin(x)


@jit
def jit_square(x: list[float] | float) -> list[float] | float:
    """JIT complied square function"""
    return x * x


@jit
def jit_jfunc(j_points: list[float], t: float, j_step_len: float) -> float:
    """JIT complied input step function.
    Input:
        j_points: list of data points used for step function
        t: time to get function value
        j_step_len: time length for each step of j function
    Return:
        value of step function at time t
    """
    return j_points[int(t // j_step_len)]


@jit
def jit_historyfunc(history_points: list[float], t: float, h: float) -> float:
    """JIT complied interpolated history function.
    Input:
        history_points: list contained history of the solution.
        t: time to get function value
        h: time interval between points
    Return:
        linear interpolated value as value of history function."""
    # create time interval which include the time evaluated at
    time_interval = [h * int(t // h), h * (int(t // h) + 1)]
    # find the historical point for time_interval
    history_interval = history_points[int(t // h) : int(t // h) + 2]
    return jit_interp(t, time_interval, history_interval)


@jit
def jit_dfdt(
    t: float,
    y: float,
    j_points: list[float],
    history_points: list[float],
    j_step_len: float,
    h: float,
    t_local: float,
    eta: float,
    gamma: float,
    phi: float,
) -> float:
    """
    JIT complied derivative function for Ikeda nonlinearity.
    Input:
        t: time where the derivative is evaluated
        y: y value where the derivative is evaluated
        j_points: list of input data used to constructed input step function
        history_points: history of past for interpolate delay function
        j_step_len: length of step for step input function
        h: integration step length
        t_local: local time scale
        eta: strength factor for nonlinear term
        gamma: strength factor for input
        phi: constant phase (in unit of pi)
    Return:
        derivative evaluated at (t, y) for Ikeda nonlinearity
    """
    # calculate the Ikeda nonlinear term
    nonlinear_term = jit_square(
        jit_sin(
            jit_historyfunc(history_points, t, h)
            + gamma * jit_jfunc(j_points, t, j_step_len)
            + np.pi * phi
        )
    )
    return -1 / t_local * y + eta * nonlinear_term


@jit
def dde_solver(
    j_points: list[float],
    init_points: list[float],
    delay_node_size: int,
    delay: float,
    j_step_len: float,
    n_int: int,
    t_local: float,
    eta: float,
    gamma: float,
    phi: float,
    return_on_nodes: bool = True,
):
    """JIT complied dde solver for Ikeda delay oscillator with input.
    Input:
        j_points: list of input data points
        init_points: discrete point for initial function of dde
        delay_node_size: number of nodes within one delay
        delay: delay time
        j_step_len: length of step for step input function
        n_int: number of integration step within one delay
        t_local: local time scale
        eta: strength factor for nonlinear term
        gamma: strength factor for input
        phi: constant phase (in unit of pi)
    Return:
        sol: numpy array contains solution
        time_steps: numpy array contains time steps"""

    # get integration step
    h = delay / n_int

    # length of input data points
    j_points_len = len(j_points)
    # initialize solution array
    sol = np.zeros((int(j_points_len / delay_node_size) * n_int + 1))
    time_steps = np.zeros((int(j_points_len / delay_node_size) * n_int + 1))
    # initialize history points (shifted solution with delay)
    history_points = np.append(init_points, sol)
    t = 0
    time_steps[0] = t
    # rk3 update for solving
    for i in range(n_int * int(j_points_len / delay_node_size)):
        y = sol[i]
        k1 = jit_dfdt(
            t, y, j_points, history_points, j_step_len, h, t_local, eta, gamma, phi
        )
        k2 = jit_dfdt(
            t + 1 / 2 * h,
            y + 1 / 2 * k1 * h,
            j_points,
            history_points,
            j_step_len,
            h,
            t_local,
            eta,
            gamma,
            phi,
        )
        k3 = jit_dfdt(
            t + 1 / 4 * h,
            y + 1 / 2 * k2 * h,
            j_points,
            history_points,
            j_step_len,
            h,
            t_local,
            eta,
            gamma,
            phi,
        )
        y_next = y + h * (2 / 9 * k1 + 1 / 3 * k2 + 4 / 9 * k3)
        t += h
        sol[i + 1] = y_next
        time_steps[i + 1] = t
        history_points[len(init_points) + i + 1] = y_next
    if return_on_nodes:
        time_steps_on_nodes = np.arange(
            0, (int(j_points_len / delay_node_size) + 1e-6) * delay, delay/delay_node_size
        )
        sol_on_nodes = jit_interp(time_steps_on_nodes, time_steps, sol)
        return sol_on_nodes, time_steps_on_nodes
    else:
        return sol, time_steps
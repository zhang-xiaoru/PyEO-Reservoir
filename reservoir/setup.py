from setup import PhysicsSettings, SolverSettings
from itertools import product



class PhysicsSettings:
    def __init__(self, t_local: float, eta: float, gamma: float, phi: float) -> None:
        """Input:
        t_local : local time scale (in unit of ps)
        eta : nonlinear strength factor (in unit of 1e12)
        gamma : input strength factor
        phi : constant phase (in unit of pi)
        """
        self.t_local = t_local
        self.gamma = gamma
        self.eta = eta
        self.phi = phi


class SolverSettings:
    def __init__(
        self, delay_node_size: int, delay: float, j_step_len: float, n_int: int
    ) -> None:
        """Input:
        delay_node_size : number of nodes within one delay.
        j_step_len: length of step for step input function.
        delay: delay time of Ikeda nonlinearity (in unit of ps).
        n_int: number of integration step within one delay.
        """
        self.delay_node_size = delay_node_size
        self.j_step_len = j_step_len
        self.delay = delay
        self.n_int = n_int
        self.int_step = delay / n_int


class CONFIG:
    def __init__(
        self, phy_setting: PhysicsSettings, sol_setting: SolverSettings
    ) -> None:
        """Input:
        phy_setting : physics setting
        sol_setting : dde solver settings"""
        self.phy_setting = phy_setting
        self.sol_setting = sol_setting


def make_setting_list(
    sweep_args: dict[str, list[float]] | None = None,
    base_args: dict[str, float] | None = None,
    setting: str = "phy",
) -> list[PhysicsSettings | SolverSettings]:
    """Create list of PhysicsSettings.
    Input:
    sweep_args: dict that contains params that require sweep
    base_args: dict that contains non-sweeping params
    setting: string that indicate generating physics or solver setting list

    Return:
    list contains PhysicsSettings or SolverSettings for all sweeping combination"""
    # convert to dict if None
    sweep_args = sweep_args or {}
    base_args = base_args or {}

    # extract params names and values from sweep_args
    keys, values = zip(*sweep_args.items()) if sweep_args else ([], [])

    # create all combination list
    combinations = [dict(zip(keys, v)) for v in product(*values)] if keys else [{}]

    # list of physics settings
    if setting == "phy":
        return [PhysicsSettings(**{**base_args, **c}) for c in combinations]
    elif setting == "sol":
        return [SolverSettings(**{**base_args, **c}) for c in combinations]
    else:
        raise ValueError("setting can only be 'phy' or 'sol'.")
    
def masking(u, m, flatten=True):
    # np.random.seed(42)
    # in_size = u.shape[0]
    # m = np.random.rand(delay_node_size, in_size) - 0.5
    # m = 2*(np.random.randint(0, 2, (delay_node_size, in_size))-0.5)
    return (m @ u).T.flatten() if flatten else m @ u


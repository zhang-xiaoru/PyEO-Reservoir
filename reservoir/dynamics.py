from ..dde_solver.jit_solver import dde_solver
import readout
from setup import CONFIG
from numpy.typing import NDArray
import numpy as np
import h5py

def calculate_state(
    init_points: list[float],
    j_points: list[float],
    config: CONFIG, # type: ignore  # noqa: F821
    return_times: bool = False,
    return_on_nodes: bool = True
):
    """main function for calculating the state evolution inside Ikeda delay oscillator.
    Input:
        init_points: discrete point for initial function of dde
        j_points: list of input data points
        config: configuration class entity contains parameters
        return_times: whether to return exact time steps for the integration
    return:
        solution and time steps if return_times is true, otherwise only solution.
    """
    # get solver setting from configuration
    delay_node_size = config.sol_setting.delay_node_size
    j_step_len = config.sol_setting.j_step_len
    n_int = config.sol_setting.n_int
    delay = config.sol_setting.delay

    # get physics setting from configuration
    t_local = config.phy_setting.t_local
    eta = config.phy_setting.eta
    gamma = config.phy_setting.gamma
    phi = config.phy_setting.phi

    # solve ikeda delay equation
    result = dde_solver(
        j_points,
        init_points,
        delay_node_size,
        delay,
        j_step_len,
        n_int,
        t_local,
        eta,
        gamma,
        phi,
        return_on_nodes
    )

    if return_times:
        return result
    else:
        return result[0]
    
def resevoir(
    j_points: NDArray,
    init_points: NDArray,
    seq_len: int | None,
    config: CONFIG,
    sample_on_nodes: bool = True,
    return_type: str | None = None,
) -> NDArray | None:
    resevoir_states = calculate_state(
        init_points, j_points, config, False, sample_on_nodes
    )
    if return_type:
        if return_type == "full":
            node_mat = readout.reshape_node_state(
                resevoir_states, seq_len, config.sol_setting.delay_node_size
            )
            return node_mat
        elif return_type == "last":
            encoding_state = readout.get_encoding_state(
                resevoir_states.reshape(1, -1), config.sol_setting.delay_node_size
            )
            return encoding_state
        else:
            raise ValueError("The allowed type can only be 'full', 'last' or None")
        

def save_resultFix_concated(result: NDArray, config, f: h5py.File) -> None:
    """Save fixed-length reservoir state in to hdf5 file.

    The HDF5 file follows this hierarchical structure:

    ```
    /                           (Root group)
    |--- /gamma_<value>/        (Group for result with gamma = value)
    |    |--- eta_<value>/      (Subgroup for result with eta = value)
    |    |    |--- data         (Dataset: Main data. Concated horizontally for different phis)
    |    |    |--- phis         (Dataset attributes recording phi values for stored data)
    |    |    |--- feature_size (Dataset attributes recording feature_size)
    ```

    Args:
        result (NDArray): NDArray saving the simulation result
        config (_type_): configuration used for simulation
        f (h5py.File): opened hdf5 file
    """
    group_name = f"gamma_{config.phy_setting.gamma:.2f}"
    # get the group if exists, otherwise create the group
    group = f.require_group(group_name)
    # get the subgroup if exists, otherwise create the subgroup
    subgroup_name = f"eta_{config.phy_setting.eta:.2f}"
    subgroup = group.require_group(subgroup_name)
    # store data in dataset 'data'
    dataset_name = "data"
    # dataset shape used for creating. If not 2D NDArray, set as column vector
    dshape = result.shape if len(result.shape) != 1 else (result.shape[0], 1)
    # check if the dataset already exists
    if dataset_name in subgroup:
        dataset = subgroup[dataset_name]
        # append new data horizontally
        dataset.resize((dataset.shape[1] + dshape[1]), axis=1)
        dataset[:, -dshape[1] :] = result
        # append phi value
        dataset.attrs["phis"] = np.append(dataset.attrs["phis"], config.phy_setting.phi)
    else:
        dataset = subgroup.create_dataset(
            dataset_name, data=result, shape=dshape, maxshape=(result.shape[0], None)
        )
        # create dataset attributes
        dataset.attrs["phis"] = np.array([config.phy_setting.phi])
        dataset.attrs["feature_size"] = dshape[1]
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import pickle
import h5py
from itertools import product
from setup import PhysicsSettings, SolverSettings

def get_metadata(filepath: str) -> dict[str, float | int]:
    """get metadata of simulation from hdf5 file

    Args:
        filepath (str): filepath to the hdf5 file

    Returns:
        dict[str, float | int]: dictionary contains keys and values for metadata
    """
    with open(filepath, "rb") as f:
        metadata = pickle.load(f)
    return metadata

def get_param_list(
    filepath: str,
) -> tuple[list[str], list[list[str]], list[list[list[float]]]]:
    """get recorded sweep parameter list contained in the hdf5 file

    Args:
        filepath (str): file path to hdf5 file

    Returns:
        tuple[list[str], list[list[str]], list[list[list[float]]]]: tuple contains list for gamma, eta and phi values
        - list[str]: list contains group names (gamma values)
        - list[list[str]]: list contains datasets name (eta values) for each group
        - list[list[list[float]]]: list contains dataset attrs (phi values) for each datasets in each group
    """
    gamma_list = []
    eta_list = []
    phi_list = []
    with h5py.File(filepath, "r") as f:
        for g in list(f.keys()):
            if g != "metadata":
                gamma_list.append(g)
                eta_list.append(list(f[g].keys()))
                temp_phi_list = []
                for e in list(f[g].keys()):
                    # temp_phi_list.append(f[g][e].attrs["phis"])
                    temp_phi_list.append(list(f[g][e].keys()))
                phi_list.append(temp_phi_list)
    f.close()
    return gamma_list, eta_list, phi_list



def get_group_data(filepath: str, group_name: str) -> list[list[NDArray]]:
    """get all data and corresponding parameters within a group

    Args:
        filepath (str): file path to hdf5 file
        group_name (str): specified group name

    Returns:
        tuple[list[NDArray], list[tuple[float, float]]]: tuple contains list of data and corresponding parameters.
        - list[list[NDArray]]: list of numpy array for all the data within the group

    """
    data_list = []
    param_list = []
    with h5py.File(filepath, "r") as f:
        group = f[group_name]
        for e in list(group.keys()):
            subgroup = group[e]
            for phi in list(subgroup.keys()):
                dataset = subgroup[phi]
                index_len = dataset.attrs["(start_index, len)"]
                data = [dataset[:, row[0] : row[0] + row[1]] for row in index_len]
                data_list.append(data)
                param_list.append((e, phi))
            # dataset = group[e]
            # for i, phi in enumerate(dataset.attrs["phis"]):
            #    data_list.append(dataset[i, :])
            #    param_list.append((e, phcpi))
    return data_list, param_list


def load_data_from_config(config, f: h5py.File, type='concated'):
    group_name = f"gamma_{config.phy_setting.gamma:.2f}"
    subgroup_name = f"eta_{config.phy_setting.eta:.2f}"
    if type == 'concated':
        dataset = f[group_name][subgroup_name]["data"]
        index = np.argwhere(dataset.attrs["phis"] == config.phy_setting.phi)[0][0]
        feature_size = dataset.attrs["feature_size"]
        return dataset[:, index * feature_size : (index + 1) * feature_size]
    elif type == 'grouped':
        dataset_name=f"phi_{config.phy_setting.phi:.2f}"
        return f[group_name][subgroup_name][dataset_name][:]
    
def get_group_encoding(filepath: str, group_name: str) -> list[list[NDArray]]:
    data_list = []
    param_list = []
    with h5py.File(filepath, "r") as f:
        group = f[group_name]
        for e in list(group.keys()):
            subgroup = group[e]
            for phi in list(subgroup.keys()):
                data_list.append(subgroup[phi][:])
                param_list.append((e, phi))
    return data_list, param_list

def reshape_node_state(
    node_states: list[float], num_delay: int, num_nodes: int, skip: int | None = None
) -> NDArray:
    """convert list of node states in to matrix form for regression

    Args:
        node_states (list[float]): list that contains sampled node states
        num_delay (int): number of total delay the input data require
        num_node (int): number of node states

    Returns:
        NDArray: 2D numpy array with shape of [num_delay, num_node]. The num_delay is equivalent with dataset size, num_node is equivalent with feature dimension
    """
    if skip:
        return np.reshape(node_states[1:], [num_delay, num_nodes])[skip:, :]
    else:
        return np.reshape(node_states[1:], [num_delay, num_nodes])
    

def get_error_matrix(error_list, param_list, eta_list, phi_list) -> pd.DataFrame:
    column_names = phi_list[0]
    row_names = eta_list
    error_mat = pd.DataFrame(
        np.zeros((len(row_names), len(column_names))),
        columns=column_names,
        index=row_names,
    )
    for e, param in zip(error_list, param_list):
        error_mat.loc[param[0], param[1]] = e
    return error_mat


def create_df(etas, phis):
    column_names = [f"eta_{eta:.2f}" for eta in etas]
    row_names = [f"phi_{phi:.2f}" for phi in phis]
    df = pd.DataFrame(
        np.zeros((len(row_names), len(column_names))),
        columns=column_names,
        index=row_names,
    )
    return df


def create_df_from_str(etas_str, phis_str):
    df = pd.DataFrame(
        np.zeros((len(phis_str), len(etas_str))), columns=etas_str, index=phis_str
    )
    return df


def insert_data_to_df(data, df, param):
    df.loc[param[1], param[0]] = data


def create_df_dict(gammas, etas, phis):
    df_dict = {f"gamma_{gamma:.2f}": create_df(etas, phis) for gamma in gammas}
    return df_dict


def insert_data_to_dict(data, df_dict, config):
    df = df_dict[f"gamma_{config.phy_setting.gamma:.2f}"]
    df.loc[f"phi_{config.phy_setting.phi:.2f}", f"eta_{config.phy_setting.eta:.2f}"] = (
        data
    )




def normalize_data(data):
    return (data - np.mean(data, axis=1)) / (np.std(data, axis=1) + 1e-38)


    

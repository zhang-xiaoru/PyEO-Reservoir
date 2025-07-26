import numpy as np
import pandas as pd

# from sim_funcs import CONFIG
import os
import h5py
import pickle
import solver_funcs
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score

def normalized_rmse(y_true, y_pred):
    return rmse(y_true, y_pred) / np.std(y_true)

def normalized_mse(y_true, y_pred):
    return mse(y_true, y_pred) / (np.std(y_true) ** 2)


def get_encoding_state(node_states: list[float], num_nodes: int) -> NDArray:
    """return the nodes states in last delay cycle as equiavlent last hidden states

    Args:
        node_states (list[float]): _description_
        num_nodes (int): _description_

    Returns:
        NDArray: _description_
    """
    return node_states[:, -num_nodes:]


def get_node_states(
    sol: list[float], time_steps: list[float], delay: float, num_nodes: int
) -> list[float]:
    """resampling node states from integrated states solution

    Args:
        sol (list[float]): list contains complete solution on each integration steps.
        time_steps (list[float]): corresponding time steps for integration
        delay (float): delay time
        num_nodes (int): number of node states

    Returns:
        list[float]: list of sampled node states
    """
    time_steps_on_nodes = np.arange(
        0, int(time_steps[-1]) * (1 + 1e-6), delay / num_nodes
    )
    return solver_funcs.jit_interp(time_steps_on_nodes, sol, time_steps)


def normalized_rmse(y_true, y_pred):
    return rmse(y_true, y_pred) / np.std(y_true)


def regression(
    x_train: NDArray, y_train: NDArray, x_test: NDArray, y_test: NDArray, alpha: float
) -> tuple[tuple[NDArray, NDArray], tuple[float, float]]:
    """Perform ridge regression on training and testing dataset, return predicted data and normalized RMSE

    Args:
        x_train (NDArray): 2D numpy array for training predictor dataset with shape[n_size, n_feature]
        y_train (NDArray): numpy array for training response with shape[n_size]
        x_test (NDArray): 2D numpy array for testing predictor dataset
        y_test (NDArray): numpy array for testing response
        alpha (float): regularization parameter

    Returns:
        tuple[tuple[NDArray, NDArray], tuple[float, float]]: return tuple of model prediction and error
        - tuple[NDArray, NDArray]: tuple contains train prediction and test prediction
        - tuple[float, float]: tuple contains train and test NRMSE
    """
    model = Ridge(alpha)
    model.fit(x_train, y_train)
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)
    train_error = normalized_rmse(y_train, y_train_predict)
    test_error = normalized_rmse(y_test, y_test_predict)
    return y_train_predict, y_test_predict, train_error, test_error


def regressionCV(
        X_train, y_train, X_test, y_test, cv_train_start, cv_train_end, cv_test_start, cv_test_end, alpha
):
    train_nrmse_cv = np.zeros(len(cv_train_start))
    test_nrmse_cv = np.zeros(len(cv_test_start))
    for i, (train_start, train_end, test_start, test_end) in enumerate(
        zip(
            cv_train_start, cv_train_end, cv_test_start, cv_test_end
        )
    ):
        y_train_cv = y_train[train_start:train_end]
        y_test_cv = y_test[test_start:test_end]
        model = Ridge(alpha, solver="svd")
        X_train_cv = X_train[train_start:train_end, :]
        X_test_cv = X_test[test_start:test_end, :]
        model.fit(X_train_cv, y_train_cv)
        train_pred = model.predict(X_train_cv)
        test_pred = model.predict(X_test_cv)
        train_nrmse_cv[i] = normalized_mse(y_train_cv, train_pred)
        test_nrmse_cv[i] = normalized_mse(y_test_cv, test_pred)
    return np.mean(train_nrmse_cv), np.mean(test_nrmse_cv), np.std(train_nrmse_cv), np.std(test_nrmse_cv)

def classification(
    X_train: NDArray, y_train: NDArray, X_test: NDArray, y_test: NDArray, alpha: float
) -> tuple[tuple[NDArray, NDArray], tuple[float, float]]:
    model = RidgeClassifier(alpha)
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_predict)
    test_acc = accuracy_score(y_test, y_test_predict)
    return y_train_predict, y_test_predict, train_acc, test_acc


def get_classfication_prediction_count(prediction, splitLen_or_idx):
    if prediction.ndim > 1:
        splited_prediction = np.split(prediction, splitLen_or_idx, axis=1)
    else:
        splited_prediction = np.split(prediction, splitLen_or_idx)
    prediction_count = [np.zeros(10) for _ in splited_prediction]
    for i, p in enumerate(splited_prediction):
        unique_values, counts = np.unique(p, return_counts=True)
        for u, c in zip(unique_values, counts):
            prediction_count[i][u] = c
    prediction_per_gram = np.array([np.argmax(pc ) for pc in prediction_count])
    return prediction_count, prediction_per_gram




# def classification(
#        X: NDArray, y: NDArray, alpha: float, test_size: float, random_state: int
# ) -> tuple[tuple[NDArray, NDArray], tuple[float, float]]:
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=test_size, random_state=random_state, stratify=y
#    )
#    model= RidgeClassifier(alpha)
#    model.fit(X_train, y_train)
#    y_train_predict = model.predict(X_train)
#    y_test_predict = model.predict(X_test)
#    train_acc = accuracy_score(y_train, y_train_predict)
#    test_acc = accuracy_score(y_test, y_test_predict)
#    return y_train_predict, y_test_predict, train_acc, test_acc




def save_prediction(
    savepath: str,
    predictions: list[NDArray],
    group_name: str,
    param_list: list[tuple[str, float]],
) -> None:
    if os.path.exists(savepath):
        os.remove(savepath)
    with h5py.File(savepath, "a") as f:
        group = f.require_group(group_name)
        for p, param in zip(predictions, param_list):
            subgourp = group.require_group(param[0])
            # dataset_name = param[0]
            # if param[0] in group:
            if param[1] in subgourp:
                # get the dataset if already exsists
                dataset = group[param[1]]
                # resize the dataset for appending new result
                dataset.resize((dataset.shape[0] + 1), axis=0)
                dataset[dataset.shape[0] - 1 :] = p.T
                # append corresponding phi values to the dataset attributes
                # dataset.attrs["phis"] = np.append(dataset.attrs["phis"], param[1])
            else:
                # create dataset if not exist
                dataset = subgourp.create_dataset(
                    param[1], shape=(1, len(p)), data=p.T, maxshape=(None, len(p))
                )
                # create dataset attribute to record phi values
                # dataset.attrs["phis"] = np.array([param[1]])
    f.close()




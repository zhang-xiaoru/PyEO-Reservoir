import numpy as np
from ..resevoir import setup, readout, helper
from ..resevoir.setup import CONFIG
import pandas as pd
import shutil
import os
import h5py
import time
import pickle
from joblib import Parallel, delayed, cpu_count
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import gc


def get_split_index(data_list, concated_axis=1):
    split_idx = []
    prev_idx = 0
    for data in data_list:
        idx = prev_idx + data.shape[concated_axis]
        split_idx.append(idx)
        prev_idx = idx
    last_idx = split_idx.pop()
    return split_idx, last_idx


def get_prediction_from_decisionMat(decision_mat):
    margin = np.sum(decision_mat, axis=0)
    return np.argmax(margin)


def voice_classification(X_train, y_train, X_test, y_test, alpha):
    model = RidgeClassifier(alpha)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_pred, test_pred, train_acc, test_acc


def voice_classificationCV(X, y, skf, alpha):
    model = RidgeClassifier(alpha, solver='svd')
    score = cross_validate(
        model, X, y, cv=skf.split(X, y), return_train_score=True, scoring="accuracy"
    )
    train_acc_avg = np.mean(score["train_score"])
    train_acc_best = np.max(score["train_score"])
    train_acc_std = np.std(score["train_score"])
    test_acc_avg = np.mean(score["test_score"])
    test_acc_best = np.max(score["test_score"])
    test_acc_std = np.std(score["test_score"])
    return (
        train_acc_avg,
        test_acc_avg,
        train_acc_best,
        test_acc_best,
        train_acc_std,
        test_acc_std,
    )


"""
def voice_classification(
    X_train, y_train, X_test, train_splt_idx, test_splt_idx, alpha
):
    model = RidgeClassifier(alpha)
    model.fit(X_train, y_train)
    train_decisionMat = model.decision_function(X_train)
    test_decisionMat = model.decision_function(X_test)
    train_pred = [
        get_prediction_from_decisionMat(d)
        for d in np.split(train_decisionMat, train_splt_idx, axis=0)
    ]
    test_pred = [
        get_prediction_from_decisionMat(d)
        for d in np.split(test_decisionMat, test_splt_idx, axis=0)
    ]
    return train_pred, test_pred
"""

if __name__ == "__main__":
    # set up output directory
    version_num = 1
    date_pref = "20250526"
    output_dir = "../output/" + date_pref + f"_{version_num:02}"
    overwrite = True

    # set up sim directory
    # sim_output_dir = "/work/10331/albertzhang8452/stampede3/job_holder_output/delay_esn/20250401_00/sim_result"
    sim_output_dir = (
        "/scratch/10331/albertzhang8452/job_holder/output/20250422_04/sim_result"
    )

    # gammas = [2.5, 2.7, 2.9]

    # regularization parameter list
    # alpha_list = np.logspace(-10, 2, 50)
    alpha_list = np.append(0, np.logspace(-9, 1, 21))
    #alpha_list = [0]
    chunksize = 48 * 5
    num_com_node = -1

    # get simulation metadata
    metadata = helper.get_metadata(sim_output_dir + "/metadata.pkl")
    tau = metadata["delay (ps)"]
    n_int = metadata["num of int steps"]
    num_nodes = metadata["delay node sizes"]
    t_local = metadata["local time scale"]
    warmup_skip = metadata["warmup length"]
    gammas = metadata["gammas"]
    save_partition = metadata["partition size"]
    etas = metadata["etas"]
    phis = metadata["phis"]

    # load data
    data_dir = "/work/10331/albertzhang8452/stampede3/projects/delay_esn/data/voice"
    with open(data_dir + "/y_data_3person.pkl", "rb") as f:
        labels = pickle.load(f)

    # get train and test data

    # set up solver setting
    sol_setting_list = []
    for d in num_nodes:
        sol_base_setting = {
            "delay_node_size": d,
            "delay": tau,
            "j_step_len": tau / d,
            "n_int": n_int,
        }
        sol_setting_list += setup.make_setting_list(None, sol_base_setting, "sol")

    # set up physics setting
    phy_base_setting = {"t_local": t_local}
    gammas_partition = np.split(gammas, save_partition)
    phy_setting_list = []
    for g in gammas_partition:
        phy_sweep_setting = {"eta": etas, "gamma": g, "phi": phis}
        phy_setting_list.append(
            setup.make_setting_list(phy_sweep_setting, phy_base_setting, setting="phy")
        )


    # set up output directory
    # use this for TACC
    if overwrite:
        fit_output_dir = output_dir + "/fit_result"
        try:
            os.makedirs(fit_output_dir)
        except:
            shutil.rmtree(fit_output_dir)
            os.makedirs(fit_output_dir)
    else:
        while os.path.exists(output_dir):
            version_num += 1
            output_dir = "../output/" + date_pref + f"_{version_num:02}"
        fit_output_dir = output_dir + "/fit_result"
        os.makedirs(fit_output_dir, exist_ok=True)

    print("Using simulation file at " + sim_output_dir)
    print("Data saved in " + output_dir)
    print(
        f"Number of core in used : {cpu_count() if num_com_node == -1 else num_com_node}"
    )
    print(f"Chunksize: {chunksize}\n")

    train_acc_avg_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    test_acc_avg_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    train_acc_best_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    test_acc_best_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    train_acc_std_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    test_acc_std_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    
    # set up cross validation
    skf = StratifiedKFold(10, shuffle=True, random_state=42)

    ###
    ### Start classification
    ####
    for sol_setting in sol_setting_list:
        # saving file name
        acc_data_path = [
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_acc_avg_train.pkl",
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_acc_avg_test.pkl",
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_acc_best_train.pkl",
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_acc_best_test.pkl",
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_acc_std_train.pkl",
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_acc_std_test.pkl",
        ]
        # form configuration setting list
        for p in range(save_partition):
            print(f"Load data for partition {p + 1}/{save_partition}.")
            partition_start_time = time.perf_counter()
            config_list = [
                CONFIG(phy_setting, sol_setting) for phy_setting in phy_setting_list[p]
            ]

            # saved data file name
            sim_data_path = (
                sim_output_dir
                + f"/N_{sol_setting.delay_node_size:03}_partition{p:d}.hdf5"
            )

            # get split index
            with h5py.File(sim_data_path, "r") as f:
                split_idx = f["labels_and_split_idx"]["split_idx"][:]
                # labels_repetaed = np.split(
                #    f["labels_and_split_idx"]["repeated_labels"][:], split_idx
                # )

            unskiped_split_idx = split_idx + np.arange(
                warmup_skip, warmup_skip * len(labels), warmup_skip
            )

            # get split index for train and test dataset
            # train_splt_idx, _ = get_split_index(y_train_repeated, 0)
            # test_splt_idx, _ = get_split_index(y_test_repeated, 0)
            # y_train_repeated = np.concatenate(y_train_repeated)
            # y_test_repeated = np.concatenate(y_test_repeated)

            for i in range(0, len(config_list), chunksize):
                print(f"Start chunk {i // chunksize + 1:d} of the partition.")
                chunk_start_time = time.perf_counter()
                config_chunk = config_list[i : i + chunksize]
                start_time = time.perf_counter()

                hidden_states_list = np.zeros(
                    (
                        len(config_chunk),
                        labels.shape[0],
                        sol_setting.delay_node_size * 2,
                    )
                )

                # get reservoir states
                with h5py.File(sim_data_path, "r") as f:
                    for c, config in enumerate(config_chunk):
                        hidden_states = helper.load_data_from_config(
                            config, f, "grouped"
                        )
                        mean_hidden_states_splt_rm = np.vstack(
                            [
                                np.mean(h[warmup_skip:], axis=0)
                                for h in np.split(
                                    hidden_states, unskiped_split_idx, axis=0
                                )
                            ]
                        )

                        ### construct using square of mean
                        #hidden_states_list[c] = np.hstack(
                        #    (mean_hidden_states_splt_rm, mean_hidden_states_splt_rm ** 2)
                        #    )

                        ### construct using mean of square
                        mean_hidden_states2_splt_rm = np.vstack(
                            [
                                np.mean(h[warmup_skip:] ** 2, axis=0)
                                for h in np.split(
                                    hidden_states, unskiped_split_idx, axis=0
                                )
                            ]

                        )
                        hidden_states_list[c] = np.hstack(
                            (mean_hidden_states_splt_rm, mean_hidden_states2_splt_rm)
                        )

                end_time = time.perf_counter()
                print(
                    f"**** Time used for loading states data: {end_time - start_time:.4f}"
                )

                start_time = time.perf_counter()
                # run regression for different regularization parameter
                for j, a in enumerate(alpha_list):
                    result = Parallel(
                        n_jobs=num_com_node, batch_size="auto", verbose=0
                    )(
                        delayed(voice_classificationCV)(
                            h,
                            labels,
                            skf,
                            a,
                        )
                        for h in hidden_states_list
                    )
                    (
                        train_acc_avg_list,
                        test_acc_avg_list,
                        train_acc_best_list,
                        test_acc_best_list,
                        train_acc_std_list,
                        test_acc_std_list,
                    ) = map(list, zip(*result))

                    # save error into error matrix
                    for (
                        train_acc_avg,
                        test_acc_avg,
                        train_acc_best,
                        test_acc_best,
                        train_acc_std,
                        test_acc_std,
                        config,
                    ) in zip(
                        train_acc_avg_list,
                        test_acc_avg_list,
                        train_acc_best_list,
                        test_acc_best_list,
                        train_acc_std_list,
                        test_acc_std_list,
                        config_chunk,
                    ):
                        helper.insert_data_to_dict(
                            train_acc_avg, train_acc_avg_dict_list[j], config
                        )
                        helper.insert_data_to_dict(
                            test_acc_avg, test_acc_avg_dict_list[j], config
                        )
                        helper.insert_data_to_dict(
                            train_acc_best, train_acc_best_dict_list[j], config
                        )
                        helper.insert_data_to_dict(
                            test_acc_best, test_acc_best_dict_list[j], config
                        )
                        helper.insert_data_to_dict(
                            train_acc_std, train_acc_std_dict_list[j], config
                        )
                        helper.insert_data_to_dict(
                            test_acc_std, test_acc_std_dict_list[j], config
                        )
                    if (j + 1) % 5 == 0:
                        end_time = time.perf_counter()
                        print(
                            f"**** {min(j / len(alpha_list), 1) * 100:.2f}% regularization parameters finished. Time elapsed: {end_time - start_time:.4f}"
                        )
                chunk_end_time = time.perf_counter()
                print(
                    f"{min((i + chunksize) / len(config_list), 1) * 100:.2f}% chunk completed. Total time used for chunk {i // chunksize + 1}: {chunk_end_time - chunk_start_time:.4f}."
                )

            partition_end_time = time.perf_counter()
            print(
                f"{p + 1}/{save_partition} of partition complete. Time used for partition: {(partition_end_time - partition_start_time) / 3600:.4f} hours.\n"
            )

        with open(acc_data_path[0], "wb") as f:
            pickle.dump(train_acc_avg_dict_list, f)

        with open(acc_data_path[1], "wb") as f:
            pickle.dump(test_acc_avg_dict_list, f)

        with open(acc_data_path[2], "wb") as f:
            pickle.dump(train_acc_best_dict_list, f)

        with open(acc_data_path[3], "wb") as f:
            pickle.dump(test_acc_best_dict_list, f)

        with open(acc_data_path[4], "wb") as f:
            pickle.dump(train_acc_std_dict_list, f)

        with open(acc_data_path[5], "wb") as f:
            pickle.dump(test_acc_std_dict_list, f)

        print("Train/Test accuracy matrix saved.\n")

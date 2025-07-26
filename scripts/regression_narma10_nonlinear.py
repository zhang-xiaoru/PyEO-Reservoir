import numpy as np
#import processing_funcs as pfuncs
#import sim_funcs as sfuncs
#from sim_funcs import CONFIG
from ..resevoir import setup, readout, helper
from ..resevoir.setup import CONFIG
import pandas as pd
import shutil
import os
import h5py
import time
import pickle
from joblib import Parallel, delayed, cpu_count
import gc


if __name__ == "__main__":
    # set up output directory
    version_num = 0
    date_pref = "20250420"
    output_dir = "../output/" + date_pref + f"_{version_num:02}"
    overwrite = True

    # set up sim directory
    # sim_output_dir = "/work/10331/albertzhang8452/ls6/job_holder_output/delay_esn/20250305_00/sim_result"
    sim_output_dir = (
        "/scratch/10331/albertzhang8452/job_holder/output/20250418_00/sim_result"
    )

    # gammas = [2.5, 2.7, 2.9]

    # regularization parameter list
    # alpha_list = np.logspace(-10, 2, 50)
    alpha_list = np.append(0, np.logspace(-10, 0, 21))
    #alpha_list = [0]
    chunksize = 48 * 5
    num_com_node = -1

    # get simulation metadata
    metadata = helper.get_metadata(sim_output_dir + "/metadata.pkl")
    tau = metadata["delay (10ps)"]
    n_int = metadata["num of int steps"]
    num_nodes = metadata["delay node sizes"]
    # train_start = metadata["train start"]
    # train_end = metadata["train end"]
    # test_start = metadata["test start"]
    # test_end = metadata["test end"]
    t_local = metadata["local time scale (10ps)"]
    gammas = metadata["gammas"]
    etas = metadata["etas"]
    phis = metadata["phis"]

    train_skip = 500
    test_skip = 500
    train_size = 1000
    test_size = 1000

    cv_train_start = np.arange(0, 3000, 300) + train_skip
    cv_train_end = cv_train_start + train_size
    cv_test_start = np.arange(0, 3000, 300) + test_skip
    cv_test_end = cv_test_start + test_size

    # train_start = 1000
    # test_start = 1000
    # train_end = 2000
    # test_end = 2000
    # train_size = train_end - train_start
    # test_size = test_end - test_start
    # train_skip = 500
    # test_skip = 500

    # load data
    narma10_train = pd.read_csv(
        "/work/10331/albertzhang8452/ls6/projects/delay_esn/data/narma10/narma10_train_5000.csv"
    )
    narma10_test = pd.read_csv(
        "/work/10331/albertzhang8452/ls6/projects/delay_esn/data/narma10/narma10_test_5000.csv"
    )

    # get train and test data
    # y_train = narma10_train["y"].values[None, train_start : train_end].T
    # y_test = narma10_test["y"].values[None, test_start : test_end].T

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
    phy_sweep_setting = {"eta": etas, "gamma": gammas, "phi": phis}
    phy_setting_list = setup.make_setting_list(
        phy_sweep_setting, phy_base_setting, setting="phy"
    )
    setup.make_setting_list(phy_sweep_setting, phy_base_setting, "phy")

    print("Data saved in " + output_dir)
    print(f"Number of core in used : {cpu_count()}")
    print(f"Chunksize: {chunksize}\n")

    # set up output directory
    # use this for TACC
    if overwrite:
        fit_output_dir = output_dir + "/fit_result"
        try:
            os.makedirs(fit_output_dir)
        except:
            shutil.rmtree(output_dir)
            os.makedirs(fit_output_dir)
    else:
        while os.path.exists(output_dir):
            version_num += 1
            output_dir = "../output/" + date_pref + f"_{version_num:02}"
        fit_output_dir = output_dir + "/fit_result"
        os.makedirs(fit_output_dir, exist_ok=True)

    train_nrmse_avg_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    test_nrmse_avg_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    train_nrmse_std_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]
    test_nrmse_std_dict_list = [
        helper.create_df_dict(gammas, etas, phis) for _ in alpha_list
    ]

    for sol_setting in sol_setting_list:
        # form configuration setting list
        config_list = [
            CONFIG(phy_setting, sol_setting) for phy_setting in phy_setting_list
        ]

        # saved data file name
        sim_data_path = [
            sim_output_dir + f"/N_{sol_setting.delay_node_size:03}_train.hdf5",
            sim_output_dir + f"/N_{sol_setting.delay_node_size:03}_test.hdf5",
        ]

        # saving file name
        error_data_path = [
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_nrmse_avg_train.pkl",
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_nrmse_avg_test.pkl",
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_nrmse_std_train.pkl",
            fit_output_dir + f"/N_{sol_setting.delay_node_size:03}_nrmse_std_test.pkl",
        ]

        for i in range(0, len(config_list), chunksize):
            print(f"Start chunk {i // chunksize + 1:d}")
            chunk_start_time = time.perf_counter()
            config_chunk = config_list[i : i + chunksize]
            start_time = time.perf_counter()

            # get reservoir states
            train_hidden_states_list = np.zeros(
                (len(config_chunk), 5000, sol_setting.delay_node_size * 2)
            )
            test_hidden_states_list = np.zeros(
                (len(config_chunk), 5000, sol_setting.delay_node_size * 2)
            )
            with h5py.File(sim_data_path[0], "r") as f:
                for c, config in enumerate(config_chunk):
                    hidden_states = helper.load_data_from_config(config, f)
                    train_hidden_states_list[c] = np.hstack(
                        (hidden_states, hidden_states**2)
                    )
            with h5py.File(sim_data_path[1], "r") as f:
                for c, config in enumerate(config_chunk):
                    hidden_states = helper.load_data_from_config(config, f)
                    test_hidden_states_list[c] = np.hstack(
                        (hidden_states, hidden_states**2)
                    )

            end_time = time.perf_counter()
            print(f"Time used for loading states data: {end_time - start_time:.4f}")

            start_time = time.perf_counter()
            # run regression for different regularization parameter
            for j, a in enumerate(alpha_list):
                y_train = narma10_train["y"].values
                y_test = narma10_test["y"].values
                result = Parallel(
                    n_jobs=num_com_node, batch_size="auto", verbose=0
                )(
                    delayed(readout.regressionCV)(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        cv_train_start,
                        cv_train_end,
                        cv_test_start,
                        cv_test_end,
                        a
                    )
                    for X_train, X_test in zip(
                        train_hidden_states_list, test_hidden_states_list
                    )
                )


                # save error into error matrix
                (
                    train_nrmse_avg_list,
                    test_nrmse_avg_list,
                    train_nrmse_std_list,
                    test_nrmse_std_list
                ) = map(list, zip(*result))

                for train_avg, test_avg, train_std, test_std, config in zip(
                    train_nrmse_avg_list,
                    test_nrmse_avg_list,
                    train_nrmse_std_list,
                    test_nrmse_std_list,
                    config_chunk,
                ):
                    helper.insert_data_to_dict(
                        train_avg, train_nrmse_avg_dict_list[j], config
                    )
                    helper.insert_data_to_dict(
                        test_avg, test_nrmse_avg_dict_list[j], config
                    )
                    helper.insert_data_to_dict(
                        train_std, train_nrmse_std_dict_list[j], config
                    )
                    helper.insert_data_to_dict(
                        test_std, test_nrmse_std_dict_list[j], config
                    )

                if (j + 1) % 7 == 0:
                    end_time = time.perf_counter()
                    print(
                        f"****{min(j / len(alpha_list), 1) * 100:.2f}% regularization parameters finished. Time elapsed: {end_time - start_time:.4f}"
                    )
            chunk_end_time = time.perf_counter()
            print(
                f"{min((i + chunksize) / len(config_list), 1) * 100:.2f}% chunk completed. Total time used for chunk {i // chunksize + 1}: {chunk_end_time - chunk_start_time:.4f}\n"
            )

        with open(error_data_path[0], "wb") as f:
            pickle.dump(train_nrmse_avg_dict_list, f)

        with open(error_data_path[1], "wb") as f:
            pickle.dump(test_nrmse_avg_dict_list, f)

        with open(error_data_path[2], "wb") as f:
            pickle.dump(test_nrmse_std_dict_list, f)

        with open(error_data_path[3], "wb") as f:
            pickle.dump(test_nrmse_std_dict_list, f)

        print("Train/Test average error and error std matrix saved.\n")

    # save regularization parameter
    metadata = {}
    metadata["regularization parameters"] = alpha_list
    metadata["cv train start"] = cv_train_start
    metadata["cv test start"] = cv_test_start
    metadata["train size"] = train_size
    metadata["test size"] = test_size
    # metadata["train skip"] = train_skip
    # metadata["test skip"] = test_skip

    with open(fit_output_dir + f"/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"metadata saved for N={sol_setting.delay_node_size:03}")

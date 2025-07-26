# %%
import numpy as np
#import sim_funcs as sfuncs
#from sim_funcs import CONFIG
import pandas as pd
from ..resevoir import helper, readout, setup, dynamics
from ..resevoir.setup import CONFIG
#import processing_funcs as pfuncs
from joblib import Parallel, delayed, cpu_count
import os
import shutil
import gc
import time
import pickle
import h5py

if __name__ == "__main__":
    ######
    ### change parameters here
    ######

    # set up data paramateters
    train_start = 0
    train_len = 5000
    test_start = 0
    test_len = 5000

    # set up simulation configuration settings
    delay_node_sizes = np.array([40])
    tau = 44 # in unit of 10^-11
    n_int = 1000
    etas = np.arange(0.05, 0.45, 0.025)
    gammas = np.append(np.arange(0.01, 0.1, 0.02), np.arange(0.1, 5, 0.2))
    phis = np.arange(0, 1, 0.05)
    # etas = np.arange(0.3, 1.5, 0.01)
    # gammas = np.array([2.5, 2.7, 2.9])
    # phis = np.arange(0, 0.3, 0.01)
    #gammas = np.array([0.01])
    # etas = np.array([0.3, 0.5])
    # phis = np.array([0.05, 0.1])

    chunksize = 48 * 5
    num_com_node = -1

    # set up output directory
    version_num = 0
    date_pref = "20250418"
    output_dir = "../output/" + date_pref + f"_{version_num:02}"
    overwrite = True

    ########
    ### main script
    ########

    # load data
    narma10_train = pd.read_csv(
        "/work/10331/albertzhang8452/ls6/projects/delay_esn/data/narma10/narma10_train_5000.csv"
    )
    narma10_test = pd.read_csv(
        "/work/10331/albertzhang8452/ls6/projects/delay_esn/data/narma10/narma10_test_5000.csv"
    )
    # set up traning and testing length
    train_end = train_start + train_len
    test_end = test_start + test_len
    # get train and test data
    u_train = narma10_train["u"].values[None, train_start:train_end]
    # y_train = narma10_train["y"].values[None, train_start + train_skip : train_end].T
    u_test = narma10_test["u"].values[None, test_start:test_end]
    # y_test = narma10_test["y"].values[None, test_start + test_skip : test_end].T

    # set up solver setting
    sol_setting_list = []
    for d in delay_node_sizes:
        sol_base_setting = {
            "delay_node_size": d,
            "delay": tau,
            "j_step_len": tau / d,
            "n_int": n_int,
        }
        sol_setting_list += setup.make_setting_list(None, sol_base_setting, "sol")

    # set up physics setting
    phy_base_setting = {"t_local": 5.5}
    phy_sweep_setting = {"eta": etas, "gamma": gammas, "phi": phis}
    phy_setting_list = setup.make_setting_list(
        phy_sweep_setting, phy_base_setting, setting="phy"
    )

    # set up output directory
    # use this for TACC
    if overwrite:
        sim_output_dir = output_dir + "/sim_result"
        fit_output_dir = output_dir + "/fit_result"
        try:
            os.makedirs(sim_output_dir)
            os.makedirs(fit_output_dir)
        except:
            shutil.rmtree(output_dir)
            os.makedirs(sim_output_dir)
            os.makedirs(fit_output_dir)
    else:
        while os.path.exists(output_dir):
            version_num += 1
            output_dir = "../output/" + date_pref + f"_{version_num:02}"
        sim_output_dir = output_dir + "/sim_result"
        fit_output_dir = output_dir + "/fit_result"
        os.makedirs(sim_output_dir, exist_ok=True)
        os.makedirs(fit_output_dir, exist_ok=True)

    print("Data saved in " + output_dir)
    print(f"Number of core in used : {cpu_count()}")
    print(f"Chunksize: {chunksize}\n")

    # start Ikeda simulations
    total_iter = len(sol_setting_list)
    for sol_setting in sol_setting_list:
        print(f"Start parameteric sweep for N={sol_setting.delay_node_size:03}.")

        # create input masked data
        np.random.seed(42)
        m = (
            np.random.randint(0, 2, (sol_setting.delay_node_size, u_train.shape[0]))
            - 0.5
        ) / 5
        j_train = setup.masking(u_train, m)
        j_test = setup.masking(u_test, m)
        init_points = np.zeros((sol_setting.n_int + 1))

        # form configuration setting list
        config_list = [
            CONFIG(phy_setting, sol_setting) for phy_setting in phy_setting_list
        ]
        # saveing file name
        sim_data_path = [
            sim_output_dir + f"/N_{sol_setting.delay_node_size:03}_train.hdf5",
            sim_output_dir + f"/N_{sol_setting.delay_node_size:03}_test.hdf5",
        ]

        # parallel calulcation of states for traning data
        for i in range(0, len(config_list), chunksize):
            print(f"Start chunk {i // chunksize + 1:d}")
            chunk_start_time = time.perf_counter()
            config_chunk = config_list[i : i + chunksize]
            start_time = time.perf_counter()
            train_states_list = Parallel(
                n_jobs=num_com_node, batch_size="auto", verbose=0
            )(
                delayed(dynamics.resevoir)(
                    j_train, init_points, train_len, config, True, "full"
                )
                for config in config_chunk
            )
            test_states_list = Parallel(
                n_jobs=num_com_node, batch_size="auto", verbose=0
            )(
                delayed(dynamics.resevoir)(
                    j_test, init_points, train_len, config, True, "full"
                )
                for config in config_chunk
            )
            end_time = time.perf_counter()
            print(f"Time used for computing: {end_time - start_time:.4f}")

            # save hidden states to hpf5
            start_time = time.perf_counter()
            with h5py.File(sim_data_path[0], "a") as f:
                for train_result, config in zip(train_states_list, config_chunk):
                    dynamics.save_resultFix_concated(train_result, config, f)
            with h5py.File(sim_data_path[1], "a") as f:
                for test_result, config in zip(test_states_list, config_chunk):
                    dynamics.save_resultFix_concated(test_result, config, f)
            end_time = time.perf_counter()
            print(f"Time used for saving result: {end_time - start_time:.4f}")

            chunk_end_time = time.perf_counter()
            print(
                f"{min((i + chunksize) / len(config_list), 1) * 100:.2f}% chunk completed. Total time used for chunk {i // chunksize + 1}: {chunk_end_time - chunk_start_time:.4f}\n"
                ""
            )

    # save sol_setting as metadata
    metadata = {}
    metadata["delay node sizes"] = delay_node_sizes
    metadata["delay (10ps)"] = tau
    metadata["local time scale (10ps)"] = phy_base_setting["t_local"]
    metadata["num of int steps"] = n_int
    metadata["int step len"] = tau / n_int
    metadata["train start"] = train_start
    metadata["train end"] = train_end
    metadata["test start"] = test_start
    metadata["test end"] = test_end
    metadata["gammas"] = gammas
    metadata["etas"] = etas
    metadata["phis"] = phis

    with open(sim_output_dir + f"/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"metadata saved for N={sol_setting.delay_node_size:03}.")

    print("Simulation complete.")


# %%

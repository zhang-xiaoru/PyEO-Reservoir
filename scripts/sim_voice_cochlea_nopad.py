# %%
import numpy as np
#from numpy.typing import NDArray
#import sim_funcs as sfuncs
#from sim_funcs import CONFIG
from ..resevoir import dynamics, helper, readout, setup
from ..resevoir.setup import CONFIG
from joblib import Parallel, delayed, cpu_count
import h5py
import os
import shutil
import gc
import time
import pickle

def get_split_index(data_list, concated_axis=1):
    split_idx = []
    prev_idx = 0
    for data in data_list:
        idx = prev_idx + data.shape[concated_axis]
        split_idx.append(idx)
        prev_idx = idx
    last_idx = split_idx.pop()
    return split_idx, last_idx



if __name__ == "__main__":
    ######
    ### change parameters here
    ######

    # set up simulation configuration settings
    delay_node_sizes = np.array([40])
    tau = 44 # in 1e-11 unit
    n_int = 1000
    warm_up_len = 50

    #etas = np.arange(0.05, 0.6, 0.05)
    etas = np.arange(0.05, 0.55, 0.025)
    gammas = np.array([0.01, 0.03, 0.05, 0.07])
    #gammas = np.append(np.arange(0.01, 0.1, 0.02), np.arange(0.1, 5, 0.2))
    phis = np.arange(0, 1, 0.05)
    # etas = np.arange(0.3, 1.5, 0.01)
    # gammas = np.array([1])
    # phis = np.arange(0, 0.3, 0.01)
    #etas = np.array([0.8])
    #gammas = np.array([1])
    #phis = np.array([0.05])

    chunksize = 80*5
    num_com_node = -1
    save_partition = 1

    # set up output dirrectory
    version_num = 4
    date_pref = "20250422"
    output_dir = "../output/" + date_pref + f"_{version_num:02}"
    overwrite = True

    ########
    ### main script
    ########

    # load data
    with open(
        "/work/10331/albertzhang8452/stampede3/projects/delay_esn/data/voice/cochleagram_data_3person_nopad.pkl",
        "rb",
    ) as f:
        X = pickle.load(f)
    with open(
        "/work/10331/albertzhang8452/stampede3/projects/delay_esn/data/voice/y_data_3person.pkl", "rb"
    ) as f:
        y = pickle.load(f)

    cochleagram_channel = X[0].shape[0]


    # create warm up array for warm up the resevoir
    warm_up = np.ones((cochleagram_channel, warm_up_len)) * 0.01

    # concated warm up array 
    X_concated = [np.hstack([warm_up, x]) for x in X]

    # repeat labels for all time step within cochleagram
    y_repeated = np.hstack([np.repeat(y, x.shape[1]) for x, y in zip(X, y)])

    # get the index for spliting the hidden states with respect to the length of each cochleagram
    split_idx, total_len = get_split_index(X)

    # release memeory
    del X, y
    gc.collect()

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
    gammas_partition = np.split(gammas, save_partition)
    phy_setting_list = []
    for g in gammas_partition:
        phy_sweep_setting = {"eta": etas, "gamma": g, "phi": phis}
        phy_setting_list.append(setup.make_setting_list(
            phy_sweep_setting, phy_base_setting, setting="phy"
        ))
        
    # set up output directory
    # use this for TACC
    if overwrite:
        sim_output_dir = output_dir + "/sim_result"
        try:
            os.makedirs(sim_output_dir)
        except:  # noqa: E722
            shutil.rmtree(output_dir)
            os.makedirs(sim_output_dir)
    else:
        while os.path.exists(output_dir):
            version_num += 1
            output_dir = "../output/" + date_pref + f"_{version_num:02}"
        sim_output_dir = output_dir + "/sim_result"
        os.makedirs(sim_output_dir, exist_ok=True)

    print("Data saved in " + output_dir)
    print(f"Number of saving partition: {save_partition}")
    print(f"Number of core in used: {cpu_count()}")
    print(f'Chunksize: {chunksize}\n')

    ##################################
    #### start Ikeda simulations #####
    ##################################
    np.random.seed(42)

    for sol_setting in sol_setting_list:
        print(f"Start parameteric sweep for N={sol_setting.delay_node_size:02}.")
        for p in range(save_partition):
        #for p in [4]:
            partition_start_time = time.perf_counter()
            print(f"Partition {p + 1}/{save_partition} of Gammas start ")
            # form configuration setting list
            config_list = [
                CONFIG(phy_setting, sol_setting) for phy_setting in phy_setting_list[p]
            ]
            
            # saveing file name
            sim_data_path = sim_output_dir + f"/N_{sol_setting.delay_node_size:03}_partition{p:d}.hdf5"
    
            # initial points
            init_points = np.zeros((sol_setting.n_int + 1))
    
            # create input mask (-0.1, 0.1)
            m = (
                np.random.randint(0, 2, (sol_setting.delay_node_size, cochleagram_channel))
                - 0.5
            ) / 5
    
            j_list = [setup.masking(x, m) for x in X_concated]
    
    
            for i in range(0, len(config_list), chunksize):
                # create chunk of config_list  
                config_chunk = config_list[i : i + chunksize]
    
                # run simulation in chunk
                start_time = time.perf_counter()
                config_start_time = time.perf_counter()
                for j, config in enumerate(config_chunk):
                    hidden_states = Parallel(
                        n_jobs=num_com_node, batch_size="auto", verbose=0
                    )(
                        delayed(dynamics.resevoir)(
                            j_train, init_points, X_concated[k].shape[1], config, True, "full"
                        )
                        for k, j_train in enumerate(j_list)
                    )
        
                    # save encoding states to HDF5
                    with h5py.File(sim_data_path, "a") as f:
                        dynamics.save_result_grouped(np.vstack(hidden_states), config, f)
                        
                    if (j + 1)%10 == 0:
                        config_end_time = time.perf_counter()
                        print(
                            f"**** {j + 1}/{len(config_chunk)} of chunk completed. Time elapesd : {config_end_time -     config_start_time:.4f}"
                        )
                        config_start_time = time.perf_counter()
        
                end_time = time.perf_counter()
                print(f"{min((i + chunksize) / len(config_list), 1) * 100}% of one partition complete. Time elapesd for chunk: {end_time - start_time:.4f}.")
            with h5py.File(sim_data_path, 'a') as f:
                g = f.require_group('labels_and_split_idx')
                #g.create_dataset("repeated_labels", data=y_repeated)
                g.create_dataset("split_idx", data=split_idx)
            partition_end_time = time.perf_counter()
            print(f"{p + 1}/{save_partition} of partition complete and saved. Time used for partition: {(partition_end_time - partition_start_time) / 3600:.4f} hours.\n")


    # save sol_setting as metadata
    metadata = {}
    metadata["delay node sizes"] = delay_node_sizes
    metadata["local time scale"] = phy_base_setting["t_local"]
    metadata["delay (ps)"] = tau
    metadata["num of int steps"] = n_int
    metadata["int step len"] = tau / n_int
    metadata["gammas"] = gammas
    metadata["etas"] = etas
    metadata["phis"] = phis
    metadata["warmup length"] = warm_up_len
    metadata["partition size"] = save_partition

    with open(sim_output_dir + f"/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("metadata saved. ")

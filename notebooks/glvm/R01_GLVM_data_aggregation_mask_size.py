#!/usr/bin/env python
# coding: utf-8

# In[2]:

################################################################
#                                                              #
#    Data aggregation for supplementary analyses               #
#           varying mask size                                  #
#             Supp. Fig S1                                     #
#                                                              #
################################################################


# pylint: disable=undefined-variable
get_ipython().run_line_magic("cd", "../../")
# pylint: enable=undefined-variable
import copy
import os
import pickle
import time
from datetime import datetime

import numpy as np
import yaml

from maskedvae.utils.utils import KLD_uvg, mse

# pylint: disable=undefined-variable
get_ipython().run_line_magic("matplotlib", "inline")
# pylint: enable=undefined-variable

# folder name for evaluation
date = datetime.now().strftime("%d%m%Y_%H%M" "%S")  # more easily readable


# In[6]:

many_data_conditions = False

description = (
    "_Figure_2_Gaussian_LVM" + "_many_data_conditions"
    if many_data_conditions
    else "_Figure_2_Gaussian_LVM"
)
n_conditions = 1
n_masks = 5

local_dir = "./data/glvm_rev/"
drop_dir = "./data/glvm_rev/"
paper_base = "./data/glvm_rev/"
run_directory = "./runs_rev/"
os.makedirs(local_dir, exist_ok=True)
os.makedirs(drop_dir, exist_ok=True)
os.makedirs(paper_base, exist_ok=True)

flag_decoder = "_1_19"
# load txt file as a list
with open(
    os.path.join(f"./runs/summary/R_all_runs_mask_size{flag_decoder}.txt"), "r"
) as f:
    mask_size_runs = f.read().splitlines()


# In[7]:
class return_args(object):
    """args"""

    def __init__(self, run):
        self.dir = run
        self.fig_root = "figs"
        self.seed = 0
        self.iters = 10000


methods = [
    "zero_imputation_mask_concatenated_encoder_only",
    "zero_imputation",
]
runs_split = {runc: [] for runc in methods}

for run_id, run in enumerate(mask_size_runs):
    # check if the run is a directory
    if os.path.isdir(os.path.join(run_directory, run)):
        args = return_args(run)

    # list the subdirectories
    subdirs = [
        name
        for name in os.listdir(os.path.join(run_directory, run))
        if os.path.isdir(os.path.join(run_directory, run, name))
    ]
    print(subdirs)
    for run_condition in subdirs:
        print(run_condition)
        # now read in the args file

        # load the args file at args.fig_root, args.dir, run_condition, "args.pkl"
        with open(
            os.path.join(run_directory, run, run_condition, "args.pkl"), "rb"
        ) as file:
            argsread = pickle.load(file)
            print(argsread)

        runs_split[run_condition].append(run)

print(runs_split)

runs_masked = runs_split["zero_imputation_mask_concatenated_encoder_only"]
runs_all_obs = runs_split["zero_imputation"]
runs = mask_size_runs  # runs_all_obs + runs_masked

methods = [
    "zero_imputation_mask_concatenated_encoder_only",
    "zero_imputation",
]
# In[8]:
# Save run_masked to yaml in summary folder
with open(
    os.path.join(f"./runs/summary/runs_masked_mask_sizes{flag_decoder}.yaml"),
    "w",
) as f:
    yaml.dump(runs_masked, f)
with open(
    os.path.join(f"./runs/summary/runs_all_obs_mask_sizes{flag_decoder}.yaml"),
    "w",
) as f:
    yaml.dump(runs_all_obs, f)
# # prepare dictionaries for the summary data

# In[6]:
# add dataset size to the dictionary
mask_size = range(20)

dicts = {}
for i in range(n_conditions):
    dicts[i] = []
mean_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
std_of_mean_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
corr_data_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
latent_mse_data_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
kld_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
mean_kld_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
std_kld_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
median_kld_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}


test_data = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
full_variance = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
full_mean = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
full_gt_variance = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}

test_gt_data = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
test_latents = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
test_inputs = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
test_recons = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
test_latent_mean = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
test_latent_var = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}


# gt variance
mean_gt_dict = {
    ds: {i: copy.deepcopy(dicts) for i in range(n_masks)} for ds in mask_size
}
# posterior variance
var_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
std_of_var_dict = {
    ds: {
        i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
    }
    for ds in mask_size
}
# gt variance
var_gt_dict = {
    ds: {i: copy.deepcopy(dicts) for i in range(n_masks)} for ds in mask_size
}  # ensure here that for all conditions it is the same value for the gt var


# In[10]:


import sys

sys.path.append("./maskedvae/")

# %%
mean_mask_ed = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
mean_enc = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
mean_zero = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}

var_mask_ed = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
var_enc = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
var_zero = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}

mean_gt = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
var_gt = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}


mean_gt_all = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
var_gt_all = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}


condition_list = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
C_list = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
d_list = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}
noise_list = {ds: {i: [] for i in range(n_masks)} for ds in mask_size}


for run_id, run in enumerate(runs):
    args = return_args(run)
    # SPECIFY THE ROOT FOLDER WITH THE RUNS
    args.fig_root = run_directory
    for run_condition in [
        "zero_imputation_mask_concatenated_encoder_only",
        "zero_imputation",
    ]:
        for mm in range(n_masks):
            if os.path.isdir(os.path.join(args.fig_root, args.dir, run_condition)):
                if (
                    run in runs_masked
                    and run_condition
                    == "zero_imputation_mask_concatenated_encoder_only"
                ) or (run in runs_all_obs and run_condition == "zero_imputation"):
                    try:
                        file = open(
                            os.path.join(
                                args.fig_root, args.dir, run_condition, "args.pkl"
                            ),
                            "rb",
                        )
                    except:
                        print(
                            "File not found... Run probably failed: ",
                            os.path.join(args.fig_root, args.dir, "econder_only"),
                        )
                    else:
                        argsread = pickle.load(file)
                        file.close()
                        c0 = argsread.noise[0][0]
                        # dataset size
                        ds_size = argsread.n_masked_vals

                        if c0 not in condition_list[ds_size][mm]:
                            idx_cond = len(condition_list[ds_size][mm])
                            condition_list[ds_size][mm].append(c0)
                            C_list[ds_size][mm].append(argsread.C)
                            d_list[ds_size][mm].append(argsread.d)
                            noise_list[ds_size][mm].append(argsread.noise)

                            print("not found... added: ", idx_cond)
                        else:
                            idx_cond = condition_list[ds_size][mm].index(c0)
                            print("found... condition: ", idx_cond)

                        file = open(
                            os.path.join(
                                args.fig_root,
                                args.dir,
                                run_condition,
                                "_posterior_var_mean{:d}.pkl".format(mm),
                            ),
                            "rb",
                        )
                        # dump information to that file
                        data = pickle.load(file)
                        # close the file
                        file.close()

                        # posterior mean and variance
                        mean_enc[ds_size][mm].append(data[0])
                        mean_gt[ds_size][mm].append(data[1])

                        var_enc[ds_size][mm].append(data[2])
                        var_gt[ds_size][mm].append(data[3])

                        # compute the kl divergencce and other summay metrics
                        mean_dict[ds_size][mm][run_condition][idx_cond].append(data[0])
                        corr_data_dict[ds_size][mm][run_condition][idx_cond].append(
                            np.corrcoef(data[0][:, 0], data[1])[0, 1]
                        )
                        kld_dict[ds_size][mm][run_condition][idx_cond].append(
                            KLD_uvg(data[0][:, 0], data[2], data[1], data[3])
                        )
                        median_kld_dict[ds_size][mm][run_condition][idx_cond].append(
                            np.nanmedian(
                                kld_dict[ds_size][mm][run_condition][idx_cond][-1]
                            )
                        )
                        mean_kld_dict[ds_size][mm][run_condition][idx_cond].append(
                            np.nanmean(
                                kld_dict[ds_size][mm][run_condition][idx_cond][-1]
                            )
                        )
                        std_kld_dict[ds_size][mm][run_condition][idx_cond].append(
                            np.nanstd(
                                kld_dict[ds_size][mm][run_condition][idx_cond][-1]
                            )
                        )
                        latent_mse_data_dict[ds_size][mm][run_condition][
                            idx_cond
                        ].append(mse(data[0][:, 0], data[1]))
                        mean_gt_dict[ds_size][mm][idx_cond].append(data[1])

                        var_dict[ds_size][mm][run_condition][idx_cond].append(
                            np.mean(data[2])
                        )
                        std_of_var_dict[ds_size][mm][run_condition][idx_cond].append(
                            np.std(data[2])
                        )
                        var_gt_dict[ds_size][mm][idx_cond].append(data[3])
                        full_variance[ds_size][mm][run_condition][idx_cond].append(
                            data[2]
                        )
                        full_gt_variance[ds_size][mm][run_condition][idx_cond].append(
                            data[3]
                        )

                        # open the samples
                        file = open(
                            os.path.join(
                                args.fig_root,
                                args.dir,
                                run_condition,
                                "samples___gt_prior_recons_train_{:d}idx1_0_idx2_1.pkl".format(
                                    mm
                                ),
                            ),
                            "rb",
                        )
                        # dump information to that file
                        samples = pickle.load(file)
                        # close the file
                        file.close()
                        test_data[ds_size][mm][run_condition][idx_cond].append(
                            samples[0].T
                        )

                        file = open(
                            os.path.join(
                                args.fig_root,
                                args.dir,
                                run_condition,
                                "samples__gt_test_recons_{:d}.pkl".format(mm),
                            ),
                            "rb",
                        )
                        # dump information to that file
                        samples = pickle.load(file)
                        # close the file
                        file.close()
                        test_gt_data[ds_size][mm][run_condition][idx_cond].append(
                            samples[0].T
                        )
                        test_latents[ds_size][mm][run_condition][idx_cond].append(
                            samples[1].T
                        )
                        test_inputs[ds_size][mm][run_condition][idx_cond].append(
                            samples[2].T
                        )
                        test_recons[ds_size][mm][run_condition][idx_cond].append(
                            samples[3].T
                        )
                        test_latent_mean[ds_size][mm][run_condition][idx_cond].append(
                            samples[5].T
                        )
                        test_latent_var[ds_size][mm][run_condition][idx_cond].append(
                            samples[7].T
                        )
            else:
                print(
                    "run not found: ",
                    os.path.join(args.fig_root, args.dir, run_condition),
                )

# In[12]:

if many_data_conditions:
    print("many data conditions")
    with open(
        os.path.join(
            local_dir,
            f"entire_dictionary_list_many_data_conditions_ds_first{flag_decoder}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump([kld_dict, mean_kld_dict, std_kld_dict, median_kld_dict], f)

else:
    with open(
        os.path.join(local_dir, f"entire_dictionary_list_ds_first{flag_decoder}.pkl"),
        "wb",
    ) as f:
        pickle.dump(
            [
                mean_dict,
                std_of_mean_dict,
                corr_data_dict,
                latent_mse_data_dict,
                kld_dict,
                mean_kld_dict,
                std_kld_dict,
                median_kld_dict,
                test_data,
                full_variance,
                full_mean,
                full_gt_variance,
                test_gt_data,
                test_latents,
                test_inputs,
                test_recons,
                test_latent_mean,
                test_latent_var,
                mean_gt_dict,
                var_dict,
                std_of_var_dict,
                var_gt_dict,
                mean_mask_ed,
                mean_enc,
                mean_zero,
                var_mask_ed,
                var_enc,
                var_zero,
                mean_gt,
                var_gt,
                mean_gt_all,
                var_gt_all,
                condition_list,
                C_list,
                d_list,
                noise_list,
            ],
            f,
        )


# %%
# Store the models for the first n_conditions

import torch
import matplotlib.pyplot as plt

if many_data_conditions:
    n_conditions = 10
else:
    n_conditions = 1

condition_list = {ds: {i: [] for i, _ in enumerate(methods)} for ds in mask_size}

os.makedirs(
    os.path.join(local_dir, f"stored_models_ds_first{flag_decoder}"), exist_ok=True
)

for run_id, run in enumerate(runs):
    args = return_args(run)
    print(args.dir)
    args.fig_root = run_directory

    for i, method in enumerate(methods):
        print(i, method)

        model_to_load = os.path.join(
            args.fig_root, args.dir, method, "model_end_of_training.pt"
        )
        # check if the model exists else continue
        if not os.path.exists(model_to_load):
            print("model not found ", run, run_id, method)
            continue

        with open(model_to_load, "rb") as f:
            model = torch.load(f, map_location=torch.device("cpu"))
        c0 = model.args.noise[0][0]
        ds_size = model.args.n_masked_vals
        if c0 not in condition_list[ds_size][i]:
            idx_cond = len(condition_list[ds_size][i])
            condition_list[ds_size][i].append(c0)
            print("not found... added: ", idx_cond)
            model_path = os.path.join(
                local_dir,
                f"stored_models_ds_first{flag_decoder}",
                f"model_cond_{c0:.2f}_{method}_{ds_size}.pt",
            )
            with open(model_path, "wb") as f:
                torch.save(model, f)
        else:
            idx_cond = condition_list[ds_size][i].index(c0)
            print("found... condition: ", idx_cond)
        torch.manual_seed(model.args.seed)

        model_path = os.path.join(
            local_dir,
            f"stored_models_ds_first{flag_decoder}",
            f"model_{run_id}_{method}_{ds_size}.pt",
        )

        # store the model in the local data dictionary under model__runid__method.pt
        with open(model_path, "wb") as f:
            torch.save(model, f)

    # # if condition list full for all masks, break
    # if all(
    #     [len(condition_list[ds_size][i]) == n_conditions for i, _ in enumerate(methods)]
    # ):
    #     break

    if all(
        all(
            len(condition_list[ds_size][i]) == n_conditions
            for i, _ in enumerate(methods)
        )
        for ds_size in range(20)
    ):
        break

# %%

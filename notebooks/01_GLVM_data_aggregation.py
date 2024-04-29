#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic("cd", "../")

import copy
import os
import pickle
import time
from datetime import datetime

import numpy as np
import yaml

from maskedvae.utils.utils import KLD_uvg, mse

get_ipython().run_line_magic("matplotlib", "inline")


# folder name for evaluation
date = datetime.now().strftime("%d%m%Y_%H%M" "%S")  # more easily readable


# In[6]:

many_data_conditions = False

description = (
    "_Figure_2_Gaussian_LVM" + "_many_data_conditions"
    if many_data_conditions
    else "_Figure_2_Gaussian_LVM"
)
n_conditions = 10
n_masks = 5

local_dir = "./data/glvm/"
drop_dir = "./data/glvm/"
paper_base = "./data/glvm/"
run_directory = "./runs/"

# read in yaml files containing the runs see conversion helper in utils
with open(os.path.join(drop_dir, "runs_all_obs_glvm.yaml"), "r") as outfile:
    runs_all_obs = yaml.load(outfile, Loader=yaml.FullLoader)

with open(os.path.join(drop_dir, "runs_masked_glvm.yaml"), "r") as outfile:
    runs_masked = yaml.load(outfile, Loader=yaml.FullLoader)


if (
    many_data_conditions
):  # run the same model on many data conditions i.e. different C and sigma args.listlen = 10
    with open(
        os.path.join(drop_dir, "runs_all_obs_many_data_conditions.yaml"), "r"
    ) as outfile:
        runs_all_obs = yaml.load(outfile, Loader=yaml.FullLoader)

    with open(
        os.path.join(drop_dir, "runs_masked_many_data_conditions.yaml"), "r"
    ) as outfile:
        runs_masked = yaml.load(outfile, Loader=yaml.FullLoader)

runs = runs_all_obs + runs_masked

methods = [
    "zero_imputation_mask_concatenated_encoder_only",
    "zero_imputation",
]


# # prepare dictionaries for the summary data

# In[6]:


dicts = {}
for i in range(n_conditions):
    dicts[i] = []
mean_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
std_of_mean_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
corr_data_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
latent_mse_data_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
kld_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
mean_kld_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
std_kld_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
median_kld_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}


test_data = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
full_variance = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
full_mean = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
full_gt_variance = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}

test_gt_data = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
test_latents = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
test_inputs = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
test_recons = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
test_latent_mean = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
test_latent_var = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}


# gt variance
mean_gt_dict = {
    i: copy.deepcopy(dicts) for i in range(n_masks)
}  # ensure here that for all conditions it is the same value for the gt var
# posterior variance
var_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
std_of_var_dict = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
# gt variance
var_gt_dict = {
    i: copy.deepcopy(dicts) for i in range(n_masks)
}  # ensure here that for all conditions it is the same value for the gt var


# In[10]:


import sys

sys.path.append("./maskedvae/")

# %%
mean_mask_ed = {i: [] for i in range(n_masks)}
mean_enc = {i: [] for i in range(n_masks)}
mean_zero = {i: [] for i in range(n_masks)}

var_mask_ed = {i: [] for i in range(n_masks)}
var_enc = {i: [] for i in range(n_masks)}
var_zero = {i: [] for i in range(n_masks)}

mean_gt = {i: [] for i in range(n_masks)}
var_gt = {i: [] for i in range(n_masks)}


mean_gt_all = {i: [] for i in range(n_masks)}
var_gt_all = {i: [] for i in range(n_masks)}


condition_list = {i: [] for i in range(n_masks)}
C_list = {i: [] for i in range(n_masks)}
d_list = {i: [] for i in range(n_masks)}
noise_list = {i: [] for i in range(n_masks)}


class return_args(object):
    """args"""

    def __init__(self, run):
        self.dir = run
        self.fig_root = "figs"
        self.seed = 0
        self.iters = 10000


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
                        c0 = argsread.C[0][0]

                        if c0 not in condition_list[mm]:
                            idx_cond = len(condition_list[mm])
                            condition_list[mm].append(c0)
                            C_list[mm].append(argsread.C)
                            d_list[mm].append(argsread.d)
                            noise_list[mm].append(argsread.noise)

                            print("not found... added: ", idx_cond)
                        else:
                            idx_cond = condition_list[mm].index(c0)
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
                        mean_enc[mm].append(data[0])
                        mean_gt[mm].append(data[1])

                        var_enc[mm].append(data[2])
                        var_gt[mm].append(data[3])

                        # compute the kl divergencce and other summay metrics
                        mean_dict[mm][run_condition][idx_cond].append(data[0])
                        corr_data_dict[mm][run_condition][idx_cond].append(
                            np.corrcoef(data[0][:, 0], data[1])[0, 1]
                        )
                        kld_dict[mm][run_condition][idx_cond].append(
                            KLD_uvg(data[0][:, 0], data[2], data[1], data[3])
                        )
                        median_kld_dict[mm][run_condition][idx_cond].append(
                            np.nanmedian(kld_dict[mm][run_condition][idx_cond][-1])
                        )
                        mean_kld_dict[mm][run_condition][idx_cond].append(
                            np.nanmean(kld_dict[mm][run_condition][idx_cond][-1])
                        )
                        std_kld_dict[mm][run_condition][idx_cond].append(
                            np.nanstd(kld_dict[mm][run_condition][idx_cond][-1])
                        )
                        latent_mse_data_dict[mm][run_condition][idx_cond].append(
                            mse(data[0][:, 0], data[1])
                        )
                        mean_gt_dict[mm][idx_cond].append(data[1])

                        var_dict[mm][run_condition][idx_cond].append(np.mean(data[2]))
                        std_of_var_dict[mm][run_condition][idx_cond].append(
                            np.std(data[2])
                        )
                        var_gt_dict[mm][idx_cond].append(data[3])
                        full_variance[mm][run_condition][idx_cond].append(data[2])
                        full_gt_variance[mm][run_condition][idx_cond].append(data[3])

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
                        test_data[mm][run_condition][idx_cond].append(samples[0].T)

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
                        test_gt_data[mm][run_condition][idx_cond].append(samples[0].T)
                        test_latents[mm][run_condition][idx_cond].append(samples[1].T)
                        test_inputs[mm][run_condition][idx_cond].append(samples[2].T)
                        test_recons[mm][run_condition][idx_cond].append(samples[3].T)
                        test_latent_mean[mm][run_condition][idx_cond].append(
                            samples[5].T
                        )
                        test_latent_var[mm][run_condition][idx_cond].append(
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
        os.path.join(local_dir, "entire_dictionary_list_many_data_conditions.pkl"), "wb"
    ) as f:
        pickle.dump([kld_dict, mean_kld_dict, std_kld_dict, median_kld_dict], f)

else:
    with open(os.path.join(local_dir, "entire_dictionary_list.pkl"), "wb") as f:
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
    n_conditions = 3

condition_list = {i: [] for i, _ in enumerate(methods)}
os.makedirs(os.path.join(local_dir, "stored_models"), exist_ok=True)

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
        if c0 not in condition_list[i]:
            idx_cond = len(condition_list[i])
            condition_list[i].append(c0)
            print("not found... added: ", idx_cond)
            model_path = os.path.join(
                local_dir, "stored_models", f"model_cond_{c0:.2f}_" + method + ".pt"
            )
            with open(model_path, "wb") as f:
                torch.save(model, f)
        else:
            idx_cond = condition_list[i].index(c0)
            print("found... condition: ", idx_cond)
        torch.manual_seed(model.args.seed)

        model_path = os.path.join(
            local_dir, "stored_models", "model_" + str(run_id) + "_" + method + ".pt"
        )

        # store the model in the local data dictionary under model__runid__method.pt
        with open(model_path, "wb") as f:
            torch.save(model, f)

    # if condition list full for all masks, break
    if all(
        [len(condition_list[i]) == n_conditions for i, methods in enumerate(methods)]
    ):
        break

# %%

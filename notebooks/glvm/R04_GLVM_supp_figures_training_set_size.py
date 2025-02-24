#!/usr/bin/env python
# coding: utf-8

# In[2]:
################################################################
#                                                              #
#                                                              #
#          Code to generate Supplementary                      #
#                 Figs S3 and S4                               #
#          Gaussian Latent Variable Model                      #
#          with varying training dataset size                  #
#                                                              #
#                                                              #
################################################################

# pylint: disable=undefined-variable
get_ipython().run_line_magic("cd", "../../")
# pylint: enable=undefined-variable

import copy
import logging
import math
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import scipy.stats as stats
import yaml

import matplotlib
import matplotlib.pyplot as plt

# In[8]:
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import torch
from maskedvae.plotting.plotting_utils import (
    cm2inch,
    make_col_dict_new,
    make_label_dict,
)
from maskedvae.utils.utils import compute_mse, compute_rmse, return_args

logging.getLogger("matplotlib.font_manager").disabled = True
# pylint: disable=undefined-variable
get_ipython().run_line_magic("matplotlib", "inline")
# pylint: enable=undefined-variable


# folder name for evaluation
date = datetime.now().strftime("%d%m%Y_%H%M" "%S")  # more easily readable

# import warnings
# warnings.filterwarnings("ignore")
# %%


fontsize = 10
matplotlib.rcParams["font.size"] = fontsize
matplotlib.rcParams["figure.titlesize"] = fontsize
matplotlib.rcParams["legend.fontsize"] = fontsize
matplotlib.rcParams["axes.titlesize"] = fontsize
matplotlib.rcParams["axes.labelsize"] = fontsize
matplotlib.rcParams["xtick.labelsize"] = fontsize
matplotlib.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["figure.dpi"] = 300  # Set a specific DPI
plt.rcParams["savefig.dpi"] = 300


# Specify the font properties for matplotlib
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]


# In[6]:

description = "Dataset_size_fixed_decoder_different_ds_range_"
n_conditions = 4
n_masks = 5

col_dct = make_col_dict_new(task="gaussian", n_masks=n_masks)
label_dct = make_label_dict(task="gaussian", n_masks=n_masks)


# --------------------------------------
# Things to change
# which runs should be selected ignored ones(3) ... [:1]
seed_range = range(7)  # [0,1]
num_seeds = 7  # number of runs

# dataset size
ds_size = 9000  # dataset_size = dataset_size_range
dataset_size_range = [
    1,
    10,
    30,
    60,
    100,
    1000,
    5000,
    9000,
]
# adjust xticks
dataset_ticks = [10, 100, 1000, 9000]
log_xaxis = True
# --------------------------------------

# drop_dir, local_dir = generate_figure_folder(date=date,description=description)
local_dir = "./data/glvm_rev/"
drop_dir = "./data/glvm_rev/"
paper_base = "./data/glvm_rev/"
run_directory = "./runs/"
os.makedirs(local_dir + "/figures/", exist_ok=True)
drop_dir = drop_dir + description + date
os.makedirs(drop_dir + "/figures/", exist_ok=True)

# flag_decoder = ""  # "_flexible_dec"
flag_decoder = "_frozen_dec_different_ds_range"  # "_flexible_dec_different_ds_range"  #


# read in those yaml files
with open(
    os.path.join(f"./runs/summary/runs_all_obs_many_dataset_sizes{flag_decoder}.yaml"),
    "r",
) as outfile:
    runs_all_obs = yaml.load(outfile, Loader=yaml.FullLoader)

with open(
    os.path.join(f"./runs/summary/runs_masked_many_dataset_sizes{flag_decoder}.yaml"),
    "r",
) as outfile:
    runs_masked = yaml.load(outfile, Loader=yaml.FullLoader)

runs = runs_all_obs + runs_masked

methods = [
    "zero_imputation_mask_concatenated_encoder_only",
    "zero_imputation",
]


with open(
    os.path.join(local_dir, f"entire_dictionary_list_ds_first{flag_decoder}.pkl"), "rb"
) as f:
    (
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
    ) = pickle.load(f)


# adjust data indices to match the paper figures
length = len(condition_list[ds_size][0])

# Create the identity mapping
map_conditions = {i: i for i in range(length)}
noise_level = condition_list[ds_size][0]


# --------------------------------------
#  Paper Figure Supp S3
# --------------------------------------


f, axs = plt.subplots(
    length,
    n_masks - 1,
    constrained_layout=True,
    sharey="col",
    sharex=True,
    figsize=cm2inch((20, 12)),
)

for data_cond in range(length):
    for ii, idx in enumerate([1, 3]):  # range(n_masks - 1):
        for meth in methods:
            ds_sizes = dataset_size_range
            mse_values = []
            for ds_size in ds_sizes:
                # Collecting mse values for each ds_size
                mse_values.append(
                    np.sqrt(latent_mse_data_dict[ds_size][idx][meth][data_cond])
                )

            # Plot the collected points for each method
            mse_values = np.array(
                mse_values
            )  # Convert to numpy array for easier handling
            axs[data_cond, ii].plot(
                ds_sizes,
                mse_values.mean(axis=1),
                "-",
                label=f"{meth[-1]} mask {idx} noise {noise_level[data_cond]:.2f}",
                color=col_dct[meth][idx],
                alpha=0.5,
            )

            # Error bars to show the variability across the 3 seeds
            axs[data_cond, ii].errorbar(
                ds_sizes,
                mse_values.mean(axis=1),
                yerr=mse_values.std(axis=1),
                fmt="o",
                color=col_dct[meth][idx],
                ms=2,
            )
            # move axes outwards
            axs[data_cond, ii].spines["left"].set_position(("outward", 3))
            axs[data_cond, ii].spines["bottom"].set_position(("outward", 3))
            if log_xaxis:
                axs[data_cond, ii].set_xscale("log")
        axs[0, ii].set_title(f"mask {idx +1}" if idx < 3 else "all observed")
        axs[-1, ii].set_xlabel("dataset size")

    axs[data_cond, 0].set_ylabel(f"mean {noise_level[data_cond]:.2f}")


if not log_xaxis:
    axs[0, 0].set_xticks(dataset_ticks)
else:
    axs[0, 0].set_xticks(10 ** np.arange(0, 5))

for data_cond in range(length):
    for ii, idx in enumerate([1, 3]):  # range(n_masks - 1):
        for meth in methods:
            ds_sizes = dataset_size_range
            mse_values = []
            for ds_size in ds_sizes:
                # Collecting mse values for each ds_size
                mse_values.append(
                    [
                        compute_rmse(var, var_gt_dict[ds_size][idx][data_cond][0])
                        for var in var_dict[ds_size][idx][meth][data_cond]  # [1:]
                    ]
                )

            # Plot the collected points for each method
            mse_values = np.array(
                mse_values
            )  # Convert to numpy array for easier handling
            axs[data_cond, ii + 2].plot(
                ds_sizes,
                mse_values.mean(axis=1),
                "-",
                label=f"{meth[-1]} mask {idx} noise {noise_level[data_cond]:.2f}",
                color=col_dct[meth][idx],
                alpha=0.5,
            )

            # Error bars to show the variability across the 3 seeds
            axs[data_cond, ii + 2].errorbar(
                ds_sizes,
                mse_values.mean(axis=1),
                yerr=mse_values.std(axis=1),
                fmt="o",
                color=col_dct[meth][idx],
                ms=2,
            )
            # move axes outwards
            axs[data_cond, ii + 2].spines["left"].set_position(("outward", 3))
            axs[data_cond, ii + 2].spines["bottom"].set_position(("outward", 3))
            if log_xaxis:
                axs[data_cond, ii + 2].set_xscale("log")
            # Set scientific notation for y-axis
            axs[data_cond, ii + 2].yaxis.set_major_formatter(
                mticker.ScalarFormatter(useMathText=True)
            )
            axs[data_cond, ii + 2].yaxis.get_major_formatter().set_powerlimits(
                (-1, 1)
            )  # forces scientific notation

        axs[0, ii + 2].set_title(f"mask {idx +1}" if idx < 3 else "all obs")
        axs[-1, ii + 2].set_xlabel("dataset size")

    axs[data_cond, 2].set_ylabel(f"var {noise_level[data_cond]:.2f}")

axs[0, 2].set_yticks([0, 0.3])
if not log_xaxis:
    axs[0, 0].set_xticks(dataset_ticks)
else:
    axs[0, 0].set_xticks(10 ** np.arange(0, 5))
plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=0.8)
plt.savefig(
    drop_dir
    + f"/figures/Supp_Fig_dataset_size_posterior_mean_and_variance_RMSE_stacked{flag_decoder}.pdf"
)
plt.savefig(
    drop_dir
    + f"/figures/Supp_Fig_dataset_size_posterior_mean_and_variance_RMSE_stacked{flag_decoder}.png"
)


# In[8]:

# --------------------------------------
#  Paper Supp Fig S4
# --------------------------------------


for ds_size in dataset_size_range:
    f, axs = plt.subplots(
        2,
        4,
        constrained_layout=True,
        sharey=True,
        figsize=cm2inch((20, 8)),
    )
    plt.subplots_adjust(wspace=4.5)
    # axs=axs.reshape(-1)
    f.tight_layout()

    meth = "zero_imputation_mask_concatenated_encoder_only"
    # select the dataset to investigate
    data_cond = map_conditions[0]  # condition

    # sort indices by variance
    idx0 = 3
    idx1 = 2
    idx2 = 0
    idx3 = 1

    for dd, data_cond in enumerate(
        [3, 2, 1, 0]
    ):  # [0, 1, 2, 3]:  # just plot data for one dataset
        noise = noise_list[ds_size][0][data_cond][0]
        meth = "zero_imputation_mask_concatenated_encoder_only"
        data_arr = [
            var_dict[ds_size][idx0][meth][data_cond],
            var_dict[ds_size][idx1][meth][data_cond],
            var_dict[ds_size][idx2][meth][data_cond],
            var_dict[ds_size][idx3][meth][data_cond],
        ]
        gt_var = [
            var_gt_dict[ds_size][idx0][data_cond][0],
            var_gt_dict[ds_size][idx1][data_cond][0],
            var_gt_dict[ds_size][idx2][data_cond][0],
            var_gt_dict[ds_size][idx3][data_cond][0],
        ]

        plt.tight_layout(pad=0.8, w_pad=3.5, h_pad=1.8)
        sns.swarmplot(data=data_arr, color=".25", size=2, ax=axs[0, dd])  # , ax=ax1)
        sns.boxplot(
            data=data_arr, color=".8", palette="Reds", width=0.5, ax=axs[0, dd]
        )  # , ax=ax1)
        # sns.scatterplot(
        #     x=[0, 1, 2, 3], y=gt_var, s=15, color=".5", marker="s", ax=axs[data_cond, 0]
        # )  # ,label="analytical")

        sns.scatterplot(
            x=[0, 1, 2, 3],
            y=gt_var,
            s=15,
            color=".5",
            marker="s",
            ax=axs[0, dd],
            zorder=2e10,
        )

        nbins = 4
        axs[0, dd].set_xticks([0, 1, 2, 3])
        axs[0, dd].set_xticklabels(["all", "mask\n1", "mask\n2", "mask\n3"])
        axs[0, dd].locator_params(axis="x", nbins=nbins)
        axs[0, dd].locator_params(axis="y", nbins=nbins)
        axs[0, dd].set_ylabel("variance")
        axs[0, dd].set_title(f"{noise[0]} noise")
        # meth = "zero_imputation"

        plt.tight_layout(pad=0.8, w_pad=3.5, h_pad=1.8)
        meth = "zero_imputation"
        data_arr = [
            var_dict[ds_size][idx0][meth][data_cond],
            var_dict[ds_size][idx1][meth][data_cond],
            var_dict[ds_size][idx2][meth][data_cond],
            var_dict[ds_size][idx3][meth][data_cond],
        ]

        sns.swarmplot(data=data_arr, color=".25", size=2, ax=axs[1, dd])  # , ax=ax1)
        sns.boxplot(
            data=data_arr, color=".8", palette="Blues", width=0.5, ax=axs[1, dd]
        )  # , ax=ax1) ax=axs[1])#, ax=ax1)
        sns.scatterplot(
            x=[0, 1, 2, 3],
            y=gt_var,
            s=15,
            color=".5",
            marker="s",
            ax=axs[1, dd],
            # label=f"gt {noise[0]} noise",
            zorder=2e10,
        )

        axs[1, dd].set_xticks([0, 1, 2, 3])
        axs[1, dd].set_xticklabels(["all", "mask\n1", "mask\n2", "mask\n3"])
        axs[1, dd].locator_params(axis="x", nbins=nbins)
        axs[1, dd].locator_params(axis="y", nbins=nbins)
        # axs[1,data_cond].legend(frameon=True)

    plt.tight_layout()
    file_name = f"Supp_Fig_horizontal_posterior_variance_multiple_masks_dataset_condition{flag_decoder}"
    plt.savefig(
        f"{drop_dir}/figures/{file_name}_{ds_size}_{noise_level[data_cond]}.pdf"
    )
    plt.savefig(
        f"{drop_dir}/figures/{file_name}_{ds_size}_{noise_level[data_cond]}.png"
    )
    plt.savefig(
        f"{local_dir}/figures/{file_name}_{ds_size}_{noise_level[data_cond]}.pdf"
    )
    plt.savefig(
        f"{local_dir}/figures/{file_name}_{ds_size}_{noise_level[data_cond]}.png"
    )
    plt.savefig(f"{local_dir}/figures/{file_name}.svg")
    # Now each dictionary in dict_list contains only the ds_size entry


# %%

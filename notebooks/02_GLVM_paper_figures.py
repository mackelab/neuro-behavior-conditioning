#!/usr/bin/env python
# coding: utf-8

# In[2]:
################################################################
#                                                              #
#                                                              #
#          Code to generate figures of Figure 2                #
#          Gaussian Latent Variable Model                      #
#          First ensure to either run the model                #
#          and data aggregation to run this file               #
#          note that the provided data contains                #
#          more model runs with different seeds                #
#          and thus results may vary numerically               #
#          from the paper version. All qualitative             #
#          results are identical                               #
#                                                              #
#                                                              #
#                                                              #
################################################################

get_ipython().run_line_magic("cd", "../")

import copy
import logging
import math
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
import yaml

from maskedvae.plotting.plotting_utils import (
    cm2inch,
    make_col_dict,
    make_label_dict,
    create_index_mapping,
)
from maskedvae.utils.utils import return_args

logging.getLogger("matplotlib.font_manager").disabled = True

get_ipython().run_line_magic("matplotlib", "inline")


# folder name for evaluation
date = datetime.now().strftime("%d%m%Y_%H%M" "%S")  # more easily readable

# import warnings
# warnings.filterwarnings("ignore")
# %%

import matplotlib

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

description = "_Figure_2_Gaussian_LVM"
n_conditions = 3
n_masks = 5

col_dct = make_col_dict(task="gaussian", n_masks=n_masks)
label_dct = make_label_dict(task="gaussian", n_masks=n_masks)


# drop_dir, local_dir = generate_figure_folder(date=date,description=description)
local_dir = "./data/glvm/"
drop_dir = "./data/glvm/"
paper_base = "./data/glvm/"
run_directory = "./runs/"


# read in those yaml files
with open(os.path.join(drop_dir, "runs_all_obs_glvm.yaml"), "r") as outfile:
    runs_all_obs = yaml.load(outfile, Loader=yaml.FullLoader)

with open(os.path.join(drop_dir, "runs_masked_glvm.yaml"), "r") as outfile:
    runs_masked = yaml.load(outfile, Loader=yaml.FullLoader)

runs = runs_all_obs + runs_masked

methods = [
    "zero_imputation_mask_concatenated_encoder_only",
    "zero_imputation",
]


with open(os.path.join(local_dir, "entire_dictionary_list.pkl"), "rb") as f:
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
map_conditions = create_index_mapping(condition_list[0])


# In[9]:

##################################################
#                                                #
#         panel a: Gaussian z and x|z            #
#                                                #
##################################################


fig, axs = plt.subplots(5, 2, figsize=cm2inch((4, 6)), sharey=True, sharex=True)
# pad by 1 cm between top and second row
fig.subplots_adjust(hspace=1, wspace=0.8)
# plot the pdf of a standard gaussian in the middle of the first row
x = np.linspace(-6, 6, 100)
axs[2, 0].plot(x, stats.norm.pdf(x, 0, 1), label="p(z)", color="black")
# add a vertical line at 1
axs[2, 0].axvline(x=1, color="black", linestyle="--", linewidth=1)
# place the label z_0 at 1.1 at the top of the plot
axs[2, 0].text(1.3, 0.4, r"$z_0$", fontsize=fontsize)
axs[2, 0].set_xlabel("z")
axs[2, 0].set_ylabel("p(z)")
axs[2, 0].set_xlim([-5, 5])

# switch off all other first row axis
for i in range(5):
    if i != 2:
        axs[i, 0].axis("off")
axs[2, 1].axis("off")


# set off the x axis a bit to the bottom that the axes dont overlap
axs[2, 0].spines["bottom"].set_position(("outward", 5))
axs[2, 0].spines["left"].set_position(("outward", 5))


z_0 = 1
d = 1
Cs = [0.8, -0.8, 0.0, 1.4, -1.4]
sigmas = [0.6, 1.5, 0.0, 0.8, 1.1]
mus = [C * z_0 + d for C in Cs]
labels = ["x0", "x1", "x2", "xN-1", "xN"]
for i in range(5):
    if i != 2:
        axs[i, 1].plot(
            x, stats.norm.pdf(x, mus[i], sigmas[i]), label=f"p(x_{i}|z_0)", color="grey"
        )
        # axs[i,1].axvline(x=np.random.rand(mus[i], sigmas[i]), color='black', linestyle='--', linewidth=1)
        axs[i, 1].set_xlim([-5, 5])
        axs[i, 1].spines["bottom"].set_position(("outward", 5))
        axs[i, 1].spines["left"].set_position(("outward", 5))
        axs[i, 1].spines["left"].set_visible(False)
        axs[i, 1].set_yticks([])
axs[i, 1].set_xlabel(labels[i])
axs[i, 1].set_ylabel(f"p({labels[i]}|z0)", fontsize=fontsize)

plt.tight_layout()

# save the figure
file_name = f"Pan_a_vertical_schematic_latent_and_conditional_distribution"

# Save the figure
plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
plt.savefig(f"{drop_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
plt.savefig(f"{local_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.svg")

##########################################################

# In[14]:

##################################################
#                                                #
#         no panel load in models/masks          #
#                                                #
##################################################

condition_list = []  # Cx + d + noise conditions
# store models and runs for each noise and C conditions Cx + d + noise
# since the decoder are frozen = true generative model storing one model is enough
model_list = {cond: 0 for cond in range(n_conditions)}
model_runs = {cond: "0" for cond in range(n_conditions)}

for run_id, run in enumerate(runs):
    args = return_args(run)
    print(args.dir)
    # SPECIFY THE PATH TO THE RUNS
    args.fig_root = run_directory

    for i, method in enumerate(methods):
        print(i, method)

        # check if the model exists else continue
        if not os.path.exists(
            os.path.join(
                local_dir,
                "stored_models",
                "model_" + str(run_id) + "_" + method + ".pt",
            )
        ):
            print("model not found ", run, run_id, method)
            continue

        with open(
            os.path.join(
                local_dir,
                "stored_models",
                "model_" + str(run_id) + "_" + method + ".pt",
            ),
            "rb",
        ) as f:
            model = torch.load(f, map_location=torch.device("cpu"))

        # ensure one of each models is read in for each condition
        c0 = model.args.noise[0][0]
        if c0 not in condition_list:
            data_cond = len(condition_list)
            condition_list.append(c0)
            print("not found... added: ", data_cond)
            # store the models to use the fixed decoder later
            model_list[data_cond] = model
            model_runs[data_cond] = run
            # check if the decoder is frozen otherwise adjust to read in correct models
            assert (
                model_list[data_cond].args.freeze_decoder == 1
            ), "Decoder not frozen load respective models naive or masked"

        else:
            data_cond = condition_list.index(c0)
            print("found... condition: ", data_cond)
        torch.manual_seed(model.args.seed)
    if len(condition_list) == n_conditions:
        break


# plot the used masks in training next to all observed
mask = [
    np.array(model.mask_generator(torch.zeros((1, 20)), choiceval=choi))
    for choi in range(3)
]
plt.figure(figsize=cm2inch((10, 10)))
plt.imshow(np.array(mask).reshape(3, 20), cmap="Greys_r")
plt.xticks([])
plt.yticks([])
plt.savefig(drop_dir + "/figures/masks.pdf")
plt.savefig(local_dir + "/figures/masks.pdf")


# In[30]:
##################################################
#                                                #
#         panel b naive and masked               #
#         1d and 2d marginals                    #
#                                                #
##################################################


matplotlib.rcParams["path.simplify"] = True


mask_ = np.array(model.mask_generator(torch.zeros((1, 20)), choiceval=2))
# i=0
print(mask_)
seed = 0

for data_cond in [
    map_conditions[1],
]:  # just plot data for one dataset
    for mask_id in [1, 3]:  # select mask 1 and all observed
        for me, meth in enumerate(methods):
            lim = 1000
            plt.figure(figsize=cm2inch((5, 5)))
            dff = pd.concat(
                {  #'inputs': pd.DataFrame(test_inputs[mask_id][meth][data_cond][seed].T[:lim,:3]),
                    "gt": pd.DataFrame(
                        test_data[mask_id][meth][data_cond][seed][:lim, :3]
                    ),
                    "recs": pd.DataFrame(
                        test_recons[mask_id][meth][data_cond][seed].T[:lim, :3]
                    ),
                },
                names="T",
            ).reset_index(level=0)
            dff = dff.reset_index()
            dff = dff.drop(columns=["index"])
            if meth == "zero_imputation_mask_concatenated_encoder_only":
                g = sns.pairplot(
                    dff,
                    hue="T",
                    markers=".",
                    corner=True,
                    palette=dict(gt="grey", recs="darksalmon", inputs="darkred"),
                    plot_kws={"s": 20, "alpha": 0.3},
                )
            else:
                g = sns.pairplot(
                    dff,
                    hue="T",
                    markers=".",
                    corner=True,
                    palette=dict(gt="grey", recs="steelblue", inputs="midnightblue"),
                    plot_kws={"s": 20, "alpha": 0.3},
                )

            g.fig.set_figheight(cm2inch(5.5))
            g.fig.set_figwidth(cm2inch(6))
            g.set(xlim=(-5.5, 6.6))
            g.set(ylim=(-4, 6))
            g.set(ylim=(-4, 6))
            g.set(yticks=(-3, 0, 3))
            handles = g._legend_data.values()
            labels = g._legend_data.keys()
            plt.tight_layout()

            # Save the figure
            file_name = f"Pan_b_Pairplot_{meth}_mask_{mask_id}_index_seedindex_{seed}_dataset_idx_{data_cond}"
            plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
            plt.savefig(f"{drop_dir}/figures/{file_name}.png")
            plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
            plt.savefig(f"{local_dir}/figures/{file_name}.png")

        ##################################################


# In[17]:


##################################################
#                                                #
#         panel c: Gaussian posterior demo       #
#         masked works well naive fails          #
#                                                #
##################################################

import math

import scipy.stats as stats

idx_data = 129
data_cond = map_conditions[1]
x = np.linspace(0.5, 3.5, 100)
seed = 1  # dataset seed which of the 15+ runs


plt.figure(figsize=cm2inch((4, 2.5)))

for mask_id in [
    1,
]:
    count = 0

    for run_condition in [
        "zero_imputation",
        "zero_imputation_mask_concatenated_encoder_only",
    ]:

        mu = mean_dict[mask_id][run_condition][data_cond][seed][idx_data]
        variance = full_variance[mask_id][run_condition][data_cond][seed][idx_data]
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        plt.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            label=label_dct[run_condition][mask_id],
            color=col_dct[run_condition][mask_id],
        )
        count += 1

    mu = mean_gt_dict[mask_id][data_cond][seed][idx_data]
    variance = var_gt_dict[mask_id][data_cond][seed]
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    plt.plot(
        x,
        stats.norm.pdf(x, mu, sigma),
        ":",
        color="black",
        label="gt" if mask == 1 else "",
    )

mu = mean_gt_dict[3][data_cond][seed][idx_data]
variance = var_gt_dict[3][data_cond][seed]
sigma = math.sqrt(variance)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

plt.plot(
    x,
    stats.norm.pdf(x, mu, sigma),
    ":",
    color="grey",
    label="gt all" if mask == 1 else "",
)
plt.axvline(x=mean_gt_dict[1][data_cond][seed][idx_data], color="black", ymin=0.04)

plt.xlabel("inferred latent")
plt.ylabel("pdf")
plt.legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5))
# set only lower bound on spines of x axis
plt.gca().spines.left.set_bounds(0, max(plt.gca().get_ylim()))
print(plt.gca().get_xlim())
plt.gca().spines.bottom.set_bounds(
    min(plt.gca().get_xlim()) + 0.1, max(plt.gca().get_xlim())
)


file_name = (
    f"Pan_c_posterior_mean_of_sample_x_{idx_data}_cond_{data_cond}_masked_smaller"
)

# Save the figure
plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
plt.savefig(f"{drop_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
plt.savefig(f"{local_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.svg")

##################################################
# In[18]:
#################################################
#                                                #
#         panel d : posterior mean               #
#         posterior mean all x with one mask     #
#                                                #
##################################################
matplotlib.rcParams["path.simplify"] = True

fig, axs = plt.subplots(1, 1, figsize=cm2inch((4, 4)), sharey=True, sharex=True)
meth = "zero_imputation_mask_concatenated_encoder_only"
mask_id = 1
seed = 1  # dataset seed which of the 15+ runs
data_cond = map_conditions[1]  # condition

fig.tight_layout()
axs.plot([-3, 3], [-3, 3], "black", linewidth=1)

for meth in methods:
    axs.plot(
        mean_gt_dict[mask_id][data_cond][seed][:300],
        mean_dict[mask_id][meth][data_cond][seed][:300, 0],
        "o",
        color=col_dct[meth][mask_id],
        ms=0.9,
        alpha=0.3,
    )
axs.set(aspect="equal")
axs.set_xlabel("true mean")
axs.set_ylabel("predicted")

plt.yticks([-2, 0, 2])
plt.xticks([-2, 0, 2])
# offset the axis
axs.spines["left"].set_position(("outward", 4))
axs.spines["bottom"].set_position(("outward", 4))


plt.tight_layout()
file_name = f"Pan_d_posterior_mean"
plt.savefig(drop_dir + f"/figures/{file_name}.pdf")
plt.savefig(drop_dir + f"/figures/{file_name}.png")
plt.savefig(local_dir + f"/figures/{file_name}.pdf")
plt.savefig(local_dir + f"/figures/{file_name}.png")
plt.savefig(local_dir + f"/figures/{file_name}.svg")


# In[19]:
##################################################
#                                                #
#   panel e and f:                               #
#   posterior variance (e) and posterior kl (f)  #
#                                                #
##################################################


# for each data condition and for each masking cconditiong
# we take an average over the posteior kl divergence in the test set


# read in multiple datasets with different conditions Cs and sigmas 10 different one seed each
# same masks as other datasets
with open(
    os.path.join(local_dir, "entire_dictionary_list_many_data_conditions.pkl"), "rb"
) as f:
    (
        kld_dict,
        mean_kld_dict,
        std_kld_dict,
        median_kld_dict,
    ) = pickle.load(f)


median_masked_mask = []
median_masked_all = []
median_naive_mask = []
median_naive_all = []
mean_masked_mask = []
mean_masked_all = []
mean_naive_mask = []
mean_naive_all = []
for ds in range(10):
    # median kld [mask_id][maskcondition][dataset condition][seed]
    median_naive_all.append(median_kld_dict[3]["zero_imputation"][ds][0])
    median_masked_all.append(
        median_kld_dict[3]["zero_imputation_mask_concatenated_encoder_only"][ds][0]
    )
    mean_naive_all.append(mean_kld_dict[3]["zero_imputation"][ds][0])
    mean_masked_all.append(
        mean_kld_dict[3]["zero_imputation_mask_concatenated_encoder_only"][ds][0]
    )

    # compute the aggreate over the different dataset conditions (Cs and sigmas) and different masks
    for mask_id in [
        0,
        1,
        2,
    ]:  # 0,1,2,]:
        median_masked_mask.append(
            median_kld_dict[mask_id]["zero_imputation_mask_concatenated_encoder_only"][
                ds
            ][0]
        )
        median_naive_mask.append(median_kld_dict[mask_id]["zero_imputation"][ds][0])
        mean_masked_mask.append(
            mean_kld_dict[mask_id]["zero_imputation_mask_concatenated_encoder_only"][
                ds
            ][0]
        )
        mean_naive_mask.append(mean_kld_dict[mask_id]["zero_imputation"][ds][0])


f, axs = plt.subplots(
    1, 2, constrained_layout=True, sharey=False, figsize=cm2inch((5.5, 3.5))
)
plt.subplots_adjust(wspace=4.5)
# axs=axs.reshape(-1)
f.tight_layout()
# Create an array with the colors you want to use
colors = ["lightsteelblue", "salmon", "steelblue", "darkred"]
# Set your custom color palette
customPalette = sns.set_palette(sns.color_palette(colors), desat=1)

mask_id = 1

# index 3 is all observed
data_arr = [
    var_dict[mask_id]["zero_imputation"][data_cond],
    var_dict[mask_id]["zero_imputation_mask_concatenated_encoder_only"][data_cond],
]
axs[0].axhline(
    var_gt_dict[mask_id][data_cond][0],
    color="black",
    label="gt masked",
    linestyle=":",
    zorder=0,
)

sns.swarmplot(data=data_arr, color=".25", size=1, ax=axs[0])
sns.boxplot(
    data=data_arr, palette=customPalette, width=0.6, ax=axs[0], showfliers=False
)


median = False  # plot median or mean of the kld
if median:
    aggregate_tag = "median"
    data_arr_kld = [median_naive_mask, median_masked_mask]
else:
    aggregate_tag = "mean"
    data_arr_kld = [mean_naive_mask, mean_masked_mask]

sns.swarmplot(
    data=data_arr_kld, color=".25", size=1, ax=axs[1]
)  # , ax=ax1), labels=["dataset/mask"]
sns.boxplot(
    data=data_arr_kld, palette=customPalette, width=0.6, ax=axs[1], showfliers=False
)

nbins = 4

axs[0].set_ylabel("var")
axs[1].set_ylabel("KLD")


for ax in axs:
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))
    ax.locator_params(axis="x", nbins=nbins)
    ax.locator_params(axis="y", nbins=nbins)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["naive", "masked"])
    # turn the labels by 30 degree
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

axs[0].set_yticks([0.02, 0.03, 0.04])
file_name = f"Panel_e_f_posterior_variance_{data_cond}_and_KLD_many_datasets"
plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
plt.savefig(f"{drop_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
plt.savefig(f"{local_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.svg")

# In[20]:


#################################################
#                                                #
#         panel e : conditional                  #
#         conditional histogram                   #
#                                                #
##################################################


# set seed for numpy and torch
np.random.seed(0)
torch.manual_seed(0)

# sample multiple times from the posterior
# and plot the histogram of the conditional
# distribution

idx_data = 129

count = 0
plt.figure(figsize=cm2inch((4, 2.5)))
x = np.linspace(0.5, 3.5, 100)
recons_samples = {meth: [] for meth in methods}
for mask_id in [
    1,
]:
    count = 0

    for meth in methods:

        mu = mean_dict[mask_id][meth][data_cond][0][idx_data]
        variance = full_variance[mask_id][meth][data_cond][0][idx_data]
        sigma = math.sqrt(variance)
        z_method = np.random.normal(mu, sigma, 1000).reshape(1000, 1)
        z_method = torch.from_numpy(z_method).float()
        # since the decoder is fixed we can load any of the models (here masked models for the different noise and C values)
        assert (
            model_list[data_cond].args.freeze_decoder == 1
        ), "Decoder not frozen load respective models naive or masked"
        recon_batch, recon_var = model_list[data_cond].vae.decoder(z_method, m=None)
        n_samples = 20  # number of samples to plot
        recon_batch_r = np.repeat(recon_batch.cpu().data.numpy(), n_samples, axis=0)
        recon_sigma_r = np.sqrt(
            np.repeat(recon_var.cpu().data.numpy(), n_samples, axis=0)
        )
        data_recons = np.random.normal(
            loc=recon_batch_r.flatten(), scale=recon_sigma_r.flatten()
        )
        data_recons = data_recons.reshape(-1, model.args.x_dim)

        recons_samples[meth] = data_recons
        # model
    mu = mean_gt_dict[mask_id][data_cond][0][idx_data]
    variance = var_gt_dict[mask_id][data_cond][0]
    sigma = math.sqrt(variance)

    # sample multiple times from the posterior

    z_gt = np.random.normal(mu, sigma, 1000).reshape(1000, 1)
    z_gt = torch.from_numpy(z_gt).float()
    recon_batch, recon_var = model_list[data_cond].vae.decoder(z_gt, m=None)

    n_samples = 20  # number of samples to plot
    recon_batch_r = np.repeat(recon_batch.cpu().data.numpy(), n_samples, axis=0)
    recon_sigma_r = np.sqrt(np.repeat(recon_var.cpu().data.numpy(), n_samples, axis=0))
    data_recons = np.random.normal(
        loc=recon_batch_r.flatten(), scale=recon_sigma_r.flatten()
    )
    data_recons = data_recons.reshape(-1, model.args.x_dim)


label_num_dct = make_label_dict(task="gaussian", n_masks=n_masks, mask_nr=True)
col_dct = make_col_dict(task="gaussian", n_masks=n_masks)

fig, axs = plt.subplots(2, 2, sharey=False, figsize=cm2inch((6, 4)))

axs = axs.reshape(-1)
fig.tight_layout()


str_dat = " "

masked_indices = np.where(
    model_list[data_cond].mask_generator(torch.zeros((1, 20)), choiceval=1) == 0
)[1]
# set seed for numpy and torch
np.random.seed(0)
torch.manual_seed(0)
choices = np.random.choice(masked_indices, 4, replace=False)
choices.sort()
for id, idx_dat in enumerate(choices):
    str_dat = str_dat + f"{idx_dat} "
    for meth in methods:
        axs[id].hist(
            recons_samples[meth][:, idx_dat],
            bins=20,
            density=True,
            alpha=0.5,
            label=label_dct[meth][mask_id] if mask == 1 else "",
            color=col_dct[meth][mask_id],
        )
    sns.kdeplot(
        data_recons[:, idx_dat],
        color="black",
        label=f"gt {str_dat}" if mask == 1 else "",
        alpha=1,
        lw=1.5,
        ax=axs[id],
    )
    axs[id].spines["left"].set_position(("outward", 4))
    axs[id].spines["bottom"].set_position(("outward", 4))
    # axs[idx_dat].set_xlabel(f"x{idx_dat}")
    axs[id].set_ylabel("")

    # get maximum value on y axis
    y_max = axs[id].get_ylim()[1]
    if y_max < 0.5:
        axs[id].set_yticks([0, 0.2])
    # axs[id].set_yticks([0,0.2])
axs[0].set_ylabel("pdf")
plt.legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5))


file_name = f"Pan_e_multiple_conditionals_{idx_data}_cond_{data_cond}_{str_dat.replace(' ', '_')}_masked_smaller"
plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
plt.savefig(f"{drop_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
plt.savefig(f"{local_dir}/figures/{file_name}.png")


# In[21]:

#################################################
#                                                #
#         panel h calibration                    #
#         calibration metrics                    #
#                                                #
##################################################
import pickle

with open(f"{local_dir}/calibration.pkl", "rb") as f:
    calibration = pickle.load(f)
with open(f"{local_dir}/calibration_obs.pkl", "rb") as f:
    calibration_obs = pickle.load(f)
with open(f"{local_dir}/calibration_masked.pkl", "rb") as f:
    calibration_masked = pickle.load(f)


fig, axs = plt.subplots(1, 1, figsize=cm2inch((5, 3.5)), sharey=True)
meth = "zero_imputation_mask_concatenated_encoder_only"
# axi,mask_id = 1,3
label = [95, 90, 80, 60]
axs.plot(label, [lab / 100 for lab in label], "black")

for data_cond in [
    map_conditions[1],  # condition,
]:  # just plot data for one dataset
    for mask_id in [
        1,
    ]:
        for percentile in label:
            for q in range(len(calibration_obs[percentile][mask_id][meth][0])):
                meth = "zero_imputation_mask_concatenated_encoder_only"
                axs.plot(
                    percentile
                    * np.ones_like(
                        calibration_masked[percentile][mask_id][meth][data_cond][q]
                    )
                    - 1
                    + 0.9 * np.random.uniform(),
                    calibration_masked[percentile][mask_id][meth][data_cond][q],
                    ".",
                    ms=1,
                    alpha=0.3,
                    color=col_dct[meth][1],
                    label="masked"
                    # if percentile == 95 and q == 0 and mask_id == 1 and data_cond == 1
                    # else "",
                )
                meth = "zero_imputation"
                axs.plot(
                    percentile
                    * np.ones_like(
                        calibration_masked[percentile][mask_id][meth][data_cond][q]
                    )
                    + 1 * 0.9 * np.random.uniform(),
                    calibration_masked[percentile][mask_id][meth][data_cond][q],
                    ".",
                    ms=1,
                    alpha=0.3,
                    color=col_dct[meth][1],
                    label="naive",
                    # if percentile == 95 and q == 0 and mask_id == 1 and data_cond == 1
                    # else "",
                )
axs.set_xlabel("percentile")
axs.set_xticks(label)
axs.set_xticklabels([str(lab) + r"$^{\mathrm{th}}$" for lab in label])

axs.set_ylabel(r"% in n$^{\mathrm{th}}$ perc.")
# ensure unique labels in legend
handles, labels = axs.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# axs.legend(by_label.values(), by_label.keys(), loc="upper left")

axs.legend(
    by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1.0), frameon=False
)
axs.set_yticks([0.4, 0.6, 0.8, 1.0])
axs.set_yticklabels([40, 60, 80, 100])

# offset the axis
axs.spines["left"].set_position(("outward", 4))
axs.spines["bottom"].set_position(("outward", 4))

plt.tight_layout()
file_name = f"Pan_i_calibration"
plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
plt.savefig(f"{drop_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
plt.savefig(f"{local_dir}/figures/{file_name}.png")


# In[31]:
##################################################
#                                                #
#         Supp 1: civariance matrices            #
#         masked close to gt naive off           #
#                                                #
##################################################

col_map = "viridis"
meth = "zero_imputation_mask_concatenated_encoder_only"

for data_cond in [map_conditions[1]]:  # condition]:  # data condition
    for mask_id in [1, 3]:  # masks 1 and 3 (all observed)

        real_cov_x = np.cov(test_data[mask_id][meth][data_cond][seed][:lim, :].T)
        masked_cov_x = np.cov(test_recons[mask_id][meth][data_cond][seed].T[:lim, :].T)
        naive_cov_x = np.cov(
            test_recons[mask_id]["zero_imputation"][data_cond][seed].T[:lim, :].T
        )

        cov_x = np.concatenate((real_cov_x, masked_cov_x, naive_cov_x), axis=1)

        vmin_x = np.min(cov_x)
        vmax_x = np.max(cov_x)

        fig, ax = plt.subplots(1, 3, figsize=cm2inch((8, 3)), sharey=True)
        cbar_ax = fig.add_axes([0.99, 0.2, 0.02, 0.7])

        heatmap_kwargs = {
            "annot": False,
            "square": True,
            "cbar": True,
            "cbar_kws": {"shrink": 0.74},
            "annot_kws": {"size": 16},
            "vmin": vmin_x,
            "vmax": vmax_x,
            "cmap": col_map,
            "cbar_ax": cbar_ax,
        }
        # remove axis labels and ticks

        sns.heatmap(real_cov_x, ax=ax[0], **heatmap_kwargs)
        sns.heatmap(masked_cov_x, ax=ax[1], **heatmap_kwargs)
        sns.heatmap(naive_cov_x, ax=ax[2], **heatmap_kwargs)

        cbar_ax.set_ylabel("cov")
        # set the colorbar tick range from vmin to vmax and include 0
        cbar_ax.set_yticks([-1, 1, 3])
        titles = ["data", "masked", "naive"]
        for i in range(3):
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(titles[i])  # set titles

        # set titles

        plt.tight_layout()

        file_name = f"Supp_Fig_covariance_gt_vs_masked_vs_naive_mask_{mask_id}_ds_data_condition_{data_cond}_{col_map}"

        # Save the figure
        plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
        plt.savefig(f"{drop_dir}/figures/{file_name}.png")
        plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
        plt.savefig(f"{local_dir}/figures/{file_name}.png")
        plt.savefig(f"{local_dir}/figures/{file_name}.svg")

        ##################################################

# In[259]:
#################################################
#                                                #
#         supplementary figure                   #
#         posterior var for many                 #
#                                                #
##################################################

import seaborn as sns

f, axs = plt.subplots(
    3, 2, constrained_layout=True, sharey=True, figsize=cm2inch((10, 15))
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

for data_cond in [0, 1, 2]:  # just plot data for one dataset
    meth = "zero_imputation_mask_concatenated_encoder_only"
    data_arr = [
        var_dict[idx0][meth][data_cond],
        var_dict[idx1][meth][data_cond],
        var_dict[idx2][meth][data_cond],
        var_dict[idx3][meth][data_cond],
    ]
    gt_var = [
        var_gt_dict[idx0][data_cond][0],
        var_gt_dict[idx1][data_cond][0],
        var_gt_dict[idx2][data_cond][0],
        var_gt_dict[idx3][data_cond][0],
    ]

    plt.tight_layout(pad=0.8, w_pad=3.5, h_pad=1.8)
    sns.swarmplot(data=data_arr, color=".25", size=2, ax=axs[data_cond, 0])  # , ax=ax1)
    sns.boxplot(
        data=data_arr, color=".8", palette="Reds", width=0.5, ax=axs[data_cond, 0]
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
        ax=axs[data_cond, 0],
        zorder=2e10,
    )

    nbins = 4
    axs[data_cond, 0].set_xticks([0, 1, 2, 3])
    axs[data_cond, 0].set_xticklabels(["all", "mask\n1", "mask\n2", "mask\n3"])
    axs[data_cond, 0].locator_params(axis="x", nbins=nbins)
    axs[data_cond, 0].locator_params(axis="y", nbins=nbins)

    # meth = "zero_imputation"

    plt.tight_layout(pad=0.8, w_pad=3.5, h_pad=1.8)
    meth = "zero_imputation"
    data_arr = [
        var_dict[idx0][meth][data_cond],
        var_dict[idx1][meth][data_cond],
        var_dict[idx2][meth][data_cond],
        var_dict[idx3][meth][data_cond],
    ]

    sns.swarmplot(data=data_arr, color=".25", size=2, ax=axs[data_cond, 1])  # , ax=ax1)
    sns.boxplot(
        data=data_arr, color=".8", palette="Blues", width=0.5, ax=axs[data_cond, 1]
    )  # , ax=ax1) ax=axs[1])#, ax=ax1)
    sns.scatterplot(
        x=[0, 1, 2, 3],
        y=gt_var,
        s=15,
        color=".5",
        marker="s",
        ax=axs[data_cond, 1],
        label="ground truth",
    )

    axs[data_cond, 1].set_xticks([0, 1, 2, 3])
    axs[data_cond, 1].set_xticklabels(["all", "mask\n1", "mask\n2", "mask\n3"])
    axs[data_cond, 1].locator_params(axis="x", nbins=nbins)
    axs[data_cond, 1].locator_params(axis="y", nbins=nbins)
    axs[data_cond, 1].legend(frameon=True)

plt.tight_layout()
file_name = f"Supp_Fig_posterior_variance_multiple_masks_dataset_condition"
plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
plt.savefig(f"{drop_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
plt.savefig(f"{local_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.svg")

###########################################

# In[20]:

# make 2 x 2 plot of the conditional distributions


# In[21]:
# get the data means
dicts = {}
for i in range(n_conditions):
    dicts[i] = []
reconstructed_test_means = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}
SE_test_means = {
    i: {method: copy.deepcopy(dicts) for method in methods} for i in range(n_masks)
}

for data_cond in [
    map_conditions[1],
]:  # just plot data for one dataset
    for mask_id in [
        1,
    ]:  # select mask 1 and all observed
        for me, meth in enumerate(methods):
            mu = mean_dict[mask_id][meth][data_cond][0]
            variance = full_variance[mask_id][meth][data_cond][0]
            # take the square root of the variance
            sigma = np.sqrt(variance).reshape(-1, 1)
            # sigma = math.sqrt(variance)
            z_method = np.random.normal(mu, sigma)
            z_method = torch.from_numpy(z_method).float()
            # since the decoder is fixed we can load any of the models (here masked models for the different noise and C values)
            assert (
                model_list[data_cond].args.freeze_decoder == 1
            ), "Decoder not frozen load respective models naive or masked"
            recon_batch, recon_var = model_list[data_cond].vae.decoder(z_method, m=None)
            print(recon_batch.shape, sigma.shape, mu.shape)

            # only consider one seed for now
            reconstructed_test_means[mask_id][meth][
                data_cond
            ] = recon_batch.detach().numpy()
            SE_test_means[mask_id][meth][data_cond] = (
                recon_batch.detach().numpy() - test_data[mask_id][meth][data_cond][0]
            ) ** 2

# In[21]:
#################################################
#                                                #
#         panel f : MSE imputation               #
#                                                #
##################################################


fig, axs = plt.subplots(2, 2, sharey=False, sharex=True, figsize=cm2inch((5, 4)))

# plt.subplots_adjust(wspace=4.5)
axs = axs.reshape(-1)
fig.tight_layout()

for id, idx_dat in enumerate(choices):
    str_dat = str_dat + f"{idx_dat} "
    for meth in methods:
        if meth == "zero_imputation":
            offset = 1
        else:
            offset = 0
        axs[id].plot(
            offset + 0.04 * np.random.randn(1000),
            SE_test_means[mask_id][meth][data_cond][:, idx_dat],
            "o",
            ms=0.1,
            alpha=0.3,
            label=label_dct[meth][mask_id] if mask == 1 else "",
            color=col_dct[meth][mask_id],
        )
        # plot a horizontal line at the mean
        axs[id].axhline(
            y=np.mean(SE_test_means[mask_id][meth][data_cond][:, idx_dat]),
            color=col_dct[meth][mask_id],
            lw=0.5,
            ls=":",
        )
    axs[id].spines["left"].set_position(("outward", 4))
    axs[id].spines["bottom"].set_position(("outward", 4))
    axs[id].set_ylabel("")
    axs[id].set_xticks([0, 1])
    axs[id].set_xticklabels(["masked", "naive"])


axs[0].set_ylabel("Squared\nerror")

plt.legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5))


file_name = f"Supp_Fig_MSE_{idx_data}_cond_{data_cond}_{str_dat.replace(' ', '_')}_masked_smaller"
plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
plt.savefig(f"{drop_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
plt.savefig(f"{local_dir}/figures/{file_name}.png")


# In[203]:
#################################################
#                                                #
#         supplementary figure                   #
#         posterior mean MSE                     #
#                                                #
##################################################

import seaborn as sns

f, axs = plt.subplots(
    1, 2, constrained_layout=True, sharey=False, figsize=cm2inch((10, 5))
)
plt.subplots_adjust(wspace=4.5)
f.tight_layout()

data_cond = map_conditions[1]


meth = "zero_imputation_mask_concatenated_encoder_only"
data_arr = [
    latent_mse_data_dict[idx0][meth][data_cond],
    latent_mse_data_dict[idx1][meth][data_cond],
    latent_mse_data_dict[idx2][meth][data_cond],
    latent_mse_data_dict[idx3][meth][data_cond],
]


plt.tight_layout(pad=0.8, w_pad=3.5, h_pad=1.8)
sns.swarmplot(data=data_arr, color=".25", size=2, ax=axs[0])  # , ax=ax1)
sns.boxplot(
    data=data_arr, color=".8", palette="Reds", width=0.5, ax=axs[0]
)  # , ax=ax1)

nbins = 4
axs[0].set_xticks([0, 1, 2, 3])
axs[0].set_xticklabels(["all", "mask\n1", "mask\n2", "mask\n3"])
axs[0].set_ylabel("MSE gt and predicted\nposterior mean")
axs[0].locator_params(axis="x", nbins=nbins)
axs[0].locator_params(axis="y", nbins=nbins)


# meth = "zero_imputation"

data_cond = map_conditions[1]

plt.tight_layout(pad=0.8, w_pad=3.5, h_pad=1.8)
meth = "zero_imputation"
data_arr = [
    latent_mse_data_dict[idx0][meth][data_cond],
    latent_mse_data_dict[idx1][meth][data_cond],
    latent_mse_data_dict[idx2][meth][data_cond],
    latent_mse_data_dict[idx3][meth][data_cond],
]

sns.swarmplot(data=data_arr, color=".25", size=2, ax=axs[1])  # , ax=ax1)
sns.boxplot(
    data=data_arr, color=".8", palette="Blues", width=0.5, ax=axs[1]
)  # , ax=ax1) ax=axs[1])#, ax=ax1)


axs[1].set_xticks([0, 1, 2, 3])
axs[1].set_xticklabels(["all", "mask\n1", "mask\n2", "mask\n3"])
# axs[1].set_ylabel("posterior\n variance")
axs[1].locator_params(axis="x", nbins=nbins)
axs[1].locator_params(axis="y", nbins=nbins)
axs[1].legend()
plt.tight_layout()
file_name = f"Supp_Fig_method_comparison_posterior_mean_MSE"
plt.savefig(drop_dir + f"/figures/{file_name}.pdf")
plt.savefig(drop_dir + f"/figures/{file_name}.png")
plt.savefig(local_dir + f"/figures/{file_name}.pdf")
plt.savefig(local_dir + f"/figures/{file_name}.png")

#################################################


# In[204]:


import seaborn as sns

f, axs = plt.subplots(
    2, 1, constrained_layout=True, sharey=False, sharex=True, figsize=cm2inch((7, 7))
)
plt.subplots_adjust(wspace=4.5)
# axs=axs.reshape(-1)
f.tight_layout()

# condition dataset
data_cond = map_conditions[0]


meth = "zero_imputation_mask_concatenated_encoder_only"
data_arr = [
    latent_mse_data_dict[idx0][meth][data_cond],
    latent_mse_data_dict[idx1][meth][data_cond],
    latent_mse_data_dict[idx2][meth][data_cond],
    latent_mse_data_dict[idx3][meth][data_cond],
]


plt.tight_layout(pad=0.8, w_pad=3.5, h_pad=1.8)
sns.swarmplot(data=data_arr, color=".25", size=2, ax=axs[0])  # , ax=ax1)
sns.boxplot(
    data=data_arr, color=".8", palette="Reds", width=0.5, ax=axs[0]
)  # , ax=ax1)

nbins = 4
axs[0].set_xticks([0, 1, 2, 3])
axs[0].set_xticklabels(["all", "mask\n1", "mask\n2", "mask\n3"])
axs[0].set_ylabel("MSE")  # gt and predicted\nposterior mean")
axs[0].locator_params(axis="x", nbins=nbins)
axs[0].locator_params(axis="y", nbins=nbins)


# meth = "zero_imputation"

data_cond = map_conditions[1]

plt.tight_layout(pad=0.8, w_pad=3.5, h_pad=1.8)
meth = "zero_imputation"
data_arr = [
    latent_mse_data_dict[idx0][meth][data_cond],
    latent_mse_data_dict[idx1][meth][data_cond],
    latent_mse_data_dict[idx2][meth][data_cond],
    latent_mse_data_dict[idx3][meth][data_cond],
]

sns.swarmplot(data=data_arr, color=".25", size=2, ax=axs[1])  # , ax=ax1)
sns.boxplot(
    data=data_arr, color=".8", palette="Blues", width=0.5, ax=axs[1]
)  # , ax=ax1) ax=axs[1])#, ax=ax1)


axs[1].set_xticks([0, 1, 2, 3])
axs[1].set_xticklabels(["all", "mask\n1", "mask\n2", "mask\n3"])
# axs[1].set_ylabel("posterior\n variance")
axs[1].locator_params(axis="x", nbins=nbins)
axs[1].locator_params(axis="y", nbins=nbins)
axs[1].legend()
file_name = f"Supp_Fig_method_comparison_posterior_mean_MSE_stacked"
plt.savefig(drop_dir + f"/figures/{file_name}.pdf")
plt.savefig(drop_dir + f"/figures/{file_name}.png")
plt.savefig(local_dir + f"/figures/{file_name}.pdf")
plt.savefig(local_dir + f"/figures/{file_name}.png")
plt.savefig(local_dir + f"/figures/{file_name}.svg")


# In[34]:

##################################################
#                                                #
#         Calculate Calibration                  #
#                                                #
##################################################

mask_id = 1  # n_masks):
mask_ = np.array(model.mask_generator(torch.zeros((1, 20)), choiceval=mask_id))
meth = "zero_imputation_mask_concatenated_encoder_only"
i = 0
data_cond = map_conditions[1]
print(mask_)
q = 0
n_samples = 128

generate_calibration = False  # Caution: this will run for a few minutes
if generate_calibration:
    dicts = {}
    for i in range(n_conditions):
        dicts[i] = []
    calibration = {
        calib: {
            i: {method: copy.deepcopy(dicts) for method in methods}
            for i in range(n_masks)
        }
        for calib in [95, 90, 80, 60]
    }
    calibration_obs = {
        calib: {
            i: {method: copy.deepcopy(dicts) for method in methods}
            for i in range(n_masks)
        }
        for calib in [95, 90, 80, 60]
    }
    calibration_masked = {
        calib: {
            i: {method: copy.deepcopy(dicts) for method in methods}
            for i in range(n_masks)
        }
        for calib in [95, 90, 80, 60]
    }

    for q in range(len(mean_dict[mask_id][meth][data_cond])):
        print(mean_dict[0][meth][0][q][:10])
        for data_cond in range(n_conditions):
            for mask_id in [0, 1, 2, 3]:  # range(n_masks[]):
                for meth in methods:
                    lat_mean = mean_dict[mask_id][meth][data_cond][
                        q
                    ].flatten()  # np.repeat(x_sample.cpu().data.numpy(), n_samples, axis=0)
                    lat_sigma = np.sqrt(full_variance[mask_id][meth][data_cond][q])
                    batch_size = len(lat_mean)
                    # sample each value n_sample times
                    lat_mean = np.repeat(lat_mean, n_samples, axis=0)
                    lat_sigma = np.repeat(lat_sigma, n_samples, axis=0)

                    # sample mean and standard deviation passed as one vector
                    approx_post_samples = np.random.normal(
                        loc=lat_mean, scale=lat_sigma
                    )
                    # pass through generative model x = Cz+d
                    recon_batch = (
                        C_list[mask_id][data_cond] * approx_post_samples
                        + d_list[mask_id][data_cond]
                    )

                    # for the fixed generative model retrieve the correct observation noise std
                    recon_var = noise_list[mask_id][
                        data_cond
                    ]  # noise_list[mask_id][data_cond]*noise_list[mask_id][data_cond]
                    # repeat the noise term batch * n_samples times
                    recon_var = np.repeat(recon_var, recon_batch.shape[1], axis=1)
                    # sample from the observation model for each variable - flatten for faster sampling
                    data_recons = np.random.normal(
                        loc=recon_batch.flatten(), scale=recon_var.flatten()
                    )
                    # bring into the correct shape again: data dim, batch_size, n_samples
                    data_recons = data_recons.reshape(
                        model.args.x_dim, batch_size, n_samples
                    )  # model.args.x_dim,n_samples,-1)
                    print("\n data_recons \n", data_recons.shape)

                    lower = [2.5, 5, 10, 20]
                    upper = [97.5, 95, 90, 80]
                    label = [95, 90, 80, 60]

                    recon_batch = data_recons.T
                    mid_per = np.percentile(recon_batch, 50, axis=0)
                    batch_original = test_data[mask_id][meth][data_cond][q]

                    mask_ = np.array(
                        model.mask_generator(torch.zeros((1, 20)), choiceval=mask_id)
                    )
                    if mask_id == 3:
                        mask_ = np.ones_like(mask_)
                    mask_ = np.repeat(mask_, 1000, axis=0)

                    # adjust mask
                    for i, (l, u) in enumerate(zip(lower, upper)):
                        lower_per = np.percentile(recon_batch, l, axis=0)
                        upper_per = np.percentile(recon_batch, u, axis=0)
                        hits = (batch_original >= lower_per) & (
                            batch_original <= upper_per
                        )

                        summed_hits_obs = np.sum(
                            np.sum(mask_ * hits, axis=0), axis=0
                        ) / np.sum(mask_)
                        summed_hits_masked = np.sum(
                            np.sum((1 - mask_) * hits, axis=0), axis=0
                        ) / np.sum(1 - mask_)
                        summed_hits_total = np.sum(np.sum(hits, axis=0), axis=0) / (
                            hits.shape[0] * hits.shape[1]
                        )
                        print("obs ", label[i], summed_hits_obs)
                        print("masked ", label[i], summed_hits_masked)
                        print("total ", label[i], summed_hits_total)

                        summed_hits_obs = np.sum(mask_ * hits, axis=0) / np.sum(
                            mask_, axis=0
                        )
                        summed_hits_masked = np.sum(
                            (1 - mask_) * hits, axis=0
                        ) / np.sum(1 - mask_, axis=0)
                        summed_hits_total = np.sum(hits, axis=0) / hits.shape[0]

                        calibration[label[i]][mask_id][meth][data_cond].append(
                            summed_hits_total
                        )
                        calibration_obs[label[i]][mask_id][meth][data_cond].append(
                            summed_hits_obs
                        )
                        calibration_masked[label[i]][mask_id][meth][data_cond].append(
                            summed_hits_masked
                        )

    # store calibration data
    import pickle

    with open(f"{local_dir}/data/calibration.pkl", "wb") as f:
        pickle.dump(calibration, f)
    with open(f"{local_dir}/data/calibration_obs.pkl", "wb") as f:
        pickle.dump(calibration_obs, f)
    with open(f"{local_dir}/data/calibration_masked.pkl", "wb") as f:
        pickle.dump(calibration_masked, f)


##################################################
#                                                #
#         no panel demo different conditions     #
#                                                #
##################################################

mask_cond = ["miss " + str(i) for i in range(n_masks - 2)]
mask_cond.append("all obs")
mask_cond.append("new")
label_num_dct = make_label_dict(task="gaussian", n_masks=n_masks, mask_nr=True)
col_dct = make_col_dict(task="gaussian", n_masks=n_masks)
print(label_num_dct)

import seaborn as sns

f, axs = plt.subplots(
    1, 1, constrained_layout=True, figsize=cm2inch((10, 7)), sharey=True
)
plt.subplots_adjust(wspace=4.5)
f.tight_layout()

varlist = []
for i in range(n_conditions):
    varlist.append(var_gt_dict[0][i][0])
vararr = np.array(varlist)
idx = np.argsort(vararr)


def x_axis_placing(n_conditions=n_conditions, n_methods=3, shift=0.25, idx=None):
    centres = np.arange(1, n_conditions + 1)
    if idx is not None:
        centres = centres[idx]
    return centres, centres - shift, centres + shift, shift / 10


centres, centm, centp, rand_shift = x_axis_placing(idx=None)

centres_method = [centres, centm, centp]
for mask_id in range(n_masks - 1):
    for kk, data_cond in enumerate(idx):
        for i, meth in enumerate(methods):
            # np.random.seed(2)
            axs.errorbar(
                centres_method[i][kk]
                + rand_shift
                * np.random.randn(len(std_of_var_dict[mask_id][meth][data_cond])),
                var_dict[mask_id][meth][data_cond],
                yerr=std_of_var_dict[mask_id][meth][data_cond],
                fmt="o",
                ms=1,
                label=label_num_dct[meth][mask_id] if data_cond == 0 else "",
                color=col_dct[meth][mask_id],
            )
            if i == 0:
                axs.plot(
                    centres_method[i][kk] - 0.15,
                    var_gt_dict[mask_id][data_cond][0],
                    "*",
                    ms=3,
                    color=col_dct[meth][mask_id],
                    label="gt " + mask_cond[mask_id] if data_cond == 0 else "",
                    zorder=2e10,
                )

            axs.locator_params(nbins=4)
            axs.set_xlabel("condition")
            axs.set_ylabel("posterior variance")
            # place legend outside the
            axs.legend(ncol=3, loc="best", bbox_to_anchor=(0.5, 1))

            axs.set_xticks(
                [
                    1,
                    2,
                    3,
                ]
            )

plt.xlim([0, 4])
plt.ylim([0.005, 0.06])

file_name = f"Supp_Figure_posterior_variance_many_conditions"
# Save the figure
plt.savefig(f"{drop_dir}/figures/{file_name}.pdf")
plt.savefig(f"{drop_dir}/figures/{file_name}.png")
plt.savefig(f"{local_dir}/figures/{file_name}.pdf")
plt.savefig(f"{local_dir}/figures/{file_name}.png")


# %%

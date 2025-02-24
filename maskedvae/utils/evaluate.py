import gc
import math
import os
import pickle
import warnings

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from maskedvae.datasets.datasets import load_train_valid_test_data
from maskedvae.plotting.plotting import plot_rate_samples_and_spikes
from maskedvae.utils.loss import eval_VAE_prior
from maskedvae.utils.utils import (
    compute_corr,
    compute_rmse,
    cpu,
    gpu,
    poisson_sample_from_multiple_predictions,
    rebin,
)
from maskedvae.model.networks import apply_mask

from pandas import DataFrame as DF
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LinearRegression


def relative_PoissonNLLLoss(pred, truth):
    """Calculates the improvement in Poisson
    log likelihood over predicting using the mean firing rate"""
    loss = nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")

    # Explicitly define the device (CPU or CUDA) and data type
    device = pred.device if torch.is_tensor(pred) else torch.device("cpu")
    dtype = pred.dtype if torch.is_tensor(pred) else torch.float32

    base_l = loss(
        torch.tensor(truth, device=device, dtype=dtype).mean(0, keepdim=True),
        torch.tensor(truth, device=device, dtype=dtype),
    ).mean(0)
    rec_l = loss(
        torch.tensor(pred, device=device, dtype=dtype),
        torch.tensor(truth, device=device, dtype=dtype),
    ).mean(0)

    # If it was meant to move the tensor to CPU, use `.cpu()` instead.
    return cpu(base_l - rec_l)


def Corr_monkey(pred, truth):
    """Calculates the average Pearson correlation
    coefficient for a number of traces. Ignores empty traces

    Args:
        pred: prediction
        truth: ground truth

    Returns: Pearson correlation coefficient

    """
    ccc = []
    for i in range(pred.shape[1]):
        if truth[0, i, :].sum() and pred[0, i, :].sum():
            cc = np.corrcoef(pred[0, i, :], truth[0, i, :])[0, 1]
            if not math.isnan(cc):
                ccc.append(cc)
    return np.array(ccc)


def RMSE_monkey(pred, truth):
    """Calculates the root mean squared error"""
    return np.sqrt(np.mean((pred - truth) ** 2, axis=-1))


def compute_cumultative_prob_ratio_mutiple_units(
    gt_spike_trains, spike_train_samples, max_val=40, enable_err=False
):
    """
    Args:
        gt_spike_trains: Ground truth spike trains
        spike_train_samples: Spike train samples
        bin_size: Bin size for the histogram
    Returns:
        cum_sum_samples: Cumulative sum of the histogram of the spike train samples
        cum_sum_gt: Cumulative sum of the histogram of the ground truth spike train
    """
    n_samples, n_neurons, n_time_steps = spike_train_samples.shape
    n_neurons_gt, n_time_steps_gt = gt_spike_trains.shape

    assert (
        n_time_steps == n_time_steps_gt
    ), "Number of time steps in the samples and ground truth spike trains do not match"
    assert (
        n_neurons == n_neurons_gt
    ), "Number of neurons in the samples and ground truth spike trains do not match"

    # Initialize a list to hold the histograms
    histograms = []
    histograms_gt = []
    # Determine the global max bincount across all spike train samples
    max_bincount = max(spike_train_samples.max(), gt_spike_trains.max())
    if max_bincount >= max_val:
        print(
            "CAUTION: Max bincount is greater than the max_val", max_bincount, max_val
        )
        # print index of neurons with max bincount
        print(np.where(spike_train_samples.max(axis=2) >= max_val))
        print(np.where(gt_spike_trains.max(axis=1) >= max_val))
    if enable_err:
        assert max_bincount <= max_val, "Max bincount is greater than the max_val"

    max_bincount = max_val
    # Define bin edges based on bin size and max bincount
    bins = np.arange(0, max_bincount + 2, 1)

    spike_train_samples = spike_train_samples.reshape(
        n_samples * n_neurons, n_time_steps
    )
    # Iterate over the samples
    for i in range(n_samples * n_neurons):
        # Compute histogram for the current sample
        hist, _ = np.histogram(spike_train_samples[i, :], bins=bins)

        # Add the histogram to the list
        histograms.append(hist)

    histograms = np.array(histograms)

    for i in range(n_neurons):
        # Compute histogram for the current sample
        hist, _ = np.histogram(gt_spike_trains[i, :], bins=bins)

        # Add the histogram to the list
        histograms_gt.append(hist)

    histograms_gt = np.array(histograms_gt)

    cum_sum_samples = np.cumsum(histograms, axis=1) / np.sum(
        histograms, axis=1, keepdims=True
    )
    cum_sum_gt = np.cumsum(histograms_gt, axis=1) / np.sum(
        histograms_gt, axis=1, keepdims=True
    )

    cum_sum_samples = cum_sum_samples.reshape(n_samples, n_neurons, -1)
    return cum_sum_samples, cum_sum_gt, bins


def compute_calibration(multi_recons, gt_data, single_dim=False, single_index=0):
    """
    Compute the calibration multiple samples
    Check how often the gt val falls into the
    nth percentile

    Parameters
    ----------
    multi_recons reconstructed samples from approx pos
    gt_data original data
    single_dim if True, compute calibration across only
        one dimension single test sample monkey data vs.
        multiple samples fly data
    single_index if single_dim is True, specify which dimension to average across

    Returns
    -------
    sum_dict dictionary that gives ratios how often
            a value fell into the nth percentile
    """
    lower = [2.5, 5, 10, 20]
    upper = [97.5, 95, 90, 80]
    label = [95, 90, 80, 60]
    sum_dict = {i: 0 for i in label}

    # adjust mask
    for i, (l, u) in enumerate(zip(lower, upper)):
        lower_per = np.percentile(multi_recons, l, axis=0)
        upper_per = np.percentile(multi_recons, u, axis=0)
        hits = (gt_data >= lower_per) & (gt_data <= upper_per)
        if single_dim:
            summed_hits = np.sum(hits, axis=single_index) / hits.shape[single_index]
        else:
            summed_hits = np.sum(np.sum(hits, axis=0), axis=0) / (
                hits.shape[0] * hits.shape[1]
            )
        sum_dict[label[i]] = summed_hits
    return sum_dict


def get_masked_outputs(
    model,
    data,
    get_predictions=True,
    sessions=(0, 1),
    t_slice=np.index_exp[:10000],
    lat_mask=None,
    inp_mask=("xb_y", "xb_d"),
    outputs=None,
):
    """Infers all outputs given in 'outputs'. Will store network outputs as well as input data.
    Everything is stored in a MultiIndex dataframe."""
    if lat_mask is None:
        lat_mask = gpu(torch.ones(model.net.n_latents))
    if outputs is None:
        outputs = model.net.outputs.keys()

    types = ["Data", "Pred"]
    column_names = ["Modality", "Type", "Session", "Channel"]

    df = DF()

    for t in sessions:

        session_data = data.get_session(t, to_gpu=True, t_slice=t_slice)

        # Run network

        for k in model.net.scaling:
            if model.net.ifdimwise_scaling:
                for ii in range(session_data[k].shape[1]):
                    session_data[k][:, ii] = model.net.dimwise_scaling[k + str(ii)][t][
                        0
                    ] * (
                        session_data[k][:, ii]
                        - model.net.dimwise_scaling[k + str(ii)][t][1]
                    )
            else:
                session_data[k] = model.net.scaling[k][t][0] * (
                    session_data[k] - model.net.scaling[k][t][1]
                )  # Scale traces for training
            print(k, "Scaled... ")

        network_outputs = model.net.encode(session_data, inp_mask=inp_mask)
        network_outputs["z_mu"] *= lat_mask[None, :, None]
        network_outputs["z_sp"] = model.net.make_sparse(network_outputs["z_mu"])

        if get_predictions:
            network_outputs.update(
                model.net.decode(network_outputs["z_mu"], t, scale_output=False)
            )

        # Write results into data frame

        cols = pd.MultiIndex.from_product(
            [["z_mu"], ["Pred"], [t], range(model.net.n_latents)], names=column_names
        )
        df = pd.concat(
            [
                df,
                DF(
                    data=cpu(network_outputs["z_mu"][0].T).astype("float16"),
                    columns=cols,
                ),
            ],
            axis=1,
        )

        cols = pd.MultiIndex.from_product(
            [["z_lsig"], ["Pred"], [t], range(model.net.n_latents)], names=column_names
        )
        df = pd.concat(
            [
                df,
                DF(
                    data=cpu(network_outputs["z_lsig"][0].T).astype("float16"),
                    columns=cols,
                ),
            ],
            axis=1,
        )

        cols = pd.MultiIndex.from_product(
            [["z_sp"], ["Pred"], [t], range(model.net.n_group_latents)],
            names=column_names,
        )
        df = pd.concat(
            [
                df,
                DF(
                    data=cpu(network_outputs["z_sp"][0].T).astype("float16"),
                    columns=cols,
                ),
            ],
            axis=1,
        )

        if get_predictions:
            for o in outputs:
                cols = pd.MultiIndex.from_product(
                    [[o], ["Pred"], [t], range(data.n_traces[o])], names=column_names
                )
                df = pd.concat(
                    [
                        df,
                        DF(
                            data=cpu(network_outputs[o][0].T).astype("float16"),
                            columns=cols,
                        ),
                    ],
                    axis=1,
                )

                cols = pd.MultiIndex.from_product(
                    [[o + "_fac"], ["Pred"], [t], range(model.net.n_factors)],
                    names=column_names,
                )
                df = pd.concat(
                    [
                        df,
                        DF(
                            data=cpu(network_outputs[o + "_fac"][0].T).astype(
                                "float16"
                            ),
                            columns=cols,
                        ),
                    ],
                    axis=1,
                )

        for o in list(outputs) + ["xb_u"]:
            gt_data = (
                cpu(session_data[o][0].T).astype("uint8")
                if "xa" in o
                else cpu(session_data[o][0].T).astype("float16")
            )
            cols = pd.MultiIndex.from_product(
                [[o], ["Data"], [t], range(data.n_traces[o])], names=column_names
            )
            df = pd.concat([df, DF(data=gt_data, columns=cols)], axis=1)

    df = df.reindex(sorted(df.columns), axis=1)

    return df


def get_sampled_masked_outputs(
    model,
    data,
    get_predictions=True,
    sessions=(0, 1),
    t_slice=np.index_exp[:10000],
    lat_mask=None,
    inp_mask=("xb_y", "xb_d"),
    test_masks=[
        "all_obs",
        "xb_yd",
        "xa_m",
    ],  
    comp_calibration=True,  # compute calibration for the model
    outputs=None,
    store_samples=True,
    neuro=False,
    n_post_samples=50,  # number of samples from the posterior
    exp_dir="./",
    drop_dir="./",
    eval_test_nslice=1000,
):
    """Infers all outputs given in 'outputs'. Will store network outputs as well as input data.
    Everything is stored in a MultiIndex dataframe."""

    if lat_mask is None:
        lat_mask = gpu(torch.ones(model.net.n_latents))
    if outputs is None:
        outputs = model.net.outputs.keys()

    types = ["Data", "Pred"]
    column_names = ["Modality", "Type", "Session", "Channel"]

    xa_m_samples = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }
    xb_y_samples = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }
    xb_d_samples = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }
    xb_y_samples_mean = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }
    xb_d_samples_mean = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }

    calibration_dict = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xb_y", "xb_d", "xa_m"]
    }

    mean_decoding = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xb_y", "xb_d", "xa_m"]
    }
    RMSE_mean_decoding = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xb_y", "xb_d"]
    }
    LogL_mean_decoding = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xa_m"]
    }
    CorCoeff_mean_decoding = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xb_y", "xb_d"]
    }

    latent_mean = {
        train: {mask: {t: 0 for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }

    latent_sig = {
        train: {mask: {t: 0 for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }

    for t in sessions:

        session_data = data.get_session(t, to_gpu=True, t_slice=t_slice)

        # Run network

        for k in model.net.scaling:
            if model.net.ifdimwise_scaling:
                for ii in range(session_data[k].shape[1]):
                    session_data[k][:, ii] = model.net.dimwise_scaling[k + str(ii)][t][
                        0
                    ] * (
                        session_data[k][:, ii]
                        - model.net.dimwise_scaling[k + str(ii)][t][1]
                    )
            else:
                session_data[k] = model.net.scaling[k][t][0] * (
                    session_data[k] - model.net.scaling[k][t][1]
                )  # Scale traces for training
            print(k, "Scaled... ")

        # save scaled session data
        with open(exp_dir + f"session_data_{t}.pkl", "wb") as f:
            pickle.dump(session_data, f)

        model_tag = "naive" if model.full_mask == 1 else "masked"
        # select different test masks

        for mask_key in test_masks:
            # if the mask has not been generated for the model, skip
            if mask_key not in model.masks.keys() and mask_key != "all_obs":
                print(mask_key + " not in model.masks.keys() or mask_key != all_obs")
                continue
            # for all masks that are part of the model pass the mask as train_mask (multiplies the mask with the input)
            if mask_key != "all_obs":
                print(mask_key, "Masked... ")
                network_outputs = model.net.encode(
                    session_data,
                    train_mask=model.masks[mask_key][: model.net.n_inputs],
                    inp_mask=inp_mask,
                )
            else:  # for all observed train_mask is None and masking step is skipped
                print("all_observed")
                network_outputs = model.net.encode(session_data, inp_mask=inp_mask)
            # apply the latent mask a remnant of the disentanglement step (not applied for the current model)
            network_outputs["z_mu"] *= lat_mask[None, :, None]
            network_outputs["z_sp"] = model.net.make_sparse(network_outputs["z_mu"])

            # append the latent mean and latent log sigma
            latent_mean[model_tag][mask_key][t] = (
                network_outputs["z_mu"].cpu().detach().numpy()
            )
            latent_sig[model_tag][mask_key][t] = (
                torch.exp(network_outputs["z_lsig"]).cpu().detach().numpy()
            )

            # sample n_times from the approx posterior
            print(model_tag, mask_key, t, "Sampling... ")
            for i in range(n_post_samples):
                z = torch.distributions.Normal(
                    network_outputs["z_mu"], torch.exp(network_outputs["z_lsig"])
                ).rsample()
                out = model.net.decode(z, t, scale_output=False)
                if "xb_d" in model.net.outputs:
                    # check if gnll loss
                    if model.net.outputs["xb_d"] == "gnll":
                        out["xb_d_sample"] = torch.distributions.Normal(
                            out["xb_d"], torch.sqrt(out["xb_d_noise"])
                        ).rsample()
                    else:
                        out["xb_d_sample"] = out["xb_d"]
                    xb_d_samples_mean[model_tag][mask_key][t].append(
                        out["xb_d"][0].cpu().detach().numpy()
                    )
                    xb_d_samples[model_tag][mask_key][t].append(
                        out["xb_d_sample"][0].cpu().detach().numpy()
                    )
                if "xb_y" in model.net.outputs:
                    if model.net.outputs["xb_y"] == "gnll":
                        out["xb_y_sample"] = torch.distributions.Normal(
                            out["xb_y"], torch.sqrt(out["xb_y_noise"])
                        ).rsample()
                    else:
                        out["xb_y_sample"] = out["xb_y"]

                    xb_y_samples_mean[model_tag][mask_key][t].append(
                        out["xb_y"][0].cpu().detach().numpy()
                    )
                    xb_y_samples[model_tag][mask_key][t].append(
                        out["xb_y_sample"][0].cpu().detach().numpy()
                    )
                if "xa_m" in model.net.outputs:
                    xa_m_samples[model_tag][mask_key][t].append(
                        out["xa_m"][0].cpu().detach().numpy()
                    )

            for out_str in ["xb_y", "xb_d", "xa_m"]:
                if (
                    out_str in model.net.outputs
                ):  # check in case out_str is not in model.net.outputs
                    mean_decoding[out_str][model_tag][mask_key][t] = model.net.decode(
                        network_outputs["z_mu"], t, scale_output=False
                    )[out_str]
                    if out_str == "xa_m":
                        LogL_mean_decoding[out_str][model_tag][mask_key][
                            t
                        ] = relative_PoissonNLLLoss(
                            mean_decoding[out_str][model_tag][mask_key][t][
                                :, :, model.net.warmup :
                            ],
                            session_data[out_str][:, :, model.net.warmup :],
                        )
                    elif out_str in ["xb_y", "xb_d"]:
                        CorCoeff_mean_decoding[out_str][model_tag][mask_key][
                            t
                        ] = Corr_monkey(
                            mean_decoding[out_str][model_tag][mask_key][t][
                                :, :, model.net.warmup :
                            ]
                            .cpu()
                            .detach()
                            .numpy(),
                            session_data[out_str][:, :, model.net.warmup :]
                            .cpu()
                            .detach()
                            .numpy(),
                        )
                        RMSE_mean_decoding[out_str][model_tag][mask_key][
                            t
                        ] = RMSE_monkey(
                            mean_decoding[out_str][model_tag][mask_key][t][
                                :, :, model.net.warmup :
                            ]
                            .cpu()
                            .detach()
                            .numpy(),
                            session_data[out_str][:, :, model.net.warmup :]
                            .cpu()
                            .detach()
                            .numpy(),
                        )

                    mean_decoding[out_str][model_tag][mask_key][t] = (
                        mean_decoding[out_str][model_tag][mask_key][t]
                        .cpu()
                        .detach()
                        .numpy()
                    )

            if get_predictions:
                network_outputs.update(
                    model.net.decode(network_outputs["z_mu"], t, scale_output=False)
                )

            if comp_calibration:
                # ignore the first model.net.warmup steps where no loss is computed due to RNN unrolling
                if "xb_y" in model.net.outputs:
                    calibration_dict["xb_y"][model_tag][mask_key][
                        t
                    ] = compute_calibration(
                        multi_recons=np.array(xb_y_samples[model_tag][mask_key][t])[
                            :, :, model.net.warmup :
                        ],
                        gt_data=session_data["xb_y"][0, :, model.net.warmup :]
                        .cpu()
                        .detach()
                        .numpy(),
                        single_dim=True,
                        single_index=1,
                    )
                if "xb_d" in model.net.outputs:
                    calibration_dict["xb_d"][model_tag][mask_key][
                        t
                    ] = compute_calibration(
                        multi_recons=np.array(xb_d_samples[model_tag][mask_key][t])[
                            :, :, model.net.warmup :
                        ],
                        gt_data=session_data["xb_d"][0, :, model.net.warmup :]
                        .cpu()
                        .detach()
                        .numpy(),
                        single_dim=True,
                        single_index=1,
                    )

                # neural calibration is cumulative probability ratio
                # Constants and settings
                t_max = 1050  # highest possible time point
                t_low_all = 50  # lowest time point to start plotting
                time_slice = 200
                t_bins = np.arange(t_low_all, t_max, time_slice)
                cal_dict = {
                    "gt": [],
                    "samples": [],
                    "gt_rebin": [],
                    "samples_rebin": [],
                }
                for tt, t_low in enumerate(t_bins[:-1]):
                    t_high = t_low + time_slice

                    (
                        spike_train_samples,
                        gt_spike_train,
                        mean_gt_spikes,
                    ) = poisson_sample_from_multiple_predictions(
                        session_data["xa_m"][0, :, t_low:t_high].detach().cpu().numpy(),
                        np.array(xa_m_samples[model_tag][mask_key][t])[
                            :, :, t_low:t_high
                        ],
                        n_samples=10,
                        axis=0,
                    )
                    gc.collect()
                    (
                        cum_sum_samples,
                        cum_sum_gt,
                        bins,
                    ) = compute_cumultative_prob_ratio_mutiple_units(
                        gt_spike_train, spike_train_samples, max_val=15
                    )  # added maxval to ensure all have same length
                    cal_dict["gt"].append(cum_sum_gt)
                    cal_dict["samples"].append(cum_sum_samples)
                    summed_bins = 5
                    n_samples, n_neurons, n_times = spike_train_samples.shape
                    # rebin the spike trains and reshape them
                    spike_train_samples_resampled = rebin(
                        spike_train_samples.reshape(n_samples * n_neurons, n_times),
                        summed_bins=summed_bins,
                    )
                    spike_train_samples_resampled = (
                        spike_train_samples_resampled.reshape(n_samples, n_neurons, -1)
                    )
                    gt_spike_train_resampled = rebin(
                        gt_spike_train, summed_bins=summed_bins
                    )
                    (
                        cum_sum_samples,
                        cum_sum_gt,
                        bins,
                    ) = compute_cumultative_prob_ratio_mutiple_units(
                        gt_spike_train_resampled,
                        spike_train_samples_resampled,
                        max_val=40,
                    )  # added maxval to ensure all have same length
                    cal_dict["gt_rebin"].append(cum_sum_gt)
                    cal_dict["samples_rebin"].append(cum_sum_samples)
                    del spike_train_samples_resampled
                    del spike_train_samples
                    gc.collect()
                gc.collect()
                calibration_dict["xa_m"][model_tag][mask_key][t] = cal_dict
            if t == 0:
                plot_rate_samples_and_spikes(
                    session_data=session_data["xa_m"][0].cpu().detach().numpy(),
                    xa_m_samples=np.array(xa_m_samples[model_tag][mask_key][t]),
                    colscmap="Reds" if model_tag == "masked" else "Blues",
                    axn1=4,
                    axn2=4,
                    model_tag=model_tag,
                    mask_key=mask_key,
                    t=0,
                    zoom_t_low=50,
                    zoom_t_high=250,
                    fps=model.fps,
                    exp_dir=exp_dir,
                )

        print("Done")

    with open(exp_dir + "calibration_dict.pkl", "wb") as f:
        pickle.dump(calibration_dict, f)

    with open(exp_dir + "mean_decoding.pkl", "wb") as f:
        pickle.dump(mean_decoding, f)
    with open(exp_dir + "RMSE_mean_decoding.pkl", "wb") as f:
        pickle.dump(RMSE_mean_decoding, f)
    with open(exp_dir + "LogL_mean_decoding.pkl", "wb") as f:
        pickle.dump(LogL_mean_decoding, f)
    with open(exp_dir + "CorCoeff_mean_decoding.pkl", "wb") as f:
        pickle.dump(CorCoeff_mean_decoding, f)

    # plot the calibration plot 'xb_y' for the masked case
    if comp_calibration and "xb_yd" in test_masks:
        mask_key = "xb_yd"
        if (
            calibration_dict["xb_y"][model_tag][mask_key][t] is not None
            and "xb_y" in model.net.outputs
        ):
            plt.figure(figsize=(5, 3))
            label = [60, 80, 90, 95]

            for percentile in label:
                for x_index in [0, 1]:
                    plt.plot(
                        percentile,
                        100
                        * calibration_dict["xb_y"][model_tag][mask_key][t][percentile][
                            x_index
                        ],
                        ".",
                        ms=5,
                        alpha=0.8,
                        color="blue" if model_tag == "naive" else "red",
                    )
            plt.plot(label, label, "grey")
            plt.tight_layout()
            file_name = f"calibration_{mask_key}_xb_d_"
            plt.savefig(f"{exp_dir}/{file_name}.pdf", bbox_inches="tight")
            plt.savefig(f"{exp_dir}/{file_name}.png", bbox_inches="tight")

    if store_samples:
        with open(exp_dir + "samples.pkl", "wb") as f:
            pickle.dump(
                {
                    "xb_d": xb_d_samples,
                    "xb_y": xb_y_samples,
                    "xa_m": xa_m_samples if neuro else 0,
                    "xb_d_mean": xb_d_samples_mean,
                    "xb_y_mean": xb_y_samples_mean,
                },
                f,
            )

    # garbage collection
    del (
        CorCoeff_mean_decoding,
        LogL_mean_decoding,
        RMSE_mean_decoding,
        mean_decoding,
        xb_d_samples_mean,
        xb_y_samples_mean,
        xa_m_samples,
    )
    gc.collect()

    return calibration_dict, xb_d_samples, xb_y_samples


def perform_regression(
    df_eval, reg_b=sklearn.linear_model.Ridge(), reg_a=sklearn.linear_model.Ridge()
):
    # Perform regression on the latent variables

    eval_vars = [
        k
        for k in df_eval.columns.get_level_values("Modality").unique()
        if "x" in k and "fac" not in k
    ]
    sessions = [k for k in df_eval.columns.get_level_values("Session").unique()]

    for s in sessions:

        z = df_eval["Train", "z_sp", "Pred"].values
        z_inds = z.sum(0).nonzero()[0]
        z_train = z[:, z_inds]

        z = df_eval["Valid", "z_sp", "Pred"].values
        z_valid = z[:, z_inds]
        for k in eval_vars:

            if "xb" in k:
                reg_z = reg_b
            if "xa" in k:
                reg_z = reg_a

            ### Train regressor on training data

            y_train = df_eval["Train", k, "Data", s].values
            # if either y_train or z_train is all zeros, skip this channel
            if not y_train.sum() or not z_train.sum():
                print("Skipping channel due to all zeros", "session", s, "eval_var", k)
                continue
            else:
                reg_z = reg_z.fit(z_train, y_train)

                ### Apply regressor to validation data

                y_reg_z = reg_z.predict(z_valid)
                if "xa" in k:
                    y_reg_z = np.clip(y_reg_z, 0, np.inf)

                cols = pd.MultiIndex.from_product(
                    [["Valid"], [k], ["Reg(z)"], [s], range(y_reg_z.shape[-1])],
                    names=["T_V", "Modality", "Type", "Session", "Channel"],
                )
                df_eval = pd.concat(
                    [df_eval, DF(data=y_reg_z.astype("float16"), columns=cols)], axis=1
                )
                df_eval = df_eval.reindex(sorted(df_eval.columns), axis=1)

    return df_eval


def eval_helper_func(df, sdict, target, pred):
    """Helper function that evaluates multiple performance metrics on a given prediction
    and target and adds them to a dataframe

    Args:
        df: dataframe
        sdict: dictionary
        target: target
        pred: prediction

    Returns:
        prediction metric appended to dataframe
    """
    if pred is None:
        sdict["CorrCoef"] = sdict["R^2"] = sdict["RMSE"] = 0
    else:
        with warnings.catch_warnings():
            z_inds = abs(target).max(0).nonzero()[0]
            target = target[:, z_inds]
            pred = pred[:, z_inds]
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sdict["CorrCoef"] = compute_corr(pred, target).mean()
            sdict["R^2"] = r2_score(target, pred, multioutput="uniform_average")
            sdict["RMSE"] = compute_rmse(pred, target)
        if "xa" in sdict["Modality"]:
            sdict["LogL/T"] = relative_PoissonNLLLoss(pred, target).mean()

    return pd.concat([df, pd.DataFrame([sdict])], ignore_index=True)


def get_eval_metrics(df_eval):
    """Extracts evaluation metrics from a dataframe of predictions and targets"""
    df = pd.DataFrame(
        columns=(
            "Modality",
            "Session",
            "T_V",
            "Type",
            "CorrCoef",
            "R^2",
            "RMSE",
            "LogL/T",
        )
    )

    eval_vars = [
        k
        for k in df_eval.columns.get_level_values("Modality").unique()
        if "x" in k and "fac" not in k
    ]
    sessions = [k for k in df_eval.columns.get_level_values("Session").unique()]
    types = [
        k for k in df_eval.columns.get_level_values("Type").unique() if "Data" not in k
    ]

    for s in sessions:
        for k in eval_vars:
            for ty in types:
                if ty in df_eval["Train", k]:
                    df = eval_helper_func(
                        df,
                        {"Modality": k, "Type": ty, "Session": s, "T_V": "Train"},
                        df_eval["Train", k, "Data", s].values,
                        df_eval["Train", k, ty, s].values,
                    )
                if ty in df_eval["Valid", k]:
                    df = eval_helper_func(
                        df,
                        {"Modality": k, "Type": ty, "Session": s, "T_V": "Valid"},
                        df_eval["Valid", k, "Data", s].values,
                        df_eval["Valid", k, ty, s].values,
                    )

    df.set_index(["Modality", "Session", "T_V", "Type"], inplace=True)
    df.rename_axis(["Metric"], axis=1, inplace=True)
    return df


def run_and_eval(
    model,
    data_train,
    data_val,
    sessions=(0,),
    eval_vars=None,
    t_slice=np.index_exp[:2000],
    lat_mask=None,
    inp_mask=("xb_y", "xb_d"),
    run_regression=True,
):
    """
    Run the model and evaluate the performance on the training and validation data

    Args:
        model: trained model
        data_train: training data
        data_val: validation data
        sessions (tupel): sessions to evaluate
        eval_vars (list): evaluation variables
        t_slice (slice): time slice for evaluation
        lat_mask: mask for latent variables
        inp_mask: mask for input variables
        run_regression: whether to run regression on the evaluation variables
    """

    df_eval = get_masked_outputs(
        model,
        data_train,
        True,
        sessions,
        t_slice,
        lat_mask,
        inp_mask,
        outputs=eval_vars,
    )
    df_eval = pd.concat([df_eval], axis=1, keys=["Train"], names=["T_V"])
    df_eval = pd.concat(
        [
            df_eval,
            pd.concat(
                [
                    get_masked_outputs(
                        model,
                        data_val,
                        True,
                        sessions,
                        t_slice,
                        lat_mask,
                        inp_mask,
                        outputs=eval_vars,
                    )
                ],
                axis=1,
                keys=["Valid"],
                names=["T_V"],
            ),
        ],
        axis=1,
    )
    df_eval = df_eval.reindex(sorted(df_eval.columns), axis=1)

    if run_regression:
        df_eval = perform_regression(df_eval)

    return df_eval, get_eval_metrics(df_eval)


def latent_ablation(
    model, data_train, data_val, sessions=(0, 1), t_slice=np.index_exp[:10000]
):
    """
    Performs a latent ablation study on the model. For each latent variable, the model is evaluated by masking
    out one latent variable at a time and measuring the performance.

    Args:
        model (Model): Inference network (VAE)
        data_train: Training dataset
        data_val: Validation dataset
        sessions (tuple of int): Indices of sessions within data
        t_slice (slice): Time slice for evaluation

    Returns:
        df_col (DataFrame): Performance metrics for each latent ablation
    """

    df_col = pd.DataFrame()  # Initialize an empty DataFrame to store results

    for t in sessions:
        # Get masked outputs for the current session
        df_eval = get_masked_outputs(model, data_train, True, (t,), t_slice)

        # Calculate KL divergence for the latent variables
        kl_div = cpu(
            eval_VAE_prior(
                gpu(df_eval["z_mu", "Pred", t].values.T)[None],
                gpu(df_eval["z_lsig", "Pred", t].values.T)[None],
                0,
                np.log(model.prior_sig),
            ).sum(-1)
        )[0]

        lat_mask = gpu(np.ones_like(kl_div))  # Start with no latents masked

        for i in range((kl_div > 0.1).sum()):  # Loop through latents above threshold

            # Run model with current latent mask and evaluate
            _, df_perf = run_and_eval(
                model, data_train, data_val, (t,), t_slice=t_slice, lat_mask=lat_mask
            )
            df_perf["n_lat_cut"] = i

            # Concatenate the results to df_col (replacing append)
            df_col = pd.concat([df_col, df_perf], ignore_index=True, sort=True)

            if i > 0:
                # Reverse the mask and re-evaluate
                _, df_perf = run_and_eval(
                    model,
                    data_train,
                    data_val,
                    (t,),
                    t_slice=t_slice,
                    lat_mask=1 - lat_mask,
                )
                df_perf["n_lat_cut"] = -(i)

                # Concatenate the reverse mask results to df_col
                df_col = pd.concat([df_col, df_perf], ignore_index=True, sort=True)

            # Mask out the latent with the highest KL divergence
            max_ind = np.argmax(kl_div)
            kl_div[max_ind] = 0  # Set the KL divergence to 0 for this latent
            assert isinstance(
                lat_mask, (torch.Tensor, np.ndarray)
            ), "lat_mask should be either a torch tensor or numpy array."
            lat_mask[max_ind] = 0  # pylint: disable=assignment-from-no-return

    return df_col


def cross_latent_perf(model, df_eval):
    """Evaluate the ability to cross-predict latents from other latents
    Args:
        model: trained model
        df_eval: evaluation dataframe

    Returns:
        df_perf: performance metrics for cross-latent prediction
    """

    if "z_smdy" in model.net.group_latents:
        lat_inputs = [
            ("z_s",),
            ("z_m",),
            ("z_y",),
            ("z_d",),
            ("z_smdy",),
            ("z_s", "z_smdy"),
            ("z_m", "z_smdy"),
            ("z_y", "z_smdy"),
            ("z_d", "z_smdy"),
            ("z_s", "z_m", "z_d", "z_y", "z_smdy"),
        ]
    else:
        lat_inputs = [
            ("z_s",),
            ("z_m",),
            ("z_y",),
            ("z_d",),
            ("z_s", "z_m", "z_d", "z_y"),
        ]

    eval_vars = [k for k in set(df_eval.columns.get_level_values(1)) if "x" in k]
    sessions = set(df_eval.columns.get_level_values(3))
    df_perf = DF()

    for t in sessions:

        for l_is in lat_inputs:

            df_sub = df_eval.loc[:, pd.IndexSlice[:, eval_vars, "Data", t]]

            z_sp = np.concatenate(
                [
                    df_eval["Train", "z_sp", "Pred", t].values[:, model.net.lat_inds[k]]
                    for k in l_is
                ],
                1,
            )
            cols = pd.MultiIndex.from_product(
                [["Train"], ["z_sp"], ["Pred"], [t], range(z_sp.shape[1])],
                names=["T_V", "Modality", "Type", "Session", "Channel"],
            )
            df_sub = pd.concat([df_sub, DF(data=z_sp, columns=cols)], axis=1)

            z_sp = np.concatenate(
                [
                    df_eval["Valid", "z_sp", "Pred", t].values[:, model.net.lat_inds[k]]
                    for k in l_is
                ],
                1,
            )
            cols = pd.MultiIndex.from_product(
                [["Valid"], ["z_sp"], ["Pred"], [t], range(z_sp.shape[1])],
                names=["T_V", "Modality", "Type", "Session", "Channel"],
            )
            df_sub = pd.concat([df_sub, DF(data=z_sp, columns=cols)], axis=1)

            df_sub = df_sub.reindex(sorted(df_sub.columns), axis=1)
            df_sub = perform_regression(df_sub)

            df_perf_sub = get_eval_metrics(df_sub)
            df_perf_sub = pd.concat(
                [df_perf_sub], keys=[",".join(l_is)], names=["Latents"]
            )
            df_perf = pd.concat([df_perf, df_perf_sub], axis=0)

    return df_perf


def run_full_evaluation(
    model_path,
    data_path,
    save_path=None,
    run_regression=True,
    run_latent_ablation=True,
    run_cross_perf=True,
    fr_threshold=0.5,
    **kwargs,
):
    """
    Runs a full evaluation of the model on the data.
    Changed to now do it on test data (not valid)
    This includes:
    - Basic evaluation
    - Latent ablation
    - Cross latent performance

    """
    p_dict = {
        "sessions": (0,),
        "t_slice": np.index_exp[:2000],
        "inp_mask": ("xb_y", "xb_d"),
        "lat_mask": None,
        "reg_b": sklearn.linear_model.Ridge(),
        "reg_a": sklearn.linear_model.Ridge(),
        "eval_pars": None,
    }

    for key, value in kwargs.items():
        p_dict[key] = value


    if isinstance(data_path, str):
        with open(data_path, "rb") as f:
            all_sessions = pickle.load(f)
    else:
        all_sessions = data_path

    if isinstance(model_path, str):
        with open(model_path, "rb") as f:
            model = torch.load(f)
    else:
        model = model_path

    PD_train, PD_test, PD_valid = load_train_valid_test_data(
        model,
        all_sessions,
        plot=False,
        start_train=0,
        start_test=0.7,
        start_valid=0.8,
        end_valid=1,
        fr_threshold=fr_threshold,
    )

    eval_test_nslice = min([PD_test.n_bins[qq] for qq in list(p_dict["sessions"])]) - 1
    print(f"eval_test_nslice: {eval_test_nslice}")
    p_dict["t_slice_test"] = np.index_exp[:eval_test_nslice]

    assert (
        p_dict["t_slice_test"] == p_dict["t_slice"]
    ), "t_slice_test and t_slice must be the same"
    return_list = []

    print("Running basic evaluation")
    df_eval, df_perf = run_and_eval(
        model,
        PD_train,
        PD_test,
        sessions=p_dict["sessions"],
        eval_vars=p_dict["eval_pars"],
        t_slice=p_dict["t_slice"],
        lat_mask=p_dict["lat_mask"],
        inp_mask=p_dict["inp_mask"],
        run_regression=run_regression,
    )
    return_list.append(df_eval)

    if run_latent_ablation:
        print("Running latent ablation")
        df_lat_abl = latent_ablation(
            model,
            PD_train,
            PD_test,
            sessions=p_dict["sessions"],
            t_slice=p_dict["t_slice"],
        )
        return_list.append(df_lat_abl)

    if run_cross_perf:
        print("Running cross latent performance")
        df_cross = cross_latent_perf(model, df_eval)
        return_list.append(df_cross)

    if save_path is not None:

        if os.path.isfile(save_path):
            os.remove(save_path)

        df_eval.to_hdf(save_path, "df_eval")
        df_perf.to_hdf(save_path, "df_perf")
        if run_latent_ablation:
            df_lat_abl.to_hdf(save_path, "df_lat_abl")
        if run_cross_perf:
            df_cross.to_hdf(save_path, "df_cross")

        return return_list
    else:
        return return_list


def compute_calibration_score(
    model,
    data,
    get_predictions=True,
    sessions=(0, 1),
    t_slice=np.index_exp[:10000],
    lat_mask=None,
    inp_mask=("xb_y", "xb_d"),
    test_masks=[
        "all_obs",
        "xb_yd",
        "xa_m",
    ],  
    comp_calibration=True,  # compute calibration for the model
    outputs=None,
    store_samples=True,
    neuro=False,
    n_post_samples=50,  # number of samples from the posterior
    exp_dir="./",
    drop_dir="./",
    eval_test_nslice=1000,
):
    """
    Computes the calibration score for the model to be used as a metric for the model performance

    """

    if lat_mask is None:
        lat_mask = gpu(torch.ones(model.net.n_latents))
    if outputs is None:
        outputs = model.net.outputs.keys()

    types = ["Data", "Pred"]
    column_names = ["Modality", "Type", "Session", "Channel"]

    xa_m_samples = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }
    xb_y_samples = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }
    xb_d_samples = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }
    xb_y_samples_mean = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }
    xb_d_samples_mean = {
        train: {mask: {t: [] for t in sessions} for mask in test_masks}
        for train in ["naive", "masked"]
    }

    calibration_dict = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xb_y", "xb_d", "xa_m"]
    }

    mean_decoding = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xb_y", "xb_d", "xa_m"]
    }
    RMSE_mean_decoding = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xb_y", "xb_d"]
    }
    LogL_mean_decoding = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xa_m"]
    }
    CorCoeff_mean_decoding = {
        data_type: {
            train: {mask: {t: 0 for t in sessions} for mask in test_masks}
            for train in ["naive", "masked"]
        }
        for data_type in ["xb_y", "xb_d"]
    }

    for t in sessions:

        session_data = data.get_session(t, to_gpu=True, t_slice=t_slice)

        # Run network
        for k in model.net.scaling:
            if model.net.ifdimwise_scaling:
                for ii in range(session_data[k].shape[1]):
                    session_data[k][:, ii] = model.net.dimwise_scaling[k + str(ii)][t][
                        0
                    ] * (
                        session_data[k][:, ii]
                        - model.net.dimwise_scaling[k + str(ii)][t][1]
                    )
            else:
                session_data[k] = model.net.scaling[k][t][0] * (
                    session_data[k] - model.net.scaling[k][t][1]
                )  # Scale traces for training
            print(k, "Scaled... ")



        model_tag = "naive" if model.full_mask == 1 else "masked"
        # select different test masks

        for mask_key in test_masks:
            # if the mask has not been generated for the model, skip
            if mask_key not in model.masks.keys() and mask_key != "all_obs":
                print(mask_key + " not in model.masks.keys() or mask_key != all_obs")
                continue
            # for all masks that are part of the model pass the mask as train_mask (multiplies the mask with the input)
            if mask_key != "all_obs":
                print(mask_key, "Masked... ")
                network_outputs = model.net.encode(
                    session_data,
                    train_mask=model.masks[mask_key][: model.net.n_inputs],
                    inp_mask=inp_mask,
                )
            else:  # for all observed train_mask is None and masking step is skipped
                print("all_observed")
                network_outputs = model.net.encode(session_data, inp_mask=inp_mask)
            # apply the latent mask a remnant of the disentanglement step (not applied for the current model)
            network_outputs["z_mu"] *= lat_mask[None, :, None]
            network_outputs["z_sp"] = model.net.make_sparse(network_outputs["z_mu"])

            # sample n_times from the approx posterior
            print(model_tag, mask_key, t, "Sampling... ")
            for i in range(n_post_samples):
                z = torch.distributions.Normal(
                    network_outputs["z_mu"], torch.exp(network_outputs["z_lsig"])
                ).rsample()
                out = model.net.decode(z, t, scale_output=False)
                if "xb_d" in model.net.outputs:
                    # check if gnll loss
                    if model.net.outputs["xb_d"] == "gnll":
                        out["xb_d_sample"] = torch.distributions.Normal(
                            out["xb_d"], torch.sqrt(out["xb_d_noise"])
                        ).rsample()
                    else:
                        out["xb_d_sample"] = out["xb_d"]
                    xb_d_samples_mean[model_tag][mask_key][t].append(
                        out["xb_d"][0].cpu().detach().numpy()
                    )
                    xb_d_samples[model_tag][mask_key][t].append(
                        out["xb_d_sample"][0].cpu().detach().numpy()
                    )
                if "xb_y" in model.net.outputs:
                    if model.net.outputs["xb_y"] == "gnll":
                        out["xb_y_sample"] = torch.distributions.Normal(
                            out["xb_y"], torch.sqrt(out["xb_y_noise"])
                        ).rsample()
                    else:
                        out["xb_y_sample"] = out["xb_y"]

                    xb_y_samples_mean[model_tag][mask_key][t].append(
                        out["xb_y"][0].cpu().detach().numpy()
                    )
                    xb_y_samples[model_tag][mask_key][t].append(
                        out["xb_y_sample"][0].cpu().detach().numpy()
                    )
                if "xa_m" in model.net.outputs:
                    xa_m_samples[model_tag][mask_key][t].append(
                        out["xa_m"][0].cpu().detach().numpy()
                    )

            for out_str in ["xb_y", "xb_d", "xa_m"]:
                if (
                    out_str in model.net.outputs
                ):  # check in case out_str is not in model.net.outputs
                    mean_decoding[out_str][model_tag][mask_key][t] = model.net.decode(
                        network_outputs["z_mu"], t, scale_output=False
                    )[out_str]
                    if out_str == "xa_m":
                        LogL_mean_decoding[out_str][model_tag][mask_key][
                            t
                        ] = relative_PoissonNLLLoss(
                            mean_decoding[out_str][model_tag][mask_key][t][
                                :, :, model.net.warmup :
                            ],
                            session_data[out_str][:, :, model.net.warmup :],
                        )
                    elif out_str in ["xb_y", "xb_d"]:
                        CorCoeff_mean_decoding[out_str][model_tag][mask_key][
                            t
                        ] = Corr_monkey(
                            mean_decoding[out_str][model_tag][mask_key][t][
                                :, :, model.net.warmup :
                            ]
                            .cpu()
                            .detach()
                            .numpy(),
                            session_data[out_str][:, :, model.net.warmup :]
                            .cpu()
                            .detach()
                            .numpy(),
                        )
                        RMSE_mean_decoding[out_str][model_tag][mask_key][
                            t
                        ] = RMSE_monkey(
                            mean_decoding[out_str][model_tag][mask_key][t][
                                :, :, model.net.warmup :
                            ]
                            .cpu()
                            .detach()
                            .numpy(),
                            session_data[out_str][:, :, model.net.warmup :]
                            .cpu()
                            .detach()
                            .numpy(),
                        )

                    mean_decoding[out_str][model_tag][mask_key][t] = (
                        mean_decoding[out_str][model_tag][mask_key][t]
                        .cpu()
                        .detach()
                        .numpy()
                    )

            if get_predictions:
                network_outputs.update(
                    model.net.decode(network_outputs["z_mu"], t, scale_output=False)
                )

            if comp_calibration:
                # ignore the first model.net.warmup steps where no loss is computed due to RNN unrolling
                if "xb_y" in model.net.outputs:
                    calibration_dict["xb_y"][model_tag][mask_key][
                        t
                    ] = compute_calibration(
                        multi_recons=np.array(xb_y_samples[model_tag][mask_key][t])[
                            :, :, model.net.warmup :
                        ],
                        gt_data=session_data["xb_y"][0, :, model.net.warmup :]
                        .cpu()
                        .detach()
                        .numpy(),
                        single_dim=True,
                        single_index=1,
                    )
                if "xb_d" in model.net.outputs:
                    calibration_dict["xb_d"][model_tag][mask_key][
                        t
                    ] = compute_calibration(
                        multi_recons=np.array(xb_d_samples[model_tag][mask_key][t])[
                            :, :, model.net.warmup :
                        ],
                        gt_data=session_data["xb_d"][0, :, model.net.warmup :]
                        .cpu()
                        .detach()
                        .numpy(),
                        single_dim=True,
                        single_index=1,
                    )

    # garbage collection
    del (
        CorCoeff_mean_decoding,
        LogL_mean_decoding,
        RMSE_mean_decoding,
        mean_decoding,
        xb_d_samples_mean,
        xb_y_samples_mean,
        xa_m_samples,
    )
    gc.collect()

    squared_loss_x = []
    squared_loss_y = []
    if "xb_y" in model.net.outputs:
        for key_ci in list(calibration_dict["xb_y"][model_tag][mask_key][t].keys()):
            squared_loss_x.append(
                (
                    calibration_dict["xb_y"][model_tag][mask_key][t][key_ci][0]
                    - key_ci / 100
                )
                ** 2
            )
            squared_loss_y.append(
                (
                    calibration_dict["xb_y"][model_tag][mask_key][t][key_ci][1]
                    - key_ci / 100
                )
                ** 2
            )

    # return for hp parameter optimization if needed - not used here
    return squared_loss_x, squared_loss_y


def get_run_properties(temp_dir, run_dirs, run_list=[0], if_changed_args=False):
    """get run properties that are changed between runs
    Args:
        temp_dir (str): path to the run directory
        run_dirs (list): list of run directories
        run_list (list): list of run indices to be considered
    Returns:
        run_properties (dict): dictionary of run properties
    """
    assert run_dirs is not None, "run_dirs is None"
    run_properties = {run: 0 for run in run_dirs}

    exp_dir = temp_dir + run_dirs[0] + "/"
    with open(exp_dir + "args.pkl", "rb") as f:
        args = pickle.load(f)

    if "sessions" not in args.run_params.keys():
        args.run_params["sessions"] = "[35, 37, 40, 41]"

    if "inputs" not in args.run_params.keys():
        args.run_params["inputs"] = "[xa_s, xa_m]"

    list_of_args = {ars: [] for ars in args.run_params.keys()}
    for run in run_dirs:
        exp_dir = temp_dir + run + "/"
        with open(exp_dir + "args.pkl", "rb") as f:
            args = pickle.load(f)
        if "sessions" not in args.run_params.keys():
            args.run_params["sessions"] = "[35, 37, 40, 41]"
        else:
            args.run_params["sessions"] = str(args.run_params["sessions"])

        if "inputs" not in args.run_params.keys():
            args.run_params["inputs"] = "[xa_s, xa_m]"
        else:
            args.run_params["inputs"] = str(args.run_params["inputs"])

        if "group_latents" not in args.run_params.keys():
            args.run_params["group_latents"] = "{'z_smdy': 40}"
        else:
            args.run_params["group_latents"] = str(args.run_params["group_latents"])

        if "layer_pars" not in args.run_params.keys():
            args.run_params["group_latents"] = "[1, 40, 5]"
        else:
            args.run_params["layer_pars"] = str(
                args.run_params["layer_pars"]["cnn_dec"]
            )

        run_properties[run] = args.run_params
        for ars in args.run_params.keys():
            if args.run_params[ars] not in list_of_args[ars]:
                list_of_args[ars].append(args.run_params[ars])

    changed_args = [ars for ars in args.run_params.keys() if 1 < len(list_of_args[ars])]

    changed_run_properties = DF()
    for i, run in enumerate(run_dirs):
        # append all parameters to the dataframe
        # append deprecated in pandas use concat instead
        changed_run_properties = pd.concat(
            [changed_run_properties, pd.DataFrame([run_properties[run]])],
            ignore_index=True,
        )

    # now select only the changed parameters
    changed_run_properties = changed_run_properties[changed_args]
    # insert a column with run_id
    changed_run_properties.insert(
        0, "run_id", [f"run_{i}" for i in range(len(run_dirs))]
    )
    changed_run_properties.insert(
        1, "experiment_id", ["exp" for i in range(len(run_dirs))]
    )

    if if_changed_args:
        return changed_run_properties, changed_args
    else:
        return changed_run_properties


def get_eval_df(temp_dir, run_dirs, run_list=None, df_str="df_perf", iftrain=True):
    """get the evaluation dataframe"""
    df_perf_col = DF()
    if iftrain:
        df_perf_train_col = DF()
    for i, run in enumerate(run_dirs):
        if run_list is None or i in run_list:
            df_perf = pd.read_hdf(temp_dir + run + "/eval_df.h5py", df_str)
            df_perf = pd.concat([df_perf], keys=[i], names=["Run"], axis=1)
            df_perf_col = pd.concat([df_perf_col, df_perf], axis=1)
            if iftrain:
                df_perf_train = torch.load(temp_dir + run + "/model_dicts.pkl")[
                    "df_perf"
                ]
                df_perf_train = pd.concat(
                    [df_perf_train], keys=[i], names=["Run"], axis=1
                )
                df_perf_train_col = pd.concat(
                    [df_perf_train_col, df_perf_train], axis=1
                )

    if iftrain:
        return df_perf_col, df_perf_train_col
    else:
        return df_perf_col, None



# Function to process each dataset and decode the latents for each mask
def infer_latents_monkey(dataset, model, test_masks, lat_mask=None, inp_mask=("xa_s",)):
    # Initialize latent means and stds
    latent_means = []
    latent_stds = []
    spikes_matrices = []
    # Use all latents if no specific mask is provided
    if lat_mask is None:
        lat_mask = gpu(torch.ones(model.net.n_latents))

    # Get session data
    session_data = dataset.get_session(
        0, to_gpu=True
    )  # , t_slice=model.eval_params["t_slice_test"])
    print(session_data["xa_m"].shape)
    # Scale input data
    t = 0
    for k in model.net.scaling:
        if model.net.ifdimwise_scaling:
            for ii in range(session_data[k].shape[1]):
                session_data[k][:, ii] = model.net.dimwise_scaling[k + str(ii)][t][
                    0
                ] * (
                    session_data[k][:, ii]
                    - model.net.dimwise_scaling[k + str(ii)][t][1]
                )
        else:
            session_data[k] = model.net.scaling[k][t][0] * (
                session_data[k] - model.net.scaling[k][t][1]
            )  # Scale traces for training
        print(k, "Scaled...")

    # Decode latents for each mask
    for mask_key in test_masks:
        if mask_key not in model.masks.keys() and mask_key != "all_obs":
            print(mask_key + " not in model.masks.keys() or mask_key != all_obs")
            continue

        # For observed data, don't use masking
        if mask_key != "all_obs":
            print(mask_key, "Masked... ")
            network_outputs = model.net.encode(
                session_data,
                train_mask=model.masks[mask_key][: model.net.n_inputs],
                inp_mask=inp_mask,
            )
            masked_spikes = apply_mask(
                model,
                session_data,
                train_mask=model.masks[mask_key][: model.net.n_inputs],
                inp_mask=inp_mask,
            )
            spikes_matrices.append(masked_spikes[0].detach().cpu().numpy())
        else:
            print("all_observed")
            network_outputs = model.net.encode(session_data, inp_mask=inp_mask)
            masked_spikes = apply_mask(model, session_data, inp_mask=inp_mask)
            spikes_matrices.append(masked_spikes[0].detach().cpu().numpy())
        # Apply the latent mask
        network_outputs["z_sp"] = model.net.make_sparse(network_outputs["z_mu"])

        # Append latents for the current mask
        latent_means.append(network_outputs["z_mu"].detach().cpu().numpy())
        latent_stds.append(torch.exp(network_outputs["z_lsig"]).detach().cpu().numpy())

    return (
        latent_means,
        latent_stds,
        session_data["xb_y"].detach().cpu().numpy(),
        session_data["xb_d"].detach().cpu().numpy(),
        spikes_matrices,
    )


def shift_and_reshape(latent_means, velocity, delay):
    """get rid of first unused dimension and move time to the first dimension"""
    latent_means = latent_means.squeeze().transpose(1, 0)  # [time points x features]
    velocity = velocity.squeeze().transpose(1, 0)  # [time points x vwlocity dimensions]

    # now cut off the last delay elements
    if delay > 0:
        velocity = velocity[delay:, :]
        latent_means = latent_means[:-delay, :]
    return latent_means, velocity


def train_on_all_obs(train_latent_means, train_d, delay, reg_type=None):
    #
    latent_means, shifted_velocity = shift_and_reshape(
        train_latent_means, train_d, delay
    )
    if reg_type == "ridge":
        reg_x = Ridge(alpha=1e-2).fit(latent_means, shifted_velocity[:, 0])
        reg_y = Ridge(alpha=1e-2).fit(latent_means, shifted_velocity[:, 1])
    else:
        reg_x = LinearRegression().fit(latent_means, shifted_velocity[:, 0])
        reg_y = LinearRegression().fit(latent_means, shifted_velocity[:, 1])
    return reg_x, reg_y


def calculate_rmse(true_values, predicted_values):
    """
    calculate root mean square error (rmse) manually
    """
    mse = np.mean((true_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def squeeze_and_reshape(arr):
    """get rid of first unused dimension and move time to the first dimension"""
    arr = arr.squeeze().transpose(1, 0)
    return arr


def predict_and_evaluate(reg_x, reg_y, predictor, target, delay, print_shapes=False):
    """
    Predict velocities and evaluate model performance.

    Args:
        reg_x: Regression model for predicting x velocity.
        reg_y: Regression model for predicting y velocity.
        latent_means (array): Latent means data for prediction, shape [n_samples, n_features].
        velocity (array): Actual velocity data, shape [n_samples, 2].
        delay (int): Time delay to apply when shifting velocity data.

    Returns:
        dict: A dictionary containing performance metrics including:
              - performance: Average R^2 score for both x and y velocities.
              - xperformance: R^2 score for x velocity.
              - yperformance: R^2 score for y velocity.
              - x_rmse: Root mean square error for x velocity.
              - y_rmse: Root mean square error for y velocity.
              - x_correlations: Correlation between predicted and actual x velocities.
              - y_correlations: Correlation between predicted and actual y velocities.
              - predicted_velocities_x: List of predicted x velocities.
              - predicted_velocities_y: List of predicted y velocities.
    """

    predictor_res = squeeze_and_reshape(predictor)
    target_res = squeeze_and_reshape(target)

    if print_shapes:
        print("predictor shape:", predictor.shape)
        print("target shape:", target.shape)
        print("predictor shape:", predictor_res.shape)
        print("target reshape:", target_res.shape)
    # predict velocities
    pred_x = reg_x.predict(predictor_res)
    pred_y = reg_y.predict(predictor_res)

    # calculate the performance (R^2 scores)
    score_x = reg_x.score(predictor_res, target_res[:, 0])
    score_y = reg_y.score(predictor_res, target_res[:, 1])
    performance = (score_x + score_y) / 2.0

    # calculate root mean square error (rmse)
    x_rmse = calculate_rmse(target_res[:, 0], pred_x)
    y_rmse = calculate_rmse(target_res[:, 1], pred_y)

    # calculate correlation between predicted and actual velocities
    corr_x = np.corrcoef(pred_x, target_res[:, 0])[0, 1]
    corr_y = np.corrcoef(pred_y, target_res[:, 1])[0, 1]

    # store results in a dictionary
    results = {
        "mean_r2": performance,
        "v_x_r2": score_x,
        "v_y_r2": score_y,
        "v_x_rmse": x_rmse,
        "v_y_rmse": y_rmse,
        "v_x_corr": corr_x,
        "v_y_corr": corr_y,
        "predicted_v_x": pred_x,
        "predicted_v_y": pred_y,
    }

    if print_shapes:
        for key, value in results.items():
            print(key, ":", value)
    return results

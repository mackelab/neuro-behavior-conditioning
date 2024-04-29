import os
import numpy as np
import torch
import matplotlib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import pickle
from maskedvae.utils.utils import return_posterior_expectation_and_variance_multi_d
from maskedvae.plotting.plotting_utils import cm2inch

rcParams["grid.linewidth"] = 0
rcParams["pdf.fonttype"] = 42


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def test_visualisations_gauss_one_plot(
    recon_batch,
    recon_var,
    batch_original,
    mask,
    batch,
    mean,
    log_var,
    choice,
    model,
    figsize=(5, 10),
    n_samples=10,
    nbins=4,
    index_1=0,
    index_2=1,
    save_stub="",
):

    """compare the different methods

    1. plot reconstructed means against gt for respective masking mode
    2. plot training and validation losses

    2. plot samples from prior and the reconstructions index_0 against index 1
    3. plot the marginal log likelihood with ellipses for the different modes

    index_1 and index_2 which 2 dimensions should be shown in 2d case always 0 and 1 in n-d case any chosen ones
    """
    warnings.filterwarnings(
        "ignore", message=".*will override the edgecolor or facecolor properties.*"
    )
    warnings.filterwarnings("ignore", message=".*Falling back to DejaVu Sans.*")

    # TODO: adjust for multiple dimensions
    method_short = ["mask-ed-", "zero-", "mask-e-"]
    colors = ["darkred", "midnightblue", "darkgreen"]

    idx_m = [
        "zero_imputation_mask_concatenated",
        "zero_imputation",
        "zero_imputation_mask_concatenated_encoder_only",
    ].index(model.args.method)
    print(idx_m)
    title_list = []
    for ii in range(model.args.x_dim):
        title_list.append(r"$x_{:d}$ masked ".format(ii))
    title_list.append("all obs ")
    title_list.append("random")
    # title_tag = [r'$x_{:d}$ masked '.format(index_1), r'$x_{:d}$ masked '.format(index_2), 'all obs ', 'random' ][choice]
    title_tag = title_list[choice]

    # get the indices where either one of them is masked or all are observed
    x_idx_obs = [bool(a) for a in mask.cpu().data.numpy()[:, index_1]]
    x_idx_masked = [bool(a) for a in 1 - mask.cpu().data.numpy()[:, index_1]]
    y_idx_obs = [bool(a) for a in mask.cpu().data.numpy()[:, index_2]]
    y_idx_masked = [bool(a) for a in 1 - mask.cpu().data.numpy()[:, index_2]]
    all_obs = np.logical_and(x_idx_obs, y_idx_obs)
    all_masked = np.logical_and(x_idx_masked, y_idx_masked)

    # TODO do equivalent that it still works even if trainin not run before
    model.args.min_y = np.min(batch_original.cpu().data.numpy()[:, index_2])
    model.args.min_x = np.min(batch_original.cpu().data.numpy()[:, index_1])
    model.args.max_y = np.max(batch_original.cpu().data.numpy()[:, index_2])
    model.args.max_x = np.max(batch_original.cpu().data.numpy()[:, index_1])

    fig, axs = plt.subplots(9, 2, figsize=cm2inch((20, 50)))
    # ---------------- x reconstructions ---------------------------
    axs[0, 0].plot(
        batch.cpu().data.numpy()[:, index_1],
        recon_batch.cpu().data.numpy()[:, index_1],
        "o",
        color="darkgrey",
        label=r"$x_{:d}$ input".format(index_1),
    )
    axs[0, 0].plot(
        batch_original.cpu().data.numpy()[x_idx_obs, index_1],
        recon_batch.cpu().data.numpy()[x_idx_obs, index_1],
        "o",
        color="midnightblue",
        label=r"$x_{:d}$ obs".format(index_1),
    )
    axs[0, 0].plot(
        batch_original.cpu().data.numpy()[x_idx_masked, index_1],
        recon_batch.cpu().data.numpy()[x_idx_masked, index_1],
        "o",
        color="green",
        label=r"$x_{:d}$ masked".format(index_1),
    )
    axs[0, 0].legend()
    axs[0, 0].plot(
        batch_original.cpu().data.numpy()[:, index_1],
        batch_original.cpu().data.numpy()[:, index_1],
        "k",
    )
    axs[0, 0].set_xlabel("ground truth")
    axs[0, 0].set_ylabel("reconstruction")
    axs[0, 0].locator_params(nbins=nbins)
    # ---------------- x reconstructions ---------------------------

    axs[0, 1].plot(
        batch.cpu().data.numpy()[:, index_2],
        recon_batch.cpu().data.numpy()[:, index_2],
        "o",
        color="darkgrey",
        label=r"$x_{:d}$ input".format(index_2),
    )
    axs[0, 1].plot(
        batch_original.cpu().data.numpy()[y_idx_obs, index_2],
        recon_batch.cpu().data.numpy()[y_idx_obs, index_2],
        "o",
        color="darkred",
        label=r"$x_{:d}$ obs".format(index_2),
    )
    axs[0, 1].plot(
        batch_original.cpu().data.numpy()[y_idx_masked, index_2],
        recon_batch.cpu().data.numpy()[y_idx_masked, index_2],
        "o",
        color="orange",
        label=r"$x_{:d}$ masked".format(index_2),
    )

    axs[0, 1].legend()
    axs[0, 1].plot(
        batch_original.cpu().data.numpy()[:, index_2],
        batch_original.cpu().data.numpy()[:, index_2],
        "k",
    )
    axs[0, 1].set_xlabel("ground truth")
    axs[0, 1].set_ylabel("reconstruction")
    axs[0, 1].locator_params(nbins=nbins)

    # test dataset ---------------------------------------------------------
    dataset = model.test_loader.dataset

    # sample from the prior and pass through trained model
    with torch.no_grad():

        z_sample = torch.randn([batch.shape[0], model.vae.latent_size]).to(
            model.device
        )  # changed here
        x_sample, x_var = model.vae.decode(z_sample.float(), m=mask.float())

    cov_data_f = np.cov(dataset.x.T)
    mu_x_data = np.mean(dataset.x.T[index_1, :])
    mu_y_data = np.mean(dataset.x.T[index_2, :])

    cov_data = np.round(cov_data_f, 3)
    mu_x_data = np.round(mu_x_data, 3)
    mu_y_data = np.round(mu_y_data, 3)
    axs[1, 0].plot(
        dataset.x.T[index_1, : min([1000, dataset.x.shape[0]])],
        dataset.x.T[index_2, : min([1000, dataset.x.shape[0]])],
        "o",
        color="darkgray",
        label="gt",
    )

    axs[1, 0].plot(
        recon_batch.cpu().data.numpy()[:, index_1],
        recon_batch.cpu().data.numpy()[:, index_2],
        "o",
        color="midnightblue",
        label="reconstructions",
    )
    axs[1, 0].plot(
        x_sample.cpu().data.numpy()[:, index_1],
        x_sample.cpu().data.numpy()[:, index_2],
        "o",
        color="darkorange",
        label="from prior",
    )

    axs[1, 0].set_xlabel(r"$x_{:d}$".format(index_1))
    axs[1, 0].set_ylabel(r"$x_{:d}$".format(index_2))
    axs[1, 0].set_ylim([model.args.min_y - 0.5, model.args.max_y + 0.5])
    axs[1, 0].set_xlim([model.args.min_x - 0.5, model.args.max_x + 0.5])
    textstr = "\n".join(
        (
            r"$\mu~gt~x_0=%.2f$" % (mu_x_data,),
            r"$\mu~gt~x_1=%.2f$" % (mu_y_data,),
            r"$gt~cov~=~$" + str(cov_data),
        )
    )
    props = dict(color="white", facecolor="white", alpha=0.5)
    # place a text box in upper left in axes coords
    if model.x_dim < 4:
        axs[1, 0].text(
            0.05,
            0.95,
            textstr,
            transform=axs[1, 0].transAxes,
            verticalalignment="top",
            fontsize=5,
            bbox=props,
        )
    axs[1, 0].legend()

    if model.args.uncertainty:

        if model.args.loss_type == "optimal_sigma_vae":
            x_var = model.out_var_average * torch.ones_like(x_var)
            recon_var = model.out_var_average * torch.ones_like(recon_var)

        x_sample = np.repeat(x_sample.cpu().data.numpy(), n_samples, axis=0)
        x_sigma = np.sqrt(np.repeat(x_var.cpu().data.numpy(), n_samples, axis=0))
        # sample from prior

        # replace all xx_sample with prior_samples[:, index_1]
        prior_samples = np.random.normal(
            loc=x_sample.flatten(), scale=x_sigma.flatten()
        )
        prior_samples = prior_samples.reshape(-1, model.args.x_dim)
        # sample reconstructions
        recon_batch_r = np.repeat(recon_batch.cpu().data.numpy(), n_samples, axis=0)
        recon_sigma_r = np.sqrt(
            np.repeat(recon_var.cpu().data.numpy(), n_samples, axis=0)
        )

        # flatten out to account for all dimensions when sampling
        data_recons = np.random.normal(
            loc=recon_batch_r.flatten(), scale=recon_sigma_r.flatten()
        )
        # bring back into original shape
        data_recons = data_recons.reshape(-1, model.args.x_dim)

        axs[1, 1].plot(
            dataset.x.T[index_1, : min([1000, dataset.x.shape[0]])],
            dataset.x.T[index_2, : min([1000, dataset.x.shape[0]])],
            "o",
            color="darkgray",
            label="gt",
        )

        axs[1, 1].plot(
            data_recons[:, index_1],
            data_recons[:, index_2],
            "o",
            color="midnightblue",
            label="reconstructions",
        )
        axs[1, 1].plot(
            prior_samples[:, index_1],
            prior_samples[:, index_2],
            "o",
            color="darkorange",
            label="from prior",
        )

        axs[1, 1].set_xlabel(r"$x_{:d}$".format(index_1))
        axs[1, 1].set_ylabel(r"$x_{:d}$".format(index_2))
        axs[1, 1].set_ylim([model.args.min_y - 0.5, model.args.max_y + 0.5])
        axs[1, 1].set_xlim([model.args.min_x - 0.5, model.args.max_x + 0.5])

        cov_recon_f_old = np.cov(
            np.stack((data_recons[:, index_1], data_recons[:, index_2]), axis=0)
        )
        cov_recon_f = np.cov(data_recons.T)
        mu_x_recon = np.mean(data_recons[:, index_1])
        mu_y_recon = np.mean(data_recons[:, index_2])
        mu_x_recon = np.round(mu_x_recon, 3)
        mu_y_recon = np.round(mu_y_recon, 3)
        cov_recon = np.round(cov_recon_f, 3)

        # cov_prior_f = np.cov(np.stack((prior_samples[:, index_1], prior_samples[:, index_2]), axis=0))
        cov_prior_f = np.cov(prior_samples.T)
        mu_x_prior = np.mean(prior_samples[:, index_1])
        mu_y_prior = np.mean(prior_samples[:, index_2])
        mu_x_prior = np.round(mu_x_prior, 3)
        mu_y_prior = np.round(mu_y_prior, 3)
        cov_prior = np.round(cov_prior_f, 3)

        textstr = "\n".join(
            (
                r"$\mu~x_0=%.2f$" % (mu_x_recon,),
                r"$\mu~x_1=%.2f$" % (mu_y_recon,),
                r"$cov~=~$" + str(cov_recon),
                r"$\pi~\mu~x_0=%.2f$" % (mu_x_prior,),
                r"$\pi~\mu~x_1=%.2f$" % (mu_y_prior,),
                r"$\pi~cov=~$" + str(cov_prior),
            )
        )
        props = dict(color="white", facecolor="white", alpha=0.5)
        # place a text box in upper left in axes coords
        if model.x_dim < 4:
            axs[1, 1].text(
                0.05,
                0.95,
                textstr,
                transform=axs[1, 1].transAxes,
                verticalalignment="top",
                fontsize=5,
                bbox=props,
            )
        axs[1, 1].legend()

    else:
        axs[1, 1].axis("off")
        cov_prior = np.cov(x_sample.cpu().data.numpy().T)
        mu_x_prior = np.mean(x_sample.cpu().data.numpy()[:, index_1])
        mu_y_prior = np.mean(x_sample.cpu().data.numpy()[:, index_2])
        mu_x_prior = np.round(mu_x_prior, 3)
        mu_y_prior = np.round(mu_y_prior, 3)
        cov_prior = np.round(cov_prior, 3)

        cov_recon = np.cov(recon_batch.cpu().data.numpy().T)
        mu_x_recon = np.mean(recon_batch.cpu().data.numpy()[:, index_1])
        mu_y_recon = np.mean(recon_batch.cpu().data.numpy()[:, index_2])

        cov_recon = np.round(cov_recon, 3)
        mu_x_recon = np.round(mu_x_recon, 3)
        mu_y_recon = np.round(mu_y_recon, 3)

        textstr = "\n".join(
            (
                r"$\mu~x_0=%.2f$" % (mu_x_recon,),
                r"$\mu~x_1=%.2f$" % (mu_y_recon,),
                r"$cov~=~$" + str(cov_recon),
                r"$\pi~\mu~x_0=%.2f$" % (mu_x_prior,),
                r"$\pi~\mu~x_1=%.2f$" % (mu_y_prior,),
                r"$\pi~cov~=~$" + str(cov_prior),
            )
        )
        props = dict(color="white", facecolor="white", alpha=0.5)
        # place a text box in upper left in axes coords
        if model.x_dim < 4:
            axs[1, 1].text(
                0.05,
                0.95,
                textstr,
                transform=axs[1, 1].transAxes,
                verticalalignment="top",
                fontsize=5,
                bbox=props,
            )

    # ---------------------------- losses ---------------------------------------------------------------

    axs[2, 0].plot(model.logs["elbo"], label="training")
    axs[2, 0].plot(
        model.logs["time_stamp_val"], model.logs["elbo_val"], label="validation"
    )
    axs[2, 0].set_xlabel("iteration")
    axs[2, 0].set_ylabel("neg. ELBO")
    axs[2, 0].legend()
    axs[2, 0].locator_params(nbins=nbins)

    axs[2, 1].plot(model.logs["kl"], label="kl train")
    axs[2, 1].plot(
        model.logs["time_stamp_val"], model.logs["kl_val"], label="kl validation"
    )
    axs[2, 1].locator_params(nbins=nbins)
    axs[2, 1].set_xlabel("iteration")
    axs[2, 1].set_ylabel("kl")
    axs[2, 1].legend(frameon=False)

    axs[3, 0].plot(model.logs["elbo"], label="neg. ELBO obs")
    axs[3, 0].plot(model.logs["rec_loss"], label="rec loss obs")
    axs[3, 0].plot(model.logs["masked_rec_loss"], label="rec loss masked")
    axs[3, 0].plot(model.logs["kl"], label="kl")
    axs[3, 0].set_xlabel("iteration")
    axs[3, 0].set_ylabel("loss")
    axs[3, 0].set_title("training")
    axs[3, 0].locator_params(nbins=nbins)
    axs[3, 0].legend(frameon=False)

    axs[3, 1].plot(model.logs["elbo_val"], label="neg. ELBO obs")
    axs[3, 1].plot(model.logs["rec_loss_val"], label="rec loss observed")
    axs[3, 1].plot(model.logs["masked_rec_loss_val"], label="rec loss masked")
    axs[3, 1].plot(model.logs["kl_val"], label="kl")
    axs[3, 1].locator_params(nbins=nbins)
    axs[3, 1].set_xlabel("epoch")
    axs[3, 1].set_ylabel("loss")
    axs[3, 1].set_title("validation")
    axs[3, 1].legend(frameon=False)

    edges_mean = np.linspace(-3.5, 3.5, 150)
    edges_logvar = np.linspace(0, 2, 150)
    # edges_var = np.linspace(0, 1.5, 50)
    VAR_MIN = (
        min([model.args.noise[index_1][0] ** 2, model.args.noise[index_2][0] ** 2])
        - 0.2
    )
    VAR_MAX = (
        max([model.args.noise[index_1][0] ** 2, model.args.noise[index_2][0] ** 2])
        + 0.2
    )
    edges_var = np.linspace(VAR_MIN, VAR_MAX, 50)
    # axs[4, 0].hist(mean.cpu().data.numpy()[:, 0], bins=edges_mean, label='mean 0', color="midnightblue", density=True)
    if model.vae.latent_size > 1:
        axs[4, 0].hist(
            mean.cpu().data.numpy()[:, 1],
            bins=edges_mean,
            label="mean 1",
            color="darkred",
            density=True,
        )
    axs[4, 0].hist(
        model.test_loader.dataset.z[0, :],
        bins=edges_mean,
        color="darkgray",
        density=True,
        label="gt z",
    )
    axs[4, 0].hist(
        mean.cpu().data.numpy()[all_obs, 0],
        bins=edges_mean,
        label="all obs",
        density=True,
    )
    axs[4, 0].hist(
        mean.cpu().data.numpy()[x_idx_masked, 0],
        bins=edges_mean,
        label=r"$x_{:d}$ masked".format(index_1),
        density=True,
    )
    axs[4, 0].hist(
        mean.cpu().data.numpy()[y_idx_masked, 0],
        bins=edges_mean,
        label=r"$x_{:d}$ masked".format(index_2),
        density=True,
    )

    axs[4, 0].set_xlabel("mean")
    axs[4, 0].set_ylabel("frequency")
    axs[4, 0].legend()
    axs[4, 0].locator_params(nbins=nbins)

    var = torch.exp(log_var)

    if model.args.uncertainty:

        # TODO: HERE

        axs[4, 1].hist(
            recon_var.cpu().data.numpy()[x_idx_obs, index_1],
            bins=edges_var,
            label=r"$x_{:d}$ obs".format(index_1),
            density=True,
        )
        axs[4, 1].hist(
            recon_var.cpu().data.numpy()[x_idx_masked, index_1],
            bins=edges_var,
            label=r"$x_{:d}$ masked".format(index_1),
            density=True,
        )

        axs[4, 1].hist(
            recon_var.cpu().data.numpy()[y_idx_obs, index_2],
            bins=edges_var,
            label=r"$x_{:d}$ obs".format(index_2),
            density=True,
        )
        axs[4, 1].hist(
            recon_var.cpu().data.numpy()[y_idx_masked, index_2],
            bins=edges_var,
            label=r"$x_{:d}$ masked".format(index_2),
            density=True,
        )

        axs[4, 1].axvline(
            x=model.args.noise[index_1][0] ** 2,
            label=r"$x_{:d}$".format(index_1),
            color="midnightblue",
        )
        axs[4, 1].axvline(
            x=model.args.noise[index_2][0] ** 2,
            label=r"$x_{:d}$".format(index_2),
            color="darkgreen",
        )

        axs[4, 1].set_xlabel("predicted variance")
        axs[4, 1].set_ylabel("frequency")
        axs[4, 1].legend()
        axs[4, 1].locator_params(nbins=nbins)
    else:
        axs[4, 1].axis("off")

    x0_masked, var_x0_masked = return_posterior_expectation_and_variance_multi_d(
        batch=batch_original[x_idx_masked, :].cpu().data.numpy(),
        args=model.args,
        masked_idx=[
            index_1,
        ],
    )

    x1_masked, var_x1_masked = return_posterior_expectation_and_variance_multi_d(
        batch=batch_original[y_idx_masked, :].cpu().data.numpy(),
        args=model.args,
        masked_idx=[
            index_2,
        ],
    )

    all, var_all = return_posterior_expectation_and_variance_multi_d(
        batch=batch_original[all_obs, :].cpu().data.numpy(),
        args=model.args,
        masked_idx=[],
    )

    if (torch.std(mask, axis=0) == 0).all():  # ensure that all masks are the same here

        if (
            choice == model.n_unique_masks or model.args.n_masked_vals == 0
        ):  # all observed
            masked_idx = []
        else:
            masked_idx = torch.where((mask[0, :] == 0))
            masked_idx = masked_idx[0].cpu().numpy()

        mean_new, var_new = return_posterior_expectation_and_variance_multi_d(
            batch=batch_original.cpu().data.numpy(),
            args=model.args,
            masked_idx=masked_idx,
        )
        axs[5, 0].plot(
            mean.cpu().data.numpy(),
            mean_new,
            "o",
            color="darkgreen",
            ms=3,
            label="high-d",
        )
        # axs[5, 1].plot(var.cpu().data.numpy()[:, 0], var_new*np.ones_like(var.cpu().data.numpy()[:, 0]), 'o', color="C3", ms=3, label='high-d')

        axs[5, 1].hist(
            var.cpu().data.numpy()[:, 0],
            bins=50,
            color="darkgreen",
            label="predicted",
            density=True,
        )

        axs[5, 1].axvline(x=var_new, label="gt", color="C2")
        axs[5, 1].set_xlabel("posterior variance")
        axs[5, 1].set_ylabel("frequency")
        axs[5, 1].legend()
        axs[5, 1].locator_params(nbins=nbins)

        with open(
            os.path.join(
                model.args.fig_root,
                str(model.ts),
                model.method,
                save_stub + "posterior_var_mean" + str(choice) + ".pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(
                [
                    mean.cpu().data.numpy(),
                    mean_new,
                    var.cpu().data.numpy()[:, 0],
                    var_new,
                ],
                f,
            )
        f.close()
    else:

        # axs[6, 0].plot(mean.cpu().data.numpy()[:, 0], model.test_loader.dataset.z[0, :], 'o', ms=3,
        #              color="midnightblue", label="z 0")
        axs[5, 0].plot(
            mean.cpu().data.numpy()[x_idx_masked, 0],
            x0_masked,
            "o",
            color="C1",
            ms=3,
            label=r"$x_{:d}$ masked".format(index_1),
        )
        axs[5, 0].plot(
            mean.cpu().data.numpy()[y_idx_masked, 0],
            x1_masked,
            "o",
            ms=3,
            color="C2",
            label=r"$x_{:d}$ masked".format(index_2),
        )
        axs[5, 0].plot(
            mean.cpu().data.numpy()[all_obs, 0],
            all,
            "o",
            ms=3,
            color="C0",
            label="all obs",
        )

        axs[5, 1].plot(
            var.cpu().data.numpy()[all_obs, 0],
            var_all * np.ones_like(var.cpu().data.numpy()[all_obs, 0]),
            "o",
            ms=3,
            label="all obs",
        )
        axs[5, 1].plot(
            var.cpu().data.numpy()[x_idx_masked, 0],
            var_x0_masked * np.ones_like(var.cpu().data.numpy()[x_idx_masked, 0]),
            "o",
            ms=3,
            label=r"$x_{:d}$ masked".format(index_1),
        )
        axs[5, 1].plot(
            var.cpu().data.numpy()[y_idx_masked, 0],
            var_x1_masked * np.ones_like(var.cpu().data.numpy()[y_idx_masked, 0]),
            "o",
            ms=3,
            label=r"$x_{:d}$ masked".format(index_2),
        )
        axs[5, 1].set_xlabel("variance")
        axs[5, 1].set_ylabel("gt posterior variance")
        axs[5, 1].legend()
        axs[5, 1].locator_params(nbins=nbins)

    axs[5, 0].plot(
        model.test_loader.dataset.z[0, :], model.test_loader.dataset.z[0, :], "k"
    )
    axs[5, 0].set_xlabel("predicted posterior mean")
    axs[5, 0].set_ylabel("gt posterior mean")
    axs[5, 0].legend()
    axs[5, 0].locator_params(nbins=nbins)

    # ----------------- gt posterior variance ----------------------

    if model.vae.latent_size > 1:
        axs[5, 1].hist(
            log_var.cpu().data.numpy()[:, 1],
            bins=edges_logvar,
            color="darkred",
            label="var 1",
            density=True,
        )

    if model.args.uncertainty:

        axs[6, 0].plot(
            dataset.x.T[index_1, : min([1000, dataset.x.shape[0]])],
            dataset.x.T[index_2, : min([1000, dataset.x.shape[0]])],
            "o",
            color="darkgray",
            label="gt",
            ms=0.3,
        )
        axs[6, 0].plot(
            data_recons[:, index_1],
            data_recons[:, index_2],
            "o",
            color="midnightblue",
            label="reconstructions",
            ms=0.3,
        )

        axs[6, 0].set_xlabel(r"$x_{:d}$".format(index_1))
        axs[6, 0].set_ylabel(r"$x_{:d}$".format(index_2))
        axs[6, 0].set_ylim([model.args.min_y - 1, model.args.max_y + 1])
        axs[6, 0].set_xlim([model.args.min_x - 1, model.args.max_x + 1])

        textstr = "\n".join((r"$cov=~$" + str(cov_recon),))
        props = dict(color="white", facecolor="white", alpha=0.5)
        # place a text box in upper left in axes coords
        if model.x_dim < 4:
            axs[6, 0].text(
                0.05,
                0.95,
                textstr,
                transform=axs[6, 0].transAxes,
                verticalalignment="top",
                fontsize=5,
                bbox=props,
            )
        confidence_ellipse(
            dataset.x.T[index_1, :],
            dataset.x.T[index_2, :],
            axs[6, 0],
            n_std=1,
            label="gt",
            edgecolor="k",
            zorder=10e6,
        )
        confidence_ellipse(
            dataset.x.T[index_1, :],
            dataset.x.T[index_2, :],
            axs[6, 0],
            n_std=2,
            edgecolor="k",
            linestyle="--",
            zorder=10e6 + 2,
        )
        confidence_ellipse(
            dataset.x.T[index_1, :],
            dataset.x.T[index_2, :],
            axs[6, 0],
            n_std=3,
            edgecolor="k",
            linestyle=":",
            zorder=10e6 + 3,
        )

        confidence_ellipse(
            data_recons[:, index_1],
            data_recons[:, index_2],
            axs[6, 0],
            n_std=1,
            label="recs.",
            edgecolor="C0",
            zorder=10e6 + 4,
        )
        confidence_ellipse(
            data_recons[:, index_1],
            data_recons[:, index_2],
            axs[6, 0],
            n_std=2,
            edgecolor="C0",
            linestyle="--",
            zorder=10e6 + 5,
        )
        confidence_ellipse(
            data_recons[:, index_1],
            data_recons[:, index_2],
            axs[6, 0],
            n_std=3,
            edgecolor="C0",
            linestyle=":",
            zorder=10e6 + 6,
        )

        axs[6, 0].legend(ncol=2)

        axs[6, 1].plot(
            dataset.x.T[index_1, : min([1000, dataset.x.shape[0]])],
            dataset.x.T[index_2, : min([1000, dataset.x.shape[0]])],
            "o",
            color="darkgray",
            label="gt",
            ms=0.3,
        )
        axs[6, 1].plot(
            prior_samples[:, index_1],
            prior_samples[:, index_2],
            "o",
            color="darkorange",
            label="from prior",
            ms=0.3,
        )

        axs[6, 1].set_xlabel(r"$x_{:d}$".format(index_1))
        axs[6, 1].set_ylabel(r"$x_{:d}$".format(index_2))
        axs[6, 1].set_ylim([model.args.min_y - 1, model.args.max_y + 1])
        axs[6, 1].set_xlim([model.args.min_x - 1, model.args.max_x + 1])

        confidence_ellipse(
            dataset.x.T[index_1, :],
            dataset.x.T[index_2, :],
            axs[6, 1],
            n_std=1,
            label="gt",
            edgecolor="k",
            zorder=10e6 + 6,
        )
        confidence_ellipse(
            dataset.x.T[index_1, :],
            dataset.x.T[index_2, :],
            axs[6, 1],
            n_std=2,
            edgecolor="k",
            linestyle="--",
            zorder=10e6 + 6,
        )
        confidence_ellipse(
            dataset.x.T[index_1, :],
            dataset.x.T[index_2, :],
            axs[6, 1],
            n_std=3,
            edgecolor="k",
            linestyle=":",
            zorder=10e6 + 6,
        )
        confidence_ellipse(
            prior_samples[:, index_1],
            prior_samples[:, index_2],
            axs[6, 1],
            n_std=1,
            label="prior",
            edgecolor="darkred",
            zorder=10e6 + 6,
        )
        confidence_ellipse(
            prior_samples[:, index_1],
            prior_samples[:, index_2],
            axs[6, 1],
            n_std=2,
            edgecolor="darkred",
            linestyle="--",
            zorder=10e6 + 6,
        )
        confidence_ellipse(
            prior_samples[:, index_1],
            prior_samples[:, index_2],
            axs[6, 1],
            n_std=3,
            edgecolor="darkred",
            linestyle=":",
            zorder=10e6 + 6,
        )

        textstr = "\n".join((r"$\pi~cov=~$" + str(cov_prior),))
        props = dict(color="white", facecolor="white", alpha=0.5)
        # place a text box in upper left in axes coords
        if model.x_dim < 4:
            axs[6, 1].text(
                0.05,
                0.95,
                textstr,
                transform=axs[6, 1].transAxes,
                verticalalignment="top",
                fontsize=5,
                bbox=props,
            )
        axs[6, 1].legend(ncol=2)

        min_cov = min(
            [
                min(cov_recon_f.flatten()),
                min(cov_data_f.flatten()),
                min(cov_prior_f.flatten()),
            ]
        )
        max_cov = max(
            [
                max(cov_recon_f.flatten()),
                max(cov_data_f.flatten()),
                max(cov_prior_f.flatten()),
            ]
        )

        # plot covariance matrix
        im1 = axs[7, 1].imshow(cov_recon_f, vmin=min_cov, vmax=max_cov, origin="lower")
        divider = make_axes_locatable(axs[7, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical")
        axs[7, 1].set_title("cov recons")

        im2 = axs[7, 0].imshow(cov_data_f, vmin=min_cov, vmax=max_cov, origin="lower")
        divider = make_axes_locatable(axs[7, 0])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2, orientation="vertical")
        axs[7, 0].set_title("cov data")

        im3 = axs[8, 0].imshow(cov_prior_f, vmin=min_cov, vmax=max_cov, origin="lower")
        divider = make_axes_locatable(axs[8, 0])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2, orientation="vertical")
        axs[8, 0].set_title("cov samples")

    axs[8, 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            model.args.fig_dir,
            "Test"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.pdf".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_root,
            str(model.ts),
            model.method,
            "Test"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.pdf".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_dir,
            "Test"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.png".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_root,
            str(model.ts),
            model.method,
            "Test"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.png".format(
                index_1, index_2
            ),
        )
    )

    plt.clf()
    plt.close("all")

    with open(
        os.path.join(
            model.args.fig_root,
            str(model.ts),
            model.method,
            "samples_"
            + save_stub
            + "_gt_prior_recons_train_"
            + str(choice)
            + "idx1_{:d}_idx2_{:d}.pkl".format(index_1, index_2),
        ),
        "wb",
    ) as f:
        pickle.dump(
            [
                dataset.x.T,
                data_recons[:, index_1],
                data_recons[:, index_2],
                prior_samples[:, index_1],
                prior_samples[:, index_2],
            ],
            f,
        )
    f.close()

    with open(
        os.path.join(
            model.args.fig_root,
            str(model.ts),
            model.method,
            "samples_" + save_stub + "gt_test_recons_" + str(choice) + ".pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(
            [
                dataset.x.T,
                dataset.z[0, :],
                batch.cpu().data.numpy(),
                data_recons,
                prior_samples,
                mean.cpu().data.numpy(),
                mean_new,
                var.cpu().data.numpy()[:, 0],
                var_new,
            ],
            f,
        )
    f.close()

    plt.figure(figsize=cm2inch((20, 20)))

    dff = pd.concat(
        {
            "gt": pd.DataFrame(dataset.x[:300, :5]),
            "input": pd.DataFrame(batch.cpu().data.numpy()[:300, :5]),
            "recs": pd.DataFrame(data_recons[:300, :5]),
        },
        names="T",
    ).reset_index(level=0)
    dff = dff.reset_index()
    dff = dff.drop(columns=["index"])
    g = sns.pairplot(dff, hue="T", markers="o", corner=True, plot_kws={"s": 10})
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            model.args.fig_dir,
            "Test_pairplot_"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.pdf".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_root,
            str(model.ts),
            model.method,
            "Test_pairplot_"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.pdf".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_dir,
            "Test_pairplot_"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.png".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_root,
            str(model.ts),
            model.method,
            "Test_pairplot_"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.png".format(
                index_1, index_2
            ),
        )
    )

    plt.clf()
    plt.close("all")

    plt.figure(figsize=cm2inch((10, 10)))

    dff = pd.concat(
        {
            "gt": pd.DataFrame(dataset.x[:300, :4]),
            #'input': pd.DataFrame(batch.cpu().data.numpy()[:300,:4]),
            "recs": pd.DataFrame(data_recons[:300, :4]),
        },
        names="T",
    ).reset_index(level=0)
    dff = dff.reset_index()
    dff = dff.drop(columns=["index"])
    if model.args.method == "zero_imputation":
        colors = "Blues"
    else:
        colors = "Reds"
    g = sns.pairplot(
        dff, hue="T", markers="o", corner=True, plot_kws={"s": 10}, palette=colors
    )
    plt.tight_layout()
    # handles = g._legend_data.values()
    # labels = g._legend_data.keys()
    # g.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=1)

    plt.savefig(
        os.path.join(
            model.args.fig_dir,
            "Test_pairplot_New_"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.pdf".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_root,
            str(model.ts),
            model.method,
            "Test_pairplot_New_"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.pdf".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_dir,
            "Test_pairplot_New_"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.png".format(
                index_1, index_2
            ),
        )
    )
    plt.savefig(
        os.path.join(
            model.args.fig_root,
            str(model.ts),
            model.method,
            "Test_pairplot_New_"
            + save_stub
            + "_Choice_"
            + str(choice)
            + "_test_visualisations_gauss_idx1_{:d}_idx2_{:d}.png".format(
                index_1, index_2
            ),
        )
    )

    plt.clf()
    plt.close("all")

    if model.args.uncertainty:
        return (
            cov_recon_f,
            cov_prior_f,
            cov_data_f,
        )
    else:
        return -1000.0, -1000.0, cov_data_f


def plot_losses(
    logs,
    methods,
    log="elbo",
    ylabel="neg. elbo",
    xlabel="iteration",
    colors=["darkred", "indigo"],
    meth_labels=["masked", "naive"],
    ax=None,
):
    # plot training and validation losses GLVM logging
    tag = " val" if log[-3:] == "val" else " train"
    if ax is None:
        fig, ax = plt.subplots(figsize=cm2inch((10, 6)))
    for mm, method in enumerate(methods):
        ax.plot(logs[method][log], label=meth_labels[mm] + tag, lw=2, color=colors[mm])
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.locator_params(nbins=4)
    # move axes out
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))

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
        **kwargs,
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

    # get the indices where either one of them is masked or all are observed
    x_idx_obs = [bool(a) for a in mask.cpu().data.numpy()[:, index_1]]
    x_idx_masked = [bool(a) for a in 1 - mask.cpu().data.numpy()[:, index_1]]
    y_idx_obs = [bool(a) for a in mask.cpu().data.numpy()[:, index_2]]
    y_idx_masked = [bool(a) for a in 1 - mask.cpu().data.numpy()[:, index_2]]
    all_obs = np.logical_and(x_idx_obs, y_idx_obs)
    all_masked = np.logical_and(x_idx_masked, y_idx_masked)

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

        cov_recon_f = np.cov(data_recons.T)
        mu_x_recon = np.mean(data_recons[:, index_1])
        mu_y_recon = np.mean(data_recons[:, index_2])
        mu_x_recon = np.round(mu_x_recon, 3)
        mu_y_recon = np.round(mu_y_recon, 3)
        cov_recon = np.round(cov_recon_f, 3)

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


# ----------------- plot monkey --------------------------------------------


def plot_rate_samples_and_spikes(
    session_data,
    xa_m_samples,
    colscmap="Reds",
    axn1=4,
    axn2=4,
    model_tag="naive",
    mask_key="xa_m_last_half",
    t=0,
    zoom_t_low=50,
    zoom_t_high=250,
    fps=15.625,
    exp_dir="./",
    inp_mask=("xa_m",),
):

    # Colormap settings
    import matplotlib

    colour_map = plt.cm.get_cmap(colscmap, axn1 * axn2 + 4)
    clist = [
        matplotlib.colors.rgb2hex(i)
        for i in colour_map(np.linspace(0, 1, axn1 * axn2 + 4))
    ]
    clist.reverse()

    # Create subplots
    fig_trace, axs_trace = plt.subplots(axn1, axn2, figsize=(10, 6), sharey=False)
    axs_trace = axs_trace.flatten()
    time_axis = np.arange(zoom_t_low, zoom_t_high, 1) / fps
    for i in range(axn2 * axn2):
        channel = i * 2
        axs_trace[i].plot(
            time_axis,
            session_data[channel, zoom_t_low:zoom_t_high],
            color="grey",
            label="gt",
        )
        axs_trace[i].plot(
            time_axis,
            xa_m_samples[:, channel, zoom_t_low:zoom_t_high].T,
            color=clist[i],
            alpha=0.3,
        )  # , label=f'rate {rate_ch[channel]} Hz')
        axs_trace[i].set_title(f"channel {channel}")
        axs_trace[i].spines["right"].set_visible(False)
        axs_trace[i].spines["top"].set_visible(False)

    # Labels and titles
    fig_trace.supxlabel("time [s]", fontsize=14)
    fig_trace.supylabel("spikes / rates [a.u.]", fontsize=14)
    fig_trace.suptitle(f"{model_tag} model mask: {mask_key} " + " M1 ", fontsize=14)
    fig_trace.tight_layout()

    # Save plots
    drop_dir = exp_dir
    file_prefix = (
        f"session_{t}_monkey_reach_input_mask_model_tag_{model_tag}_mask_{mask_key}"
    )
    fig_trace.savefig(f"{drop_dir}plot_gt_and_inferred_rates_{file_prefix}.png")
    fig_trace.savefig(f"{drop_dir}plot_gt_and_inferred_rates_{file_prefix}.pdf")


def get_title_dict():
    """Returns a dictionary with monkey title mappings."""
    title_dict = {
        "xa_s": "S1 activity",
        "xa_m": "M1 activity",
        "xb_y": "Finger position",
        "xb_d": "Finger speed",
        "xb_u": "Target",
    }
    return title_dict


def make_col_dict_monkey():
    """make the color dictionary for the monkey reach task"""
    col_dict = {
        mask: {masked: "blue" for masked in [True, False]}
        for mask in ["masked", "naive"]
    }
    col_dict["masked"][True] = "firebrick"
    col_dict["masked"][False] = "lightcoral"
    col_dict["naive"][True] = "royalblue"
    col_dict["naive"][False] = "dodgerblue"
    return col_dict


def plot_training_perf(
    df_perf_col,
    run_properties,
    labeling="run_id",
    runs=np.index_exp[:],
    session="0",
    t_v="Valid",
    typ="Pred",
    metric_a="LogL/T",
    metric_b="CorrCoef",
    figsize=(16, 4),
    ncol=2,
    title_dict=get_title_dict(),
):
    """Performance plotting for monkey reach task"""
    modals = df_perf_col.columns.unique(level="Modality")
    # define the colour palette
    sns_cmap = sns.color_palette(n_colors=len(run_properties[labeling].unique()))
    col_dict = {k: sns_cmap[i] for i, k in enumerate(run_properties[labeling].unique())}
    unqiue_label_inds = np.unique(run_properties[labeling].values, return_index=True)[1]
    # ensure label of run_id is correp to runs even if not all runs taken
    if labeling == "run_id" and runs != np.index_exp[:] and runs != slice(None):
        unqiue_label_inds = np.arange(len(runs))
        print(
            "WARNING: other labels might be off if not sorted by run_id with only a few runs"
        )
    # sort ensure that respective modalities are next to each other
    modality_list = list(modals)
    if "xb_u" in modality_list:
        modality_list.remove("xb_u")
    fig, ax = plt.subplots(1, len(modality_list), figsize=figsize)
    modality_list.sort()
    for i, m in enumerate(modality_list):
        if "xa" in m:
            df_sub = df_perf_col.loc[
                :, pd.IndexSlice[runs, m, int(session), t_v, typ, metric_a]
            ]
        if "xb" in m:
            df_sub = df_perf_col.loc[
                :, pd.IndexSlice[runs, m, int(session), t_v, typ, metric_b]
            ]

        # plot performance metrics
        lines = ax[i].plot(df_sub.index, df_sub.values)
        # ensure labelling and colours only for runs where this metric applies
        rr = list(df_sub.columns.unique(level="Run"))
        lines_legend = lines

        for k, l in enumerate(lines):
            l.set_color(col_dict[run_properties[labeling][rr[k]]])
        # make run labels
        if i == len(modality_list) - 1:
            labels_runs = run_properties[labeling].values[unqiue_label_inds]

            if labeling == "run_id" and runs != np.index_exp[:] and runs != slice(None):
                _idx = [runs[uid] for uid in unqiue_label_inds]
                labels_runs = run_properties[labeling].values[_idx]

            ax[i].legend(
                [lines_legend[u] for u in unqiue_label_inds],
                labels_runs,
                title=labeling,
                loc="best",
                bbox_to_anchor=(1.4, 0.0),
                frameon=True,
                edgecolor="white",
                shadow=False,
                ncol=ncol,
                framealpha=0.8,
                facecolor="white",
            )

        ax[i].set(
            title=title_dict[m] + " " + metric_a
            if "xa" in m
            else title_dict[m] + " " + metric_b
        )
        ax[i].locator_params(tight=True, nbins=4)
        ax[i].set_xlabel("Iteration")
        sns.despine()

    plt.tight_layout()
    return fig


def plot_sample_cumulative_prob_ratio(
    cum_sum_samples,
    cum_sum_gt,
    neuron_id=0,
    ax=None,
    color="darkred",
    alpha=0.1,
    ms=3,
    label=None,
    ifdiag=True,
    title_off=False,
):
    """plot the cumulative sum of the histogram of the spike train samples"""
    assert (
        cum_sum_samples.shape[1] == cum_sum_gt.shape[0]
    ), "number of neurons in samples and ground truth do not match"
    assert (
        cum_sum_samples.shape[-1] == cum_sum_gt.shape[-1]
    ), "number of prob bins in samples and ground truth do not match"

    if ax is None:
        plt.plot(
            cum_sum_gt[neuron_id, :],
            cum_sum_samples[:, neuron_id, :].T,
            marker="o",
            color=color,
            ms=ms,
            alpha=alpha,
            # label=label,
        )
        if label is not None:
            plt.plot([], [], marker="o", color=color, ms=ms, alpha=alpha, label=label)
        if ifdiag:
            plt.plot([0, 1], [0, 1], color="black", ls="--")
    else:
        ax.plot(
            cum_sum_gt[neuron_id, :],
            cum_sum_samples[:, neuron_id, :].T,
            marker="o",
            color=color,
            ms=ms,
            alpha=alpha,
            # label=label,
        )
        if label is not None:
            ax.plot([], [], marker="o", color=color, ms=ms, alpha=alpha, label=label)
        if ifdiag:
            ax.plot([0, 1], [0, 1], color="black", ls="--")
        print(label)
        if not title_off:
            ax.set_title(f"ch {neuron_id}", fontsize=10)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)


def plot_multiple_sample_cumulative_prob_ratios(
    cum_sum_samples,
    cum_sum_gt,
    rate_ch,
    colscmap="Reds",
    axn1=4,
    axn2=4,
    model_tag="naive",
    mask_key="xa_m_last_half",
    t=0,
    exp_dir="./",
    inp_mask=("xa_m",),
):

    # Colormap settings
    import matplotlib

    colour_map = plt.cm.get_cmap(colscmap, axn1 * axn2 + 4)
    clist = [
        matplotlib.colors.rgb2hex(i)
        for i in colour_map(np.linspace(0, 1, axn1 * axn2 + 4))
    ]
    clist.reverse()

    # Create subplots
    fig_trace, axs_trace = plt.subplots(axn1, axn2, figsize=(10, 6), sharey=False)
    axs_trace = axs_trace.flatten()
    for i in range(axn2 * axn2):
        channel = i * 2
        plot_sample_cumulative_prob_ratio(
            cum_sum_samples,
            cum_sum_gt,
            neuron_id=channel,
            ax=axs_trace[i],
            color=clist[i],
            label="rate {0:.2f} Hz".format(rate_ch[channel])
            if rate_ch is not None
            else None,
        )

    # Labels and titles
    fig_trace.supxlabel("gt CDF", fontsize=14)
    fig_trace.supylabel("sample CDF", fontsize=14)
    fig_trace.suptitle(f"{model_tag} model mask: {mask_key} " + " M1 ", fontsize=14)
    fig_trace.tight_layout()

    # Save plots
    drop_dir = exp_dir
    file_prefix = (
        f"session_{t}_monkey_reach_input_mask_model_tag_{model_tag}_mask_{mask_key}"
    )
    fig_trace.savefig(f"{drop_dir}different_fr_cumsum_n_samples_20_{file_prefix}.png")
    fig_trace.savefig(f"{drop_dir}different_fr_cumsum_n_samples_20_{file_prefix}.pdf")


def plot_multiple_sample_cumulative_prob_ratios_small(
    cum_sum_samples,
    cum_sum_gt,
    rate_ch,
    colscmap="Reds",
    axn1=1,
    axn2=3,
    model_tag="naive",
    mask_key="xa_m_last_half",
    t=0,
    exp_dir="./",
    inp_mask=("xa_m",),
    save_dir="./",
):
    """plot only three CDFs of neurons next to one another"""

    # Colormap settings
    import matplotlib

    colour_map = plt.cm.get_cmap(colscmap, axn1 * axn2 + 1)
    clist = [
        matplotlib.colors.rgb2hex(i)
        for i in colour_map(np.linspace(0, 1, axn1 * axn2 + 1))
    ]
    clist.reverse()

    # Create subplots
    # ensure aspect ratio is even in each subplot
    fig_trace, axs_trace = plt.subplots(
        axn1, axn2, figsize=cm2inch((9, 2.5)), sharey=True
    )
    axs_trace = axs_trace.flatten()
    for i in range(axn1 * axn2):
        channel = i * 4
        plot_sample_cumulative_prob_ratio(
            cum_sum_samples,
            cum_sum_gt,
            neuron_id=channel,
            ax=axs_trace[i],
            color=clist[i],
            label="{0:.2f} Hz".format(rate_ch[channel])
            if rate_ch is not None
            else None,
        )
        axs_trace[i].set_xlim(0, 1.05)  # Set x-axis limits from 0 to 1
        axs_trace[i].set_ylim(0, 1.05)  # Set y-axis limits from 0 to 1
        axs_trace[i].set_aspect(
            "equal", "box"
        )  # Set aspect ratio to be equal, making the plot square
        axs_trace[i].legend(frameon=False, loc="lower right", fontsize=8)

    # Labels and titles
    axs_trace[1].set_xlabel("gt CDF")
    axs_trace[0].set_ylabel("sample CDF")

    # Save plots
    drop_dir = exp_dir
    file_prefix = (
        f"session_{t}_monkey_reach_input_mask_model_tag_{model_tag}_mask_{mask_key}"
    )
    fig_trace.savefig(f"{drop_dir}different_fr_cumsum_n_samples_20_{file_prefix}.png")
    fig_trace.savefig(f"{drop_dir}different_fr_cumsum_n_samples_20_{file_prefix}.pdf")
    fig_trace.savefig(f"{save_dir}different_fr_cumsum_n_samples_20_{file_prefix}.png")
    fig_trace.savefig(f"{save_dir}different_fr_cumsum_n_samples_20_{file_prefix}.pdf")


def plot_rate_samples_and_spikes_small(
    session_data,
    xa_m_samples,
    colscmap="Reds",
    axn1=1,
    axn2=3,
    model_tag="naive",
    mask_key="xa_m_last_half",
    t=0,
    zoom_t_low=100,
    zoom_t_high=300,
    fps=15.625,
    exp_dir="./",
    save_dir="./",
):
    """plot only three rate samples and spikes next to one another"""
    # Colormap settings
    import matplotlib

    colour_map = plt.cm.get_cmap(colscmap, axn1 * axn2 + 1)
    clist = [
        matplotlib.colors.rgb2hex(i)
        for i in colour_map(np.linspace(0, 1, axn1 * axn2 + 1))
    ]
    clist.reverse()

    # Create subplots
    fig_trace, axs_trace = plt.subplots(
        axn1, axn2, figsize=cm2inch((10, 2)), sharey=False
    )
    axs_trace = axs_trace.flatten()
    time_axis = np.arange(zoom_t_low, zoom_t_high, 1) / fps
    for i in range(axn1 * axn2):
        channel = i * 4
        axs_trace[i].plot(
            time_axis,
            session_data[channel, zoom_t_low:zoom_t_high],
            color="grey",
            label="gt",
        )
        axs_trace[i].plot(
            time_axis,
            xa_m_samples[:, channel, zoom_t_low:zoom_t_high].T,
            color=clist[i],
            alpha=0.3,
        )  # , label=f'rate {rate_ch[channel]} Hz')
        axs_trace[i].set_title(f"channel {channel}")
        axs_trace[i].spines["right"].set_visible(False)
        axs_trace[i].spines["top"].set_visible(False)

        # move outwards the axes
        axs_trace[i].spines["left"].set_position(("outward", 2))
        axs_trace[i].spines["bottom"].set_position(("outward", 2))

    # Labels and titles
    axs_trace[1].set_xlabel("time [s]")  # , fontsize=14)
    axs_trace[0].set_ylabel("spikes-rates [Hz]")  # , fontsize=14)

    # Save plots
    drop_dir = exp_dir
    file_prefix = (
        f"session_{t}_monkey_reach_input_mask_model_tag_{model_tag}_mask_{mask_key}"
    )
    fig_trace.savefig(f"{drop_dir}plot_gt_and_inferred_rates_{file_prefix}.png")
    fig_trace.savefig(f"{drop_dir}plot_gt_and_inferred_rates_{file_prefix}.pdf")
    fig_trace.savefig(f"{save_dir}plot_gt_and_inferred_rates_{file_prefix}.png")
    fig_trace.savefig(f"{save_dir}plot_gt_and_inferred_rates_{file_prefix}.pdf")


def plot_multiple_sample_cumulative_prob_ratios_masked_and_naive(
    cum_sum_samples_masked,
    cum_sum_gt_masked,
    cum_sum_samples_naive,
    cum_sum_gt_naive,
    rate_ch,
    axn1=2,
    axn2=3,
    mask_key="xa_m",
    t=0,
    exp_dir="./",
    save_dir="./",
    channels=[0, 4, 8],
    fig_size=cm2inch((9, 6)),
):
    """plot only three CDFs of neurons next to one another"""

    # Colormap settings
    import matplotlib

    colour_map_r = plt.cm.get_cmap("Reds", axn2 + 1)
    clist_r = [
        matplotlib.colors.rgb2hex(i) for i in colour_map_r(np.linspace(0, 1, axn2 + 1))
    ]
    clist_r.reverse()
    colour_map_b = plt.cm.get_cmap("Blues", axn2 + 1)
    clist_b = [
        matplotlib.colors.rgb2hex(i) for i in colour_map_b(np.linspace(0, 1, axn2 + 1))
    ]
    clist_b.reverse()

    assert len(channels) == axn2, "number of channels does not match number of subplots"
    # Create subplots
    # ensure aspect ratio is even in each subplot
    fig_trace, axs_trace = plt.subplots(
        axn1, axn2, figsize=fig_size, sharey=True, sharex=True
    )

    # naive on top row
    for i in range(axn2):
        channel = channels[i]
        plot_sample_cumulative_prob_ratio(
            cum_sum_samples_naive,
            cum_sum_gt_naive,
            neuron_id=channel,
            ax=axs_trace[0, i],
            color=clist_b[i],
            label="{0:.2f} Hz".format(rate_ch[channel])
            if rate_ch is not None
            else None,
        )
        axs_trace[0, i].set_xlim(0, 1.05)  # Set x-axis limits from 0 to 1
        axs_trace[0, i].set_ylim(0, 1.05)  # Set y-axis limits from 0 to 1
        axs_trace[0, i].set_aspect(
            "equal", "box"
        )  # Set aspect ratio to be equal, making the plot square
        # move
        axs_trace[0, i].legend(
            frameon=False, loc="lower right", fontsize=8, bbox_to_anchor=(1.2, -0.1)
        )

    # masked on top bottom
    for i in range(axn2):
        channel = channels[i]
        plot_sample_cumulative_prob_ratio(
            cum_sum_samples_masked,
            cum_sum_gt_masked,
            neuron_id=channel,
            ax=axs_trace[1, i],
            color=clist_r[i],
            label="{0:.2f} Hz".format(rate_ch[channel])
            if rate_ch is not None
            else None,
            title_off=True,
        )
        axs_trace[1, i].set_xlim(0, 1.05)  # Set x-axis limits from 0 to 1
        axs_trace[1, i].set_ylim(0, 1.05)  # Set y-axis limits from 0 to 1
        axs_trace[1, i].set_aspect(
            "equal", "box"
        )  # Set aspect ratio to be equal, making the plot square
        axs_trace[1, i].legend(
            frameon=False, loc="lower right", fontsize=8, bbox_to_anchor=(1.2, -0.1)
        )

    # Labels and titles
    axs_trace[1, 1].set_xlabel("gt CDF")
    axs_trace[0, 0].set_ylabel("sample CDF")

    # Save plots
    drop_dir = exp_dir
    file_prefix = f"session_{t}_monkey_reach_input_mask_model_tag_both_naive_and_masked_mask_{mask_key}"
    fig_trace.savefig(f"{drop_dir}different_fr_cumsum_n_samples_20_{file_prefix}.png")
    fig_trace.savefig(f"{drop_dir}different_fr_cumsum_n_samples_20_{file_prefix}.pdf")
    fig_trace.savefig(f"{save_dir}different_fr_cumsum_n_samples_20_{file_prefix}.png")
    fig_trace.savefig(f"{save_dir}different_fr_cumsum_n_samples_20_{file_prefix}.pdf")


def plot_rate_samples_and_spikes_masked_and_naive(
    session_data_masked,
    xa_m_samples_masked,
    session_data_naive,
    xa_m_samples_naive,
    zoom_t_low=100,
    zoom_t_high=300,
    fps=15.625,
    exp_dir="./",
    save_dir="./",
    axn1=2,
    axn2=3,
    mask_key="xa_m",
    t=0,
    channels=[0, 4, 8],
    fig_size=cm2inch((9, 6)),
    meandec=False,
    mean_masked=None,
    mean_naive=None,
):

    """plot only three rate samples and spikes next to one another"""
    # Colormap settings
    import matplotlib

    colour_map_r = plt.cm.get_cmap("Reds", axn2 + 1)
    clist_r = [
        matplotlib.colors.rgb2hex(i) for i in colour_map_r(np.linspace(0, 1, axn2 + 1))
    ]
    clist_r.reverse()
    colour_map_b = plt.cm.get_cmap("Blues", axn2 + 1)
    clist_b = [
        matplotlib.colors.rgb2hex(i) for i in colour_map_b(np.linspace(0, 1, axn2 + 1))
    ]
    clist_b.reverse()

    assert len(channels) == axn2, "number of channels does not match number of subplots"
    # Create subplots
    # ensure aspect ratio is even in each subplot
    fig_trace, axs_trace = plt.subplots(
        axn1, axn2, figsize=fig_size, sharey=False, sharex=True
    )
    time_axis = np.arange(zoom_t_low, zoom_t_high, 1) / fps

    # naive on top row
    for i in range(axn2):
        channel = channels[i]
        axs_trace[0, i].plot(
            time_axis,
            session_data_naive[channel, zoom_t_low:zoom_t_high],
            color="grey",
            label="gt",
        )
        axs_trace[0, i].plot(
            time_axis,
            xa_m_samples_naive[:, channel, zoom_t_low:zoom_t_high].T,
            color=clist_b[i],
            alpha=0.3,
        )  # , label=f'rate {rate_ch[channel]} Hz')
        if meandec and mean_naive is not None:
            axs_trace[0, i].plot(
                time_axis,
                mean_naive[channel, zoom_t_low:zoom_t_high],
                color="black",
                alpha=1,
                label="mean",
            )
        axs_trace[0, i].set_title(f"ch {channel}", fontsize=8)
        axs_trace[0, i].spines["right"].set_visible(False)
        axs_trace[0, i].spines["top"].set_visible(False)

        # move outwards the axes
        axs_trace[0, i].spines["left"].set_position(("outward", 2))
        axs_trace[0, i].spines["bottom"].set_position(("outward", 2))

    # masked on bottom row
    for i in range(axn2):
        channel = channels[i]
        axs_trace[1, i].plot(
            time_axis,
            session_data_masked[channel, zoom_t_low:zoom_t_high],
            color="grey",
            label="gt",
        )
        axs_trace[1, i].plot(
            time_axis,
            xa_m_samples_masked[:, channel, zoom_t_low:zoom_t_high].T,
            color=clist_r[i],
            alpha=0.3,
        )  # , label=f'rate {rate_ch[channel]} Hz')
        if meandec and mean_masked is not None:
            axs_trace[1, i].plot(
                time_axis,
                mean_masked[channel, zoom_t_low:zoom_t_high],
                color="black",
                alpha=1,
                label="mean",
            )
        # axs_trace[1,i].set_title(f'ch {channel}', fontsize=8)
        axs_trace[1, i].spines["right"].set_visible(False)
        axs_trace[1, i].spines["top"].set_visible(False)

        # move outwards the axes
        axs_trace[1, i].spines["left"].set_position(("outward", 2))
        axs_trace[1, i].spines["bottom"].set_position(("outward", 2))
    # Labels and titles
    axs_trace[1, 1].set_xlabel("time [s]")  # , fontsize=14)
    axs_trace[1, 0].set_ylabel("spikes-rates [Hz]")  # , fontsize=14)

    # Save plots
    drop_dir = exp_dir
    file_prefix = f"session_{t}_monkey_reach_input_mask_model_tag_masked_and_naive_mask_{mask_key}"
    fig_trace.savefig(f"{drop_dir}plot_gt_and_inferred_rates_{file_prefix}.png")
    fig_trace.savefig(f"{drop_dir}plot_gt_and_inferred_rates_{file_prefix}.pdf")
    fig_trace.savefig(f"{save_dir}plot_gt_and_inferred_rates_{file_prefix}.png")
    fig_trace.savefig(f"{save_dir}plot_gt_and_inferred_rates_{file_prefix}.pdf")


def scatter_plot_with_equal_aspect(
    x,
    y,
    figsize=cm2inch((10, 10)),
    color="darkblue",
    ms=1,
    markerstring="o",
    xlabel="X-axis",
    ylabel="Y-axis",
    title="Scatter Plot with Equal Aspect Ratio and Same Ticks",
    savestr=None,
    plot_diagonal=True,
):
    """
    Create a scatter plot with equal aspect ratio and same ticks on both axes.

    Args:
        x (array-like): x-coordinates of the data points.
        y (array-like): y-coordinates of the data points.
        color (str): Marker color.
        ms (float): Marker size.
        markerstring (str): Marker style string (e.g., 'o', 's', '^', etc.).
    """
    plt.figure(figsize=figsize)

    # Create a scatter plot
    plt.scatter(x, y, color=color, s=ms, marker=markerstring)

    # Set the aspect ratio of the plot to be equal
    plt.gca().set_aspect("equal", adjustable="box")

    # Calculate the range of data for both axes
    data_range = max(max(x) - min(x), max(y) - min(y))
    max_val = max(max(x), max(y))
    min_val = min(min(x), min(y))

    # set the same axis limits
    plt.xlim([min_val - 0.05 * data_range, max_val + 0.05 * data_range])
    plt.ylim([min_val - 0.05 * data_range, max_val + 0.05 * data_range])

    plt.locator_params(nbins=5)
    if plot_diagonal:
        # plot diagonal that extends to the max of the two
        plt.plot([min_val, max_val], [min_val, max_val], color="k")

    # Add labels and title (optional)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if savestr is not None:
        plt.savefig(savestr + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(savestr + ".pdf", dpi=300, bbox_inches="tight")


def plot_position_traces_ax(
    ax,
    session_xb_y_gt,
    q,
    color_x="olive",
    color_y="darkgreen",
    t_min=100,
    t_max=400,
    time=None,
    iflegend=False,
    ncols=1,
    fps=15.625,
):
    """plot position traces for a given session and time window"""
    if time is None:
        print(f"time is None using {t_min} and {t_max} and fps {fps}")
        time = np.arange(t_min, t_max, 1) / fps

    ax.plot(
        time[: t_max - t_min],
        session_xb_y_gt[q][0, 0, t_min:t_max],
        color=color_x,
        label="x",
    )
    ax.plot(
        time[: t_max - t_min],
        session_xb_y_gt[q][0, 1, t_min:t_max],
        color=color_y,
        label="y",
    )

    if iflegend:
        ax.legend(fontsize=6, frameon=False, ncols=ncols)


def plot_multiarr_hist_ax_samples(
    ax,
    session_neuro_gt,
    all_neuro,
    model_tags,
    q_m=0,
    q_n=-1,
    mask_key="xa_m",
    t=0,
    bins=np.arange(7),
    colors=["grey", "midnightblue", "darkred"],
    labels=["gt", "naive", "masked"],
    local_dir=None,
    save_name=None,
):
    """plot a histogram of multiple samples of the test set rates
    histogram of spikes for gt, naive and masked next to one another
    """
    # Define bins where each integer corresponds to one bin
    arr1 = session_neuro_gt[t][0, :, :].flatten()
    arr2 = np.random.poisson(
        np.array(all_neuro[q_n][model_tags[q_n]][mask_key][t])[:, :, :]
    ).flatten()  # naive
    arr3 = np.random.poisson(
        np.array(all_neuro[q_m][model_tags[q_m]][mask_key][t])[:, :, :]
    ).flatten()  # masked
    # Plot the three histograms together such that all bars for bin 0 are next to each other and then bin 1, bin 2 etc.
    ax.hist(
        (arr1, arr2, arr3),
        bins=bins,
        histtype="bar",
        stacked=False,
        color=colors,
        label=labels,
        density=True,
    )

    ax.legend()
    ax.set_xticks(bins + 0.5)
    ax.set_xticklabels(list(bins))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))
    if local_dir is not None:
        save_name = (
            f"spike_count_dist_all_three_next_{mask_key}_dark_samll_{q_m}_{q_n}"
            if save_name is None
            else save_name
        )
        plt.savefig(f"{local_dir}/{save_name}.pdf", bbox_inches="tight")
        plt.savefig(f"{local_dir}/{save_name}.png", bbox_inches="tight")


def plot_multiarr_hist_ax(
    ax,
    session_neuro_gt,
    all_mean_decoding,
    model_tags,
    q_m=0,
    q_n=-1,
    mask_key="xa_m",
    t=0,
    bins=np.arange(7),
    colors=["grey", "darkred", "midnightblue"],
    labels=["gt", "masked", "naive"],
    local_dir=None,
    save_name=None,
):
    """plot a histogram of multiple arrays
    Careful here the order of the arrays is different than in the function above"""
    np.random.seed(0)
    # Define bins where each integer corresponds to one bin
    arr1 = session_neuro_gt[t][0, :, :].flatten()
    arr2 = np.random.poisson(
        np.array(all_mean_decoding[q_m]["xa_m"][model_tags[q_m]][mask_key][t])[0, :, :]
    ).flatten()  # masked
    arr3 = np.random.poisson(
        np.array(all_mean_decoding[q_n]["xa_m"][model_tags[q_n]][mask_key][t])[0, :, :]
    ).flatten()  # naive
    # Plot the three histograms together such that all bars for bin 0 are next to each other and then bin 1, bin 2 etc.
    ax.hist(
        (arr1, arr2, arr3),
        bins=bins,
        histtype="bar",
        stacked=False,
        color=colors,
        label=labels,
        density=True,
    )

    ax.legend()
    # set the ticks at bins + 0.5
    ax.set_xticks(bins + 0.5)
    ax.set_xticklabels(list(bins))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))

    if local_dir is not None:
        save_name = (
            f"spike_count_dist_all_three_next_{mask_key}_dark_samll_{q_m}_{q_n}"
            if save_name is None
            else save_name
        )
        plt.savefig(f"{local_dir}/{save_name}.pdf", bbox_inches="tight")
        plt.savefig(f"{local_dir}/{save_name}.png", bbox_inches="tight")


def boxplot_masked_naive_ax(
    ax,
    masked_arr,
    naive_arr,
    ylabel="RMSE \npop. avg [Hz]",
    col_masked="darkred",
    col_naive="midnightblue",
    label_masked="masked",
    label_naive="naive",
    width=0.5,
    ifmean=False,
    meanaxis=1,
):
    """make a boxplot for masked and naive models
    average over the meanaxis if ifmean is True
    """
    if ifmean:
        masked_arr = np.mean(masked_arr, axis=meanaxis)
        naive_arr = np.mean(naive_arr, axis=meanaxis)

    sns.boxplot(
        data=[masked_arr, naive_arr],
        palette=[col_masked, col_naive],
        width=width,
        ax=ax,
    )
    jitter_m = np.random.normal(0, width / 20, size=len(masked_arr))
    jitter_n = np.random.normal(1, width / 20, size=len(naive_arr))
    ax.plot(jitter_m, masked_arr, ".", color="grey", ms=1, zorder=1e10)
    ax.plot(jitter_n, naive_arr, ".", color="grey", ms=1, zorder=1e10)

    ax.set_ylabel(ylabel)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mask", "naiv"])
    ax.locator_params(axis="y", nbins=3)


def plot_sample_cumulative_prob_ratio_neuron_ax(
    ax,
    cum_sum_samples,
    cum_sum_gt,
    color="darkred",
    alpha=0.1,
    ms=3,
    label=None,
    ifdiag=True,
    title_off=False,
    neuron_id=0,
):
    """plot the cumulative sum of the histogram of the spike train samples for a given neuron"""
    assert (
        cum_sum_samples.shape[-1] == cum_sum_gt.shape[-1]
    ), "number of prob bins in samples and ground truth do not match"

    ax.plot(
        cum_sum_gt, cum_sum_samples[:, :].T, marker="o", color=color, ms=ms, alpha=alpha
    )
    if label is not None:
        ax.plot([], [], marker="o", color=color, ms=ms, alpha=alpha, label=label)
    if ifdiag:
        ax.plot([0, 1], [0, 1], color="black", ls="--")
    print(label)
    if not title_off:
        ax.set_title(f"ch {neuron_id}", fontsize=10)


def plot_spikes_mean_and_samples_rates_ax(
    axs,
    session_neuro_gt,
    all_mean_decoding,
    xa_m_samples,
    q,
    t_min=100,
    t_max=500,
    session_=0,
    neuron_id=0,
    mask_key="xa_m",
    spike_offset=0.5,
    spikelen=1,
    col_dict=make_col_dict_monkey(),
    model_tags=None,
    time=None,
    fps=15.625,
    local_dir=None,
    sample_step=1,
    axeslables=False,
    t=0,
    external_offset=None,
):
    """plot mean and samples of rates and spikes for a given neuron and mask_key

    Args:
        session_neuro_gt: list of ground truth spike counts for each session
        all_mean_decoding: list of mean decoding for each session
        xa_m_samples: list of rate samples for each session
        q: index of the model run
        t_min: start time of the plot
        t_max: end time of the plot
        session_: session id
        neuron_id: neuron id
        mask_key: mask key
        spike_offset: offset of the spikes in the plot
        spikelen: length of the spikes in the plot
    """
    extra_col = {"naive": "midnightblue", "masked": "darkred"}
    if time is None:
        print(f"time is None using {t_min} and {t_max} and fps {fps}")
        time = np.arange(t_min, t_max, 1) / fps

    assert model_tags is not None, "model tags are not defined"

    # plot spikes of ground truth as raster spike plot
    # get indices of spikes
    spikes = np.where(session_neuro_gt[q][0, neuron_id, t_min:t_max] > 0)[0]

    # get time of spikes
    spikes_time = time[spikes]
    # get spike count
    spikes_count = session_neuro_gt[q][0, neuron_id, t_min:t_max][spikes]
    offset = (
        max(
            fps
            * np.array(xa_m_samples[model_tags[q]][mask_key][session_])[
                :, neuron_id, t_min:t_max
            ].flatten()
        )
        + spike_offset
    )
    if external_offset is not None:
        offset = max([external_offset, offset])
    # plot the spikes
    for i in range(len(spikes)):
        axs.plot(
            [spikes_time[i], spikes_time[i]],
            [offset, offset + spikelen],
            color="black",
            lw=spikes_count[i] * 0.5,
            zorder=1,
            solid_capstyle="butt",
        )

    axs.plot(
        time[: t_max - t_min],
        fps
        * np.array(xa_m_samples[model_tags[q]][mask_key][session_])[
            ::sample_step, neuron_id, t_min:t_max
        ].T,
        alpha=0.2,
        color=col_dict[model_tags[q]][mask_key != "all_obs"],
        label="samples",
        zorder=11,
    )
    axs.plot(
        time[: t_max - t_min],
        fps
        * np.array(all_mean_decoding[q]["xa_m"][model_tags[q]][mask_key][t])[
            0, neuron_id, t_min:t_max
        ],
        alpha=1,
        color=extra_col[model_tags[q]],
        label="mean",
        zorder=100,
    )

    if axeslables == True:
        axs.set_xlabel("time [s]")
        axs.set_ylabel(f"predicted rate [Hz]")
        axs.set_title(f"{mask_key} - {model_tags[q]} {q} - neuro {neuron_id}")

    axs.spines["left"].set_position(("outward", 4))
    axs.spines["bottom"].set_position(("outward", 4))
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    # only take unique labels
    handles, labels = plt.gca().get_legend_handles_labels()

    if local_dir is not None:
        file_name = f"samples_rates_{neuron_id}_{mask_key}_{q}"
        plt.savefig(f"{local_dir}/figures/{file_name}.pdf", bbox_inches="tight")
        plt.savefig(f"{local_dir}/figures/{file_name}.png", bbox_inches="tight")


def plot_bits_per_spike_per_neuron_ax(
    ax,
    naive_bitsperspike_neuron,
    masked_bitsperspike_neuron,
    rate_ch,
    n_neur=213,
    min_Hz=0.0,
    ylabel="bits per spike",
):
    """plot the bits per spike per neuron for masked and naive model
        or any other neuron wise metric of masked and naive conditions
        in a boxplot with jittered data points

    Args:
        ax: axis to plot on
        naive_bitsperspike_neuron : bits per spike for naive model
        masked_bitsperspike_neuron :  bits per spike for masked model
        rate_ch : array of test time firing rates for each neuron
        n_neur (int, optional): number of neurons. Defaults to 213.
        min_Hz (float, optional): minimum firing rate neurons to be plotted. Defaults to 0.5.
    """

    sns.boxplot(
        [
            np.mean(masked_bitsperspike_neuron, axis=0),
            np.mean(naive_bitsperspike_neuron, axis=0),
        ],
        ax=ax,
        palette=["darkred", "midnightblue"],
        width=0.5,
        fliersize=0,
        zorder=1,
    )
    jitter_n = np.random.normal(1, 0.05, size=n_neur)
    jitter_m = np.random.normal(0, 0.05, size=n_neur)
    ax.plot(
        jitter_n,
        np.mean(naive_bitsperspike_neuron, axis=0),
        ".",
        ms=0.5,
        alpha=0.5,
        color="grey",
        zorder=10,
    )
    ax.plot(
        jitter_m,
        np.mean(masked_bitsperspike_neuron, axis=0),
        ".",
        ms=0.5,
        alpha=0.5,
        color="grey",
        zorder=10,
    )

    # plot a connection line between each neuron
    for i in range(n_neur):
        if rate_ch[i] > min_Hz:
            ax.plot(
                [jitter_n[i], jitter_m[i]],
                [
                    np.mean(naive_bitsperspike_neuron, axis=0)[i],
                    np.mean(masked_bitsperspike_neuron, axis=0)[i],
                ],
                color="grey",
                alpha=0.2,
                zorder=0,
                lw=0.4,
            )
    # swtich off the left and top axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_position(("outward", 4))
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mask", "naiv"])
    ax.locator_params(axis="y", nbins=3)

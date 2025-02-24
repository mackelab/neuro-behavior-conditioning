import torch
from torch import nn
import numpy as np


def masked_GNLL_loss_fn(
    recon_x,
    recon_var,
    x,
    mean,
    log_var,
    mask,
    eps=1e-06,
    beta=1,
    full=False,
):
    """Calculate the Guassian negative log likelihood for a
    Gaussian with mean recon_x and variance recon_var
    Consider only observed (not masked) values unless
    explicitly stated otherwise with full=True

    Parameters
    ----------
        x : tensor with ground truth to both estimate the sigma in optimal-sigma-vae and calc masked and unmasked loss
        recon_x : reconstructed mean
        recon_var : tensor predicted variances decoder
        mean: tensor with predicted approx posterior means
        log_var: log variance of the approx posterior
        mask: mask tensor of 0s and 1s
        eps: epsilon for numerical stability
        beta: beta for KL divergence
        full: bool for full loss masked and observed
    Returns
        total loss GNLL, unobserved loss, KL divergence, unobserved loss, mean variance

    Note: Learning the variance can become unstable in some cases.
    Softly limiting log_sigma to a minimum of -6
    ensures stable training. eps in Gaussian NLL ensures
    that training is stable np.exp(-6)**2 roughly 6e-6 so std.
    implemented one is sufficient"""

    batch_size = x.size(0)

    loss = torch.nn.GaussianNLLLoss(eps=eps, reduction="none")

    GNLL_full = loss(recon_x, x, recon_var)

    # observed reconstruction loss mask 1 for observed 0 for unobserved
    GNLL = mask * GNLL_full
    GNLL = torch.sum(GNLL)

    # unobserved reconstruction loss
    GNLL_unobs = (1 - mask) * GNLL_full  # not mask -> unobserved
    GNLL_unobs = torch.sum(GNLL_unobs)

    # KL divergence between approximate posterior and standard Gaussian N(0,1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    if full:
        # use the entire loss masked and observed
        return (
            (torch.sum(GNLL_full) + beta * KLD) / batch_size,
            GNLL / batch_size,
            KLD / batch_size,
            GNLL_unobs / batch_size,
            torch.mean(recon_var, dim=0),
        )
    else:
        return (
            (GNLL + beta * KLD) / batch_size,
            GNLL / batch_size,
            KLD / batch_size,
            GNLL_unobs / batch_size,
            torch.mean(recon_var, dim=0),
        )


def masked_MSE_loss_fn(
    recon_x,
    recon_var,
    x,
    mean,
    log_var,
    mask,
    eps=1e-06,
    beta=1,
    full=False,
):
    """Calculate the Guassian negative log likelihood for a Gaussian
    with mean recon_x and variance recon_var
    Consider only observed (not masked) values unless
    explicitly stated otherwise with full=True

    Parameters
    ----------
        x : tensor with ground truth to both estimate the sigma
            in optimal-sigma-vae and calc masked and unmasked loss
        recon_x : reconstructed mean
        recon_var : tensor predicted variances decoder
        mean: tensor with predicted approx posterior means
        log_var: log variance of the approx posterior
        mask: mask tensor of 0s and 1s
        eps: epsilon for numerical stability
        beta: beta for KL divergence
        full: bool for full loss masked and observed
    Returns
        total loss GNLL, unobserved loss, KL divergence,
        unobserved loss, mean variance

    Note: Learning the variance can become unstable in some cases.
    Softly limiting log_sigma to a minimum of -6
    ensures stable training. eps in Gaussian NLL ensures
    that training is stable np.exp(-6)**2 roughly 6e-6 so std.
    implemented one is sufficient"""

    batch_size = x.size(0)

    GNLL_full = torch.nn.functional.mse_loss(recon_x, x, reduction="none")

    # observed reconstruction loss mask 1 for observed 0 for unobserved
    GNLL = mask * GNLL_full
    GNLL = torch.sum(GNLL)

    # unobserved reconstruction loss
    GNLL_unobs = (1 - mask) * GNLL_full  # not mask -> unobserved
    GNLL_unobs = torch.sum(GNLL_unobs)

    # KL divergence between approximate posterior and standard Gaussian N(0,1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    if full:
        # use the entire loss masked and observed
        return (
            (torch.sum(GNLL_full) + beta * KLD) / batch_size,
            GNLL / batch_size,
            KLD / batch_size,
            GNLL_unobs / batch_size,
            torch.mean(recon_var, dim=0),
        )
    else:
        return (
            (GNLL + beta * KLD) / batch_size,
            GNLL / batch_size,
            KLD / batch_size,
            GNLL_unobs / batch_size,
            torch.mean(recon_var, dim=0),
        )


# ----------------------- monkey loss -----------------------


def eval_RAE_z_reg(Z):
    """regularised autoencoder
    reg through sum of the square of Z"""
    return (Z**2).sum(-1)  # row sum


def eval_VAE_prior(Z_mu, Z_lsig, p_mu=0.0, p_lsig=np.log(0.1)):
    """

    Args:
        Z_mu: mean of the latents
        Z_lsig:
        p_mu:
        p_lsig:

    Returns:

    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if not torch.is_tensor(p_lsig):
        p_lsig = torch.tensor(p_lsig).to(device)
    if not torch.is_tensor(p_mu):
        p_mu = torch.tensor(p_mu).to(device)

    prior = 0.5 * (
        -1
        + 2 * (p_lsig - Z_lsig)
        + ((Z_mu - p_mu) / torch.exp(p_lsig)) ** 2
        + torch.exp(2 * Z_lsig - 2 * p_lsig)
    )

    return prior


def rec_loss(
    x, x_tilde, loss="poisson", warmup=0, obs_var=None, eps=1e-06, nll_beta=None
):
    """reconstruction loss

    Args:
        x: actual input
        x_tilde: reconstructed input (model output)
        loss: type of loss as string poisson mse or bernoulli
        warmup: number of warmup steps where loss is not cumputed (because of RNN)
        obs_var: observation noise variance

    Returns:
        loss per neuron per batch (summed over time)

    """
    if loss == "poisson":
        loss = nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")
    elif loss == "bernoulli":
        x = torch.clamp(x, 0, 1)
        loss = nn.BCEWithLogitsLoss(reduction="none")
    elif loss == "mse":
        loss = nn.SmoothL1Loss(reduction="none")
        # Creates a criterion that uses a squared term if the absolute
        #  element-wise error falls below beta and an L1 term otherwise.
        #  It is less sensitive to outliers than torch.nn.MSELoss and in
        #  some cases prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
    elif loss == "gnll" and obs_var is not None and nll_beta is not None:
        # Compute beta-NLL loss
        # :param mean: Predicted mean of shape B x D
        # :param variance: Predicted variance of shape B x D
        # :param target: Target of shape B x D
        # :param beta: Parameter from range [0, 1] controlling relative
        #     weighting between data points, where `0` corresponds to
        #     high weight on low error points and `1` to an equal weighting.
        # :returns: Loss per batch element of shape B

        # from This work was done by Maximilian Seitzer, Arash Tavakoli,
        # Dimitrije Antic, Georg Martius at the Autonomous Learning Group,
        # Max-Planck Institute for Intelligent Systems in Tübingen.

        # 0 falls back to Gaussian NLL and 1 to MSE

        loss = 0.5 * ((x - x_tilde) ** 2 / obs_var + obs_var.log())

        if nll_beta > 0:
            loss = loss * (obs_var.detach() ** nll_beta)

        return loss[..., warmup:].sum(axis=-1)

    elif loss == "gnll" and obs_var is not None:
        loss = nn.GaussianNLLLoss(eps=eps, reduction="none")
        loss_ = loss(x_tilde, x, obs_var)  # variance is given as input
        return loss_[..., warmup:].sum(-1)
    else:
        loss = nn.SmoothL1Loss(reduction="none")
    # summed over time
    return loss(x_tilde, x)[..., warmup:].sum(-1)


def masked_rec_loss(
    x,
    x_tilde,
    loss="poisson",
    mask=None,
    warmup=0,
    obs_var=None,
    reduction="sum",
    nll_beta=None,
):
    """masked reconstruction loss

    Args:
         x: actual input
         x_tilde: reconstructed input (model output)
         loss: type of loss as string poisson mse or bernoulli
         mask: mask of zeros and ones
         warmup: number of warmup steps where loss is not cumputed (because of RNN)
         obs_var: observation noise variance

     Returns:

    """
    x_rec_loss = rec_loss(x, x_tilde, loss, warmup, obs_var=obs_var, nll_beta=nll_beta)

    # only calculate loss for non masked entries
    if mask is not None:
        x_rec_loss = x_rec_loss[
            :, mask.nonzero()[:, 0]
        ]  # the returned shape is (len,1) -> 0
    if reduction == "none":
        return x_rec_loss
    else:
        return x_rec_loss.sum(-1)  # Sum over output channels


def beta_nll_loss(mean, variance, target, beta=0.5):
    """Compute beta-NLL loss

    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
        weighting between data points, where `0` corresponds to
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B

    from This work was done by Maximilian Seitzer, Arash Tavakoli,
    Dimitrije Antic, Georg Martius at the Autonomous Learning Group,
    Max-Planck Institute for Intelligent Systems in Tübingen.
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)

    return loss.sum(axis=-1)


def main():

    recon_x = torch.randn(3, 2)
    recon_var = torch.randn(3, 2) + 2
    x = torch.randn(3, 2)
    mean = torch.randn(3, 2)
    log_var = torch.randn(3, 2)
    mask = torch.ones(3, 2)

    gnll = masked_GNLL_loss_fn(recon_x, recon_var, x, mean, log_var, mask)
    mse = masked_MSE_loss_fn(
        recon_x, torch.ones_like(recon_var), x, mean, log_var, mask
    )
    gnll_ones = masked_GNLL_loss_fn(
        recon_x, torch.ones_like(recon_var), x, mean, log_var, mask
    )

    print(gnll[1], gnll_ones[1], mse[1])


if __name__ == "__main__":
    main()

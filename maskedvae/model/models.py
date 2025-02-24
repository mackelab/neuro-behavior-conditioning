import time
import sys
import os
import pickle

import copy
from copy import deepcopy
import wandb
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import wandb

from maskedvae.utils.loss import (
    masked_GNLL_loss_fn,
    masked_MSE_loss_fn,
    masked_rec_loss,
    eval_RAE_z_reg,
    eval_VAE_prior,
)
from maskedvae.model.masks import MultipleDimGauss
from maskedvae.model.networks import GLVM_VAE, AE_model
from maskedvae.utils.utils import (
    ensure_directory,
    reverse_non_all_obs_mask,
    return_posterior_expectation_and_variance_multi_d,
    cpu,
    get_mask,
)
from maskedvae.plotting.plotting import test_visualisations_gauss_one_plot
from maskedvae.utils.evaluate import run_and_eval


class ModelGLVM(object):
    def __init__(
        self,
        args,
        dataset_train,
        dataset_valid,
        dataset_test,
        logs,
        inference_model=GLVM_VAE,
        device="cpu",
        Generator=MultipleDimGauss,
        nonlin=nn.Identity(),
        dropout=0.0,
    ):

        """
        Masked VAE training for the GLVM dataset

        This class is used to train a masked VAE on the GLVM dataset.
        The class is initialized with a set of parameters
        for the network, the chosen dataset, the inference network model and the training device.
        The class contains the main training loop .fit() and evaluation functions for the model.

        Parameters
        ----------
        args: dict
            All parameters that should be changed / added to the default attributes of the network class.

        dataset_train: dataset
            Training dataset

        dataset_valid: dataset
            Validation dataset

        dataset_test: dataset
            Test dataset

        logs: dict
            Dictionary to store the training logs

        inference_model: nn.Module
            Inference network with stanard VAE as default

        args: device
            cpu if no gpu detected

        Generator: nn.Module
            Mask generator class

        nonlin: nn.Module
            Nonlinearity for the network

        dropout: float
            Dropout rate for the network


        ----------

        """
        self.args = args
        self.device = device
        self.dropout = dropout
        print(self.args)

        # set up dataloaders
        self.train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=False,
        )

        self.validation_loader = DataLoader(
            dataset=dataset_valid,
            batch_size=self.args.valid_batch_size,
            shuffle=True,
            pin_memory=False,
        )

        self.test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            pin_memory=False,
        )

        # additional testloader for importance weighting
        self.test_loader_one = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
        )

        # beta parameter for the KL term
        self.beta = args.beta
        self.beta_copy = copy.deepcopy(self.beta)
        self.args.betastep = self.beta / self.args.warmup_range

        # logging of error metrics elbo, kl and reconstruction loss,
        self.logs = logs

        # select current method
        self.method = args.method

        # Imputation of masked values (zero, mean, random, val)
        self.one_impute = self.args.one_impute
        self.mean_impute = self.args.mean_impute and not self.one_impute
        self.val_impute = self.args.val_impute and not (
            self.one_impute or self.mean_impute
        )
        self.random_impute = self.args.random_impute and not (
            self.one_impute or self.mean_impute or self.val_impute
        )

        self.x_dim = self.args.x_dim
        self.n_samples = self.args.n_samples
        self.nonlin = nonlin

        # specify the loss gnll vs mse
        # self.loss_fn = masked_gaussian_logprob_loss_fn if self.args.uncertainty else masked_mse_loss_fn
        self.loss_fn = (
            masked_GNLL_loss_fn if self.args.uncertainty else masked_MSE_loss_fn
        )
        print(
            "Gaussian Neg. log lik loss"
            if self.args.uncertainty
            else "Standard MSE loss"
        )

        # Here we compare three methods zero imputation

        # methods to be compared
        # 1. just set unobserved values to zero and don't provide info which point is missing which is a zero data val
        # 2. set unobserved values to zero and shift observed values by a fixed baseline (zero data val no longer exists)
        # 3. set unobserved values to zero but concatenate the mask with the input image in the decoder

        if self.method == "zero_imputation_baselined":
            self.baselined = True  # add baseline to all observed
            self.args.pass_mask = False  # pass mask to the encoder and decoder
            self.args.pass_mask_decoder = self.args.pass_mask
            self.impute_missing = False

        if self.method == "zero_imputation":
            self.args.pass_mask = False
            self.args.pass_mask_decoder = self.args.pass_mask
            self.baselined = False
            self.impute_missing = False

        if self.method == "zero_imputation_mask_concatenated":
            self.args.pass_mask = True
            self.args.pass_mask_decoder = self.args.pass_mask
            self.baselined = False
            self.impute_missing = True

        if self.method == "zero_imputation_mask_concatenated_encoder_only":
            self.args.pass_mask = True
            self.args.pass_mask_decoder = False
            self.baselined = False
            self.impute_missing = True

        if self.method == "zero_imputation_mask_concatenated_baselined":
            self.args.pass_mask = True
            self.args.pass_mask_decoder = self.args.pass_mask
            self.baselined = True
            self.impute_missing = True

        if args.imputeoff:
            self.impute_missing = False

        self.baseline = self.args.baseline

        self.masked = self.args.masked
        np.random.seed(1)  # ensure always the same mask is chosen
        self.mask_generator = Generator(
            self.x_dim, self.args.n_masked_vals, n_masks=self.args.unique_masks
        )
        np.random.seed(self.args.seed)
        self.n_unique_masks = self.mask_generator.n_unique_masks

        self.vae = inference_model(
            input_size=self.x_dim,
            latent_size=self.args.latent_size,
            args=self.args,
            impute_missing=self.impute_missing,
            nonlin=self.nonlin,
            uncertainty=self.args.uncertainty,
            combination=self.args.combination,
            n_hidden=self.args.n_hidden,
            freeze_gen=self.args.freeze_decoder,
            C=torch.tensor(self.args.C).to(device),
            d=torch.tensor(self.args.d).to(device),
            noise_std=torch.tensor(self.args.noise).to(device),
            n_masks=self.n_unique_masks + 1,
            dropout=self.dropout,
        ).to(device)

        # set up optimizer
        self.optimizer = torch.optim.Adam(
            self.vae.parameters(), lr=self.args.learning_rate
        )

        # set up time stamp
        self.ts = args.ts if args.ts is not None else time.time()

        # save amount of trainable parameters
        print("Traineable parameters: ", self.count_parameters(self.vae))
        self.args.n_parameters = self.count_parameters(self.vae)
        args.n_parameters = self.args.n_parameters

        # set up model path
        model_dir_path = os.path.join(self.args.fig_root, str(self.ts), self.method)
        ensure_directory(model_dir_path)

    def fit(self):  # ---------------------------------------------------
        """fit the model"""

        # if freeze after training adn restart with masked vals
        if self.args.freeze_after_training:
            self.train_whole_vae = True
        else:
            self.train_whole_vae = False

        # Magic
        if self.args.watchmodel:
            print("Watching model...")
            wandb.watch(self.vae, log="all", log_freq=50)
        # prepare early stopping stop is elbo did not improve for 10 steps
        best_elbo_valid = None
        num_epochs_since_improvement = 0

        self.beta = 0
        interation_count = 0
        self.mean_all = 0
        # ------------------ main training loop ------------------
        for epoch in range(self.args.epochs):
            if epoch < self.args.warmup_range:
                self.beta += self.args.betastep
            out_var_list = []
            for iteration, (batch, y) in enumerate(self.train_loader):
                interation_count += 1

                batch, y = batch.to(self.device), y.to(self.device)

                if self.args.masked:
                    # generate a mask and multiply with it (without baseline or giving it to the network -> zero imputation)
                    # alternate between iterations where it is fully observed and those where it is masked
                    choice = np.random.choice(self.n_unique_masks)
                    # sample 0 or 1 given p=self.args.fraction_full
                    all_full = np.random.choice(
                        2, p=[1 - self.args.fraction_full, self.args.fraction_full]
                    )

                    if all_full:
                        mask = torch.ones_like(batch).to(self.device)
                    else:
                        mask = self.mask_generator(batch, choiceval=choice).to(
                            self.device
                        )

                    # switch off for now
                    if (
                        all_full
                        or self.args.all_obs
                        or (self.train_whole_vae and self.args.freeze_after_training)
                    ):  # initial VAE training on all observed
                        fullmask = mask.shape[0]  # all set to 1 entire batch size
                    else:
                        fullmask = 0
                    mask[:fullmask, ...] = 1

                    batch_original = (
                        batch.clone().detach()
                    )  # unshifted unmasked to calculate loss

                    if self.baselined:
                        # add a baseline to all values
                        batch = self.add_baseline(batch, mask, self.baseline)
                    batch = self.apply_mask(batch, mask)

                    if self.one_impute:
                        batch = self.apply_impute_values(batch, mask, impute_type="one")
                    elif self.mean_impute:
                        batch = self.apply_impute_values(
                            batch, mask, impute_type="mean", mean=self.mean_all
                        )
                    elif self.random_impute:
                        batch = self.apply_impute_values(
                            batch, mask, impute_type="random", val=self.baseline
                        )
                    elif self.val_impute:
                        batch = self.apply_impute_values(
                            batch, mask, impute_type="val", val=self.baseline
                        )

                    # ------------------------------- forward pass vae -------------------------------
                    recon_batch, recon_var, mean, log_var, z, _ = self.vae(
                        x=batch.float(), m=mask.float()
                    )
                else:  # mask passed
                    recon_batch, recon_var, mean, log_var, z, _ = self.vae(
                        x=batch.float()
                    )

                (self.loss, rec_loss, kl, rec_loss_unobs, self.out_var,) = self.loss_fn(
                    recon_batch,
                    recon_var,
                    batch_original.float(),
                    mean,
                    log_var,
                    mask
                    if not self.args.cross_loss
                    else reverse_non_all_obs_mask(mask),
                    beta=self.beta,
                    full=self.args.full_loss,
                )

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                self.logs["elbo"].append(self.loss.item())
                self.logs["rec_loss"].append(
                    rec_loss.item()
                )  # rec loss observed data (actual rec loss in elbo)
                self.logs["kl"].append(kl.item())  # kl
                self.logs["masked_rec_loss"].append(
                    rec_loss_unobs.item()
                )  # rec loss unobserved data (not in elbo)

                wandb.log({"elbo": self.loss.item()})  # elbo on observed values
                wandb.log(
                    {"rec_loss": rec_loss.item()}
                )  # rec loss observed data (actual rec loss in elbo)
                wandb.log(
                    {"masked_rec_loss": rec_loss_unobs.item()}
                )  # rec loss unobserved data (not in elbo)
                wandb.log({"kl": kl.item()})  # kl
                wandb.log({"beta": self.beta})  # beta param

                if self.args.uncertainty:
                    out_var_list.append(
                        self.out_var
                    )  # store out variance for running average

                    wandb.log(
                        {
                            "rec mean MSE": torch.mean(
                                (torch.mean(recon_batch, dim=0) - 1) ** 2
                            ).item()
                        }
                    )  # track mse of mean

                    if all_full:
                        wandb.log(
                            {"post var full": torch.mean(log_var.exp()).item()}
                        )  # all observed variance
                    else:  # masked
                        wandb.log(
                            {"post var mask": torch.mean(log_var.exp()).item()}
                        )  # masked variance

                if (
                    iteration % self.args.print_every == 0
                    or iteration == len(self.train_loader) - 1
                ):
                    print(
                        "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                            epoch,
                            self.args.epochs,
                            iteration,
                            len(self.train_loader) - 1,
                            self.loss.item(),
                        )
                    )

                if not self.args.imputeoff:
                    batch = self.apply_learned_imputation(
                        batch, mask, alpha=self.vae.impute.alpha
                    )

            # end of epoch loop ---------------------------------------------------
            # average over entire epoch
            self.out_var_average = torch.mean(torch.stack(out_var_list), dim=0)
            if self.args.uncertainty:
                if (
                    self.args.loss_type == "optimal_sigma_vae"
                    and not self.args.across_channels
                ):
                    for idx_var in range(min([self.x_dim, 2])):
                        wandb.log(
                            {
                                "avg out var  x{:d} {:.02f}".format(
                                    idx_var,
                                    np.round(self.args.noise[idx_var][0] ** 2, 2),
                                ): self.out_var_average[idx_var].item()
                            }
                        )  # kl
                elif self.args.loss_type == "optimal_sigma_vae":
                    wandb.log({"averaged out var": self.out_var_average.item()})  # kl

            print("---------------- Evaluation Validation ----------------  ")
            print("Epoch: ", epoch)
            start_comp = time.time()
            with torch.no_grad():
                self.evaluate_valid(epoch=epoch)
            end_comp = time.time()
            print("validation ", end_comp - start_comp)
            start_postepoch = time.time()

            print(
                "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Validation Loss {:9.4f}".format(
                    epoch,
                    self.args.epochs,
                    iteration,
                    len(self.train_loader) - 1,
                    self.loss_valid.item(),
                )
            )

            # implement early stopping (if validation elbo has not improved in the last 10 iterations)
            if (
                best_elbo_valid is None or self.logs["elbo_val"][-1] < best_elbo_valid
            ):  # if not def or elbo improved
                best_elbo_valid = self.logs["elbo_val"][-1]
                num_epochs_since_improvement = 0
                if self.args.store_model:
                    # not elegant but avoids lamda issue in dataset that causes pickle error
                    dltrain = deepcopy(self.train_loader)
                    dlvalid = deepcopy(self.validation_loader)
                    dltest = deepcopy(self.test_loader)
                    dltestone = deepcopy(self.test_loader_one)

                    self.train_loader = 0
                    self.validation_loader = 0
                    self.test_loader = 0
                    self.test_loader_one = 0

                    if not self.args.watchmodel:

                        torch.save(
                            self,
                            os.path.join(
                                self.args.fig_root,
                                str(self.ts),
                                self.method,
                                "early_stopping_best_model.pt",
                            ),
                        )
                        torch.save(
                            self,
                            os.path.join(wandb.run.dir, "early_stopping_best_model.pt"),
                        )
                    self.train_loader = dltrain
                    self.validation_loader = dlvalid
                    self.test_loader = dltest
                    self.test_loader_one = dltestone

            elif num_epochs_since_improvement >= self.args.earlystop and epoch > (
                self.args.warmup_range + self.args.earlystop
            ):  # elseif 10 consecutive steps no improvement
                print("early stopping... ")

                # new round of training
                if (
                    self.args.freeze_after_training
                ):  # freeze the decoder training and restart the counter
                    self.freeze_decoder(
                        self.vae
                    )  # detach the decoder weights from trainign
                    self.train_whole_vae = False
                    num_epochs_since_improvement = 0
                    print("Frozen decoder.... reset early stopping criterion")
                else:
                    break

            else:  # no improvement so add 1 to counter
                num_epochs_since_improvement += 1

    def evaluate_valid(self, epoch):
        """Evaluation function called after every print_freq training iterations

        Parameters
        ----------
        data_train / data_val: dict
            Training / test dataset. Linear regressors are estimated on training data and then applied to test data.
            Might not be necessary ?
        """

        self.vae.eval()
        # track loss
        elbo_val = 0.0
        rec_loss_val = 0.0
        kl_val = 0.0
        masked_rec_loss_val = 0.0
        validation_size = 0.0

        for iteration_val, (batch, y) in enumerate(self.validation_loader):

            batch, y = batch.to(self.device), y.to(self.device)
            n_samples = batch.size(0)
            validation_size += n_samples
            print(n_samples)

            # store the sigmas and the mus  and plot the MSE between them during training
            # plot absolut mus
            if self.args.masked:
                # generate a mask and multiply with it (without baseline or giving it to the network -> zero imputation)
                choice = np.random.choice(self.n_unique_masks)
                mask = self.mask_generator(batch, choiceval=choice).to(self.device)
                fullmask = max(
                    0, min(int(self.args.fraction_full * mask.shape[0]), mask.shape[0])
                )
                # keep the masks but hide it at all training times to ensure that thhe masks are new at test time
                # new handle from whole data  all_obs
                if self.args.all_obs:
                    fullmask = mask.shape[0]
                mask[:fullmask, ...] = 1

                batch_original = (
                    batch.clone().detach()
                )  # unshifted unmasked to calculate loss

                if self.baselined:
                    # add a baseline to all values
                    batch = self.add_baseline(batch, mask, self.baseline)
                batch = self.apply_mask(batch, mask)
                if self.one_impute:
                    batch = self.apply_impute_values(batch, mask, impute_type="one")
                elif self.mean_impute:
                    batch = self.apply_impute_values(
                        batch, mask, impute_type="mean", mean=self.mean_all
                    )
                elif self.random_impute:
                    batch = self.apply_impute_values(
                        batch, mask, impute_type="random", val=self.baseline
                    )
                elif self.val_impute:
                    batch = self.apply_impute_values(
                        batch, mask, impute_type="val", val=self.baseline
                    )

            if self.args.masked:
                # here pass the mask to the network

                recon_batch, recon_var, mean, log_var, z, _ = self.vae(
                    x=batch.float(), m=mask.float()
                )
            else:
                recon_batch, recon_var, mean, log_var, z, _ = self.vae(x=batch.float())

            (
                self.loss_valid,
                rec_loss_valid,
                kl_valid,
                rec_loss_unobs_valid,
                self.out_var_valid,
            ) = self.loss_fn(
                recon_batch,
                recon_var,
                batch_original.float(),
                mean,
                log_var,
                mask if not self.args.cross_loss else reverse_non_all_obs_mask(mask),
                beta=self.beta,
                full=self.args.full_loss,
            )

            elbo_val += self.loss_valid.item() * n_samples

            # track loss
            rec_loss_val += (
                rec_loss_valid.item() * n_samples
            )  # rec loss observed data (actual rec loss in elbo)
            kl_val += kl_valid.item() * n_samples  # kl
            masked_rec_loss_val += (
                rec_loss_unobs_valid.item() * n_samples
            )  # rec loss unobserved data (not in elbo)

        self.logs["elbo_val"].append(elbo_val / validation_size)
        # track loss
        self.logs["rec_loss_val"].append(
            rec_loss_val / validation_size
        )  # rec loss observed data (actual rec loss in elbo)
        self.logs["kl_val"].append(kl_val / validation_size)  # kl
        self.logs["masked_rec_loss_val"].append(
            masked_rec_loss_val / validation_size
        )  # rec loss unobserved data (not in elbo)
        self.logs["time_stamp_val"].append(
            len(self.logs["kl"])
        )  # rec loss unobserved data (not in elbo)

        wandb.log({"elbo_val": self.logs["elbo_val"][-1]})
        wandb.log(
            {"rec_loss_val": self.logs["rec_loss_val"][-1]}
        )  # rec loss observed data (actual rec loss in elbo)
        wandb.log(
            {"masked_rec_loss_val": self.logs["masked_rec_loss_val"][-1]}
        )  # rec loss unobserved data (not in elbo)
        wandb.log({"kl_val": self.logs["kl_val"][-1]})  # kl

        self.vae.train()

    def evaluate_test_all(self, save_stub="_"):
        """Compute the posterior mean and variance for the test set for each mask
        compute the ground truth posterior mean and variance for each mask
        plot an overall evaluation test plot

        """

        self.vae.eval()
        # track loss
        validation_size = 0.0

        for iteration_val, (batch, y) in enumerate(self.test_loader):
            batch_clone = batch.clone().detach()

            n_samples = batch.size(0)
            validation_size += n_samples
            batch_original = (
                batch.clone().detach()
            )  # unshifted unmasked to calculate loss

            self.args.n_masks = (
                self.n_unique_masks + 2
            )  # number of masks + 2: fully observed and random

            choice_list = np.arange(self.args.n_masks)
            if self.args.masked:
                # generate a mask and multiply with it (without baseline or giving it to the network -> zero imputation)
                for cc, choice in enumerate(choice_list):
                    batch_c, y = batch_clone.to(self.device), y.to(self.device)

                    if choice == self.n_unique_masks:  # fully observed
                        mask = torch.ones_like(batch_c)
                    elif choice == self.n_unique_masks + 1:  # make half unobserved
                        mask = torch.ones_like(batch_c)
                        mask[:, : self.args.n_masked_vals] = 0
                    else:  # cycle through the unique masks defined by the mask generator
                        mask = self.mask_generator(batch_c, choiceval=choice).to(
                            self.device
                        )

                    if self.baselined:
                        # add a baseline to all values
                        batch_c = self.add_baseline(batch_c, mask, self.baseline)
                    batch_c = self.apply_mask(batch_c, mask)
                    if self.one_impute:
                        batch_c = self.apply_impute_values(
                            batch_c, mask, impute_type="one"
                        )
                    elif self.mean_impute:
                        batch_c = self.apply_impute_values(
                            batch_c, mask, impute_type="mean", mean=self.mean_all
                        )
                    elif self.random_impute:
                        batch_c = self.apply_impute_values(
                            batch_c, mask, impute_type="random", val=self.baseline
                        )
                    elif self.val_impute:
                        batch_c = self.apply_impute_values(
                            batch_c, mask, impute_type="val", val=self.baseline
                        )

                    if self.args.masked:
                        # here pass the mask to the network
                        recon_batch, recon_var, mean, log_var, z, _ = self.vae(
                            x=batch_c.float(), m=mask.float()
                        )
                    else:
                        recon_batch, recon_var, mean, log_var, z, _ = self.vae(
                            x=batch_c.float()
                        )

                    if not self.args.imputeoff:
                        batch_c = self.apply_learned_imputation(
                            batch_c.to(self.device),
                            mask.to(self.device),
                            alpha=self.vae.impute.alpha.to(self.device),
                        )

                    if choice <= self.n_unique_masks + 1:

                        if (
                            torch.std(mask, axis=0) == 0
                        ).all():  # ensure that all masks are the same here

                            if (
                                choice == self.n_unique_masks
                                or self.args.n_masked_vals == 0
                            ):  # all observed
                                masked_idx = []
                            else:
                                masked_idx = torch.where((mask[0, :] == 0))
                                masked_idx = masked_idx[0].cpu().numpy()

                            (
                                mean_new,
                                var_new,
                            ) = return_posterior_expectation_and_variance_multi_d(
                                batch=batch_original.cpu().data.numpy(),
                                args=self.args,
                                masked_idx=masked_idx,
                            )

                            with open(
                                os.path.join(
                                    self.args.fig_root,
                                    str(self.ts),
                                    self.method,
                                    "direct_posterior_var_mean" + str(choice) + ".pkl",
                                ),
                                "wb",
                            ) as f:
                                pickle.dump(
                                    [
                                        mean.cpu().data.numpy(),
                                        mean_new,
                                        torch.exp(log_var).cpu().data.numpy()[:, 0],
                                        var_new,
                                    ],
                                    f,
                                )
                            f.close()
                            if self.args.cross_loss:
                                # for the cross loss calculate the 1-mask mean and variance
                                if (
                                    choice == self.n_unique_masks
                                    or self.args.n_masked_vals == 0
                                ):  # all observed
                                    masked_idx = []
                                else:
                                    masked_idx = torch.where((1 - mask[0, :] == 0))
                                    masked_idx = masked_idx[0].cpu().numpy()

                                (
                                    mean_new,
                                    var_new,
                                ) = return_posterior_expectation_and_variance_multi_d(
                                    batch=batch_original.cpu().data.numpy(),
                                    args=self.args,
                                    masked_idx=masked_idx,
                                )

                                with open(
                                    os.path.join(
                                        self.args.fig_root,
                                        str(self.ts),
                                        self.method,
                                        "cross_loss_posterior_var_mean_"
                                        + str(choice)
                                        + ".pkl",
                                    ),
                                    "wb",
                                ) as f:
                                    pickle.dump(
                                        [
                                            mean.cpu().data.numpy(),
                                            mean_new,
                                            torch.exp(log_var).cpu().data.numpy()[:, 0],
                                            var_new,
                                        ],
                                        f,
                                    )
                                f.close()

                        max_plot = 0
                        for ixd_1 in range(min([2, self.x_dim])):
                            if max_plot > 4:
                                break
                            for idx_2 in range(min([2, self.x_dim])):
                                if idx_2 > ixd_1:
                                    test_visualisations_gauss_one_plot(
                                        recon_batch,
                                        recon_var,
                                        batch_original.float().to(self.device),
                                        mask,
                                        batch_c,
                                        mean,
                                        log_var,
                                        choice,
                                        self,
                                        n_samples=1,
                                        index_1=ixd_1,
                                        index_2=idx_2,
                                        save_stub=save_stub,
                                    )
                                    max_plot += 1
                                if max_plot > 4:
                                    break

    def evaluate_test_all_zero_assumption(self, save_stub="_zeros_"):
        """Compute the posterior mean and variance for the test set for each mask
        compute the ground truth posterior mean and variance for each mask
        but pretend that all values are actually zeros
        """

        self.vae.eval()
        # track loss
        validation_size = 0.0

        for iteration_val, (batch, y) in enumerate(self.test_loader):
            batch_clone = batch.clone().detach()

            n_samples = batch.size(0)
            validation_size += n_samples
            batch_original = (
                batch.clone().detach()
            )  # unshifted unmasked to calculate loss

            self.args.n_masks = (
                self.n_unique_masks + 2
            )  # number of masks + 2: fully observed and random

            choice_list = np.arange(self.args.n_masks)
            if self.args.masked:
                # generate a mask and multiply with it (without baseline or giving it to the network -> zero imputation)
                for cc, choice in enumerate(choice_list):
                    batch_c, y = batch_clone.to(self.device), y.to(self.device)

                    if choice == self.n_unique_masks:  # fully observed
                        mask = torch.ones_like(batch_c)
                    elif choice == self.n_unique_masks + 1:  # make half unobserved
                        mask = torch.ones_like(batch_c)
                        mask[:, : self.args.n_masked_vals] = 0
                    else:  # cycle through the unique masks defined by the mask generator
                        mask = self.mask_generator(batch_c, choiceval=choice).to(
                            self.device
                        )

                    if self.baselined:
                        # add a baseline to all values
                        batch_c = self.add_baseline(batch_c, mask, self.baseline)
                    batch_c = self.apply_mask(batch_c, mask)
                    if self.one_impute:
                        batch_c = self.apply_impute_values(
                            batch_c, mask, impute_type="one"
                        )
                    elif self.mean_impute:
                        batch_c = self.apply_impute_values(
                            batch_c, mask, impute_type="mean", mean=self.mean_all
                        )
                    elif self.random_impute:
                        batch_c = self.apply_impute_values(
                            batch_c, mask, impute_type="random", val=self.baseline
                        )
                    elif self.val_impute:
                        batch_c = self.apply_impute_values(
                            batch_c, mask, impute_type="val", val=self.baseline
                        )

                    if self.args.masked:
                        # here pass the mask to the network
                        recon_batch, recon_var, mean, log_var, z, _ = self.vae(
                            x=batch_c.float(), m=torch.ones_like(mask.float())
                        )  # Note: torch ones added to indicate all masked values actually zero
                    else:
                        recon_batch, recon_var, mean, log_var, z, _ = self.vae(
                            x=batch_c.float()
                        )

                    if not self.args.imputeoff:
                        batch_c = self.apply_learned_imputation(
                            batch_c.to(self.device),
                            mask.to(self.device),
                            alpha=self.vae.impute.alpha.to(self.device),
                        )

                    if choice <= self.n_unique_masks + 1:

                        if (
                            torch.std(mask, axis=0) == 0
                        ).all():  # ensure that all masks are the same here

                            if (
                                choice == self.n_unique_masks
                                or self.args.n_masked_vals == 0
                            ):  # all observed
                                masked_idx = []
                            else:
                                masked_idx = torch.where((mask[0, :] == 0))
                                masked_idx = masked_idx[0].cpu().numpy()

                            # compute the posterior mean and variance for the test set as if all masked values were true zeros
                            (
                                mean_new,
                                var_new,
                            ) = return_posterior_expectation_and_variance_multi_d(
                                batch=batch_c.cpu().data.numpy(),
                                args=self.args,
                                masked_idx=[],
                            )  # ADDED batch_c that was masked instead of original and put in none masked

                            with open(
                                os.path.join(
                                    self.args.fig_root,
                                    str(self.ts),
                                    self.method,
                                    "break_zero_posterior_var_mean"
                                    + str(choice)
                                    + ".pkl",
                                ),
                                "wb",
                            ) as f:
                                pickle.dump(
                                    [
                                        mean.cpu().data.numpy(),
                                        mean_new,
                                        torch.exp(log_var).cpu().data.numpy()[:, 0],
                                        var_new,
                                    ],
                                    f,
                                )
                            f.close()

    def count_parameters(self, model):
        """count number of parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def freeze_decoder(self, model):
        """set required grad to false for decoder parameters"""
        print(
            "number of traininable params before freezing: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        for params in model.decoder.parameters():
            params.requires_grad = False

        print(
            "number of traininable params after freezing: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    def reinit_dataloaders(self, dataset_train, dataset_valid, dataset_test):
        "dataloaders are no longer stored in model - run model.reinit... when evaluating the run"
        self.train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=False,
        )

        self.validation_loader = DataLoader(
            dataset=dataset_valid,
            batch_size=self.args.valid_batch_size,
            shuffle=True,
            pin_memory=False,
        )

        self.test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            pin_memory=False,
        )

        self.test_loader_one = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
        )

    def apply_mask(self, batch, mask):
        """
        Copy batch of objects and zero unobserved features by multiplying with 0 entry (unobserved) in mask
        same as mean imputation
        """
        return batch * mask

    def apply_impute_values(self, batch, mask, impute_type="one", mean=None, val=1):
        """
        Copy batch of objects and add either the mean of a respective pixel to the unobserved part or a one
        random add torch randn like
        """

        if impute_type == "mean":
            assert mean is not None
            batch_dim = batch.shape[0]
            impute = (
                torch.tensor(mean).repeat(batch_dim, 1, 1, 1).to(self.device).float()
            )
            batch[~mask.bool()] = batch[~mask.bool()] + impute[~mask.bool()]
        elif impute_type == "one":
            batch = batch + (1 - mask) * 1
        elif impute_type == "val":
            batch = batch + (1 - mask) * val
        elif impute_type == "random":
            batch = batch + (1 - mask) * (val + torch.randn_like(batch))
        else:
            impute = torch.ones_like(batch)
            batch[~mask.bool()] = batch[~mask.bool()] + impute[~mask.bool()]

        return batch

    def apply_learned_imputation(self, batch, mask, alpha=None):
        "apply the learned imputation value alpha to a batch"
        batch = batch + (1 - mask) * alpha
        return batch

    def add_baseline(self, batch, mask, baseline):
        """add a baseline value to all the batch values that are observed to ensure a zero is actually a zero

        Parameters
        ----------
        batch: data batch
        mask:  mask with same dimensions as batch with 1 at observed and 0 at unobserved features
        baseline: baseline value that is added to each observed feature
        """
        batch[mask.bool()] = batch[mask.bool()] + baseline
        return batch

    def apply_mean_imputation(self, batch, mask, mean):
        """add a baseline value to all the batch values that are observed to ensure a zero is actually a zero

        Parameters
        ----------
        batch: data batch
        mask:  mask with same dimensions as batch with 1 at observed and 0 at unobserved features
        mean: mean of each pixel
        """
        batch[~mask.bool()] = batch[~mask.bool()] + mean
        return batch

    def load_best_model(self):
        """evaluate test set"""
        with open(
            os.path.join(
                self.args.fig_root,
                str(self.ts),
                self.method,
                "early_stopping_best_model.pt",
            ),
            "rb",
        ) as f:
            self.best_model = torch.load(f)
            print("best model loaded")
        self.best_model.vae.eval()
        print(self.best_model.vae)
        print("start eval")


# ------------------ monkey ------------------
class MonkeyModel:
    def __init__(self, net_pars):
        """
        Masked VAE training for the Monkey dataset

        This class is used to train a masked VAE on the Monkey dataset.
        The class is initialized with a set of parameters
        for the network
        The class contains the main training loop .fit() and evaluation functions for the model.

        Parameters
        ----------
        net_pars: dict
            All parameters that should be changed / added to the default attributes of the network class.

        Attributes
        ----------

        net : instance of AE_model
            The network containing encoding and decoding function
        grad_norm : float
            Gradient norm used for gradient clipping of the network weights
        weight_decay :
            Weight decay used for the optimizer
        vae_beta: float
            beta parameter scaling the KL term between posterior and prior
        RAE_beta: float
            Penalty used when training RAE models
        lasso_lambda : bool
            Parameter scaling the degree of sparsity (see http://proceedings.mlr.press/v80/ainsworth18a/ainsworth18a.pdf)
        loss_facs : dict of floats
            Scaling for the loss of each output. Could be used to balance the scale of the different losses.

        filename : str
            Name of the file (actual location will have .pkl added)
        exp_params : dict
            Optional dict of parameters used for the specific submitted job
        description : str
            Optional description of the experiment

        col_dict : dict
            This dictionary contains all information on training progress.
            It get's saved seperately under filename_dicts for easy access.
            Default parameters for evaluation
        eval_params : list of dicts
            Evaulation can be run on different sessions for example.
            Overwrite defaults for each seperate evaluation with the given parameters.

        _iter_count : int
            Current training iteration (we don't use epochs)
        """

        self.net = (
            AE_model(net_pars).cuda()
            if torch.cuda.is_available()
            else AE_model(net_pars)
        )

        # init parameters that get adjusted outside
        self.sessions = None
        self.vae_beta = None
        self.vae_warmup = None
        self.prior_sig = None
        self.fps = None
        self.warmup = None
        self.lasso_lr = None
        self.min_shared_sessions = None
        self.fps_thr = None
        self.iter_start_masking = None
        self.max_iters = None
        self.cross_loss = None
        self.nll_beta = None
        self.batch_size = None
        self.batch_T = None
        self.mask_ratio = None
        self.print_freq = None
        self.full_mask = None
        self.masks = None
        self.half = None
        self.mask_choice = None
        self.lr = None
        self.opt_shared_pars = None
        self.opt_session_pars = None
        self.optim_shared = None
        self.optims_session = None
        self.optim_lasso = None
        self.gen_wm_prior = None
        self.gen_wmats = None
        self.chosen_masks = None

        self.grad_norm = 0.03
        self.weight_decay = 0.2
        self.RAE_beta = 0.2
        self.vae_beta = 0.2
        self.lasso_lambda = 0.0
        self.loss_facs = {k: 1 for k in self.net.outputs}

        self.filename = None
        self.exp_params = None
        self.description = None

        self.col_dict = {}
        self.eval_params = [{}]

        self._iter_count = 0

    def init_dicts(self):
        """Initialize some standard parameters tracked during training"""

        self.col_dict["exp_params"] = self.exp_params
        self.col_dict["eval_params"] = self.eval_params
        self.col_dict["cost_hist"] = pd.Series()
        self.col_dict["update_time"] = pd.Series()
        self.col_dict["zmask_l0"] = pd.Series()
        self.col_dict["zmask_l1"] = pd.Series()
        self.col_dict["df_perf"] = None

    def eval_func(self, data_train, data_val):
        """Evaluation function called after every print_freq training iterations

        Parameters
        ----------
        data_train / data_val: dict
            Training / test dataset. Linear regressors are estimated on training data and then applied to test data.
            Might not be necessary ?
        """
        _, df_perf = run_and_eval(
            self,
            data_train,
            data_val,
            self.eval_params["sessions"],
            self.eval_params["eval_vars"],
            self.eval_params["t_slice"],
            None,
            self.eval_params["inp_mask"],
        )

        if self.col_dict["df_perf"] is None:
            self.col_dict["df_perf"] = pd.DataFrame(df_perf.stack([0])).transpose()
            self.col_dict["df_perf"].set_index(
                np.array([self._iter_count]), inplace=True
            )
        else:
            self.col_dict["df_perf"] = pd.concat(
                [
                    self.col_dict["df_perf"],
                    pd.Series(df_perf.stack([0]), name=self._iter_count),
                ],
                axis=1,
            )

    def get_structured_mask(self, dim, mask_arr=["xa_m", "xb_y", "xb_d"], first=True):
        """Create a structured mask for the network outputs"""
        mask_ = torch.ones(dim)

        for k in mask_arr:
            # check if the first few letters fit the outouts to allow for partial masking
            if k[:4] in self.net.outputs:
                if len(k) <= 4:
                    # this means all xa_m or all xb_y or all xb_d should be masked
                    arr_len = self.net.n_outputs
                else:
                    # else read in the number of elements that should be masked
                    arr_len = int(k[5:])
                if first:
                    # mask the first arr_len elements
                    mask_[self.net.out_inds[k[:4]]][:arr_len] = 0
                else:
                    # mask the last arr_len elements
                    mask_[self.net.out_inds[k[:4]]][arr_len:] = 0
        return mask_.cuda() if torch.cuda.is_available() else mask_

    def fit(
        self,
        data_train=None,
        data_val=None,
        batch_size=32,
        batch_T=100,
        mask_ratio=0.7,
        max_iters=50000,
        learning_rate=5e-4,
        print_output=True,
        print_freq=100,
        full_mask=0.5,
        chosen_masks=None,
    ):
        """
        Function that trains the network.

        Parameters
        ----------
        data_train / data_val: dict
            Training / test dataset.
        batch_size : int
            Batch size for SGD
        batch_T : int
            Length of the traces used for training
        mask_ratio: float between 0 and 1
            At every training iteration we randomely split the
            trace into two sets using the given ratio.
            One of them (with size mask_ratio) is zeroed out in
            the networkinput and the other (1 - mask_ratio) when computing the loss function.
        max_iters : int
            Number of training iterations. This amounts to the total
            amount of gradient updates and not passes through the dataset (epochs).
        learning_rate : float
            Learning rate
        print_output : bool
            Whether to print model evaluations throughout training
        print_freq : int
            After every print_freq iterations the evaluation function
            is run, the model is stored and training progress is printed
        """

        self.batch_size = batch_size
        self.batch_T = batch_T
        self.mask_ratio = mask_ratio
        self.print_freq = print_freq
        self.full_mask = full_mask
        self.lr = learning_rate

        self.init_dicts()

        # ------------------ specifiy the masks ------------------
        self.masks = {}
        self.half = str(int(self.net.out_inds["xa_m"].stop / 2))
        self.masks["xa_m_first_half"] = self.get_structured_mask(
            self.net.n_outputs, mask_arr=[f"xa_m_{self.half}"], first=True
        )
        self.masks["xa_m_last_half"] = self.get_structured_mask(
            self.net.n_outputs, mask_arr=[f"xa_m_{self.half}"], first=False
        )

        # specify the masks where 5, 20, 50 etc. neurons are masked if passed to the function
        # the defailt is None resulting only in xa_m and xb_yd being masked resulting only in
        # either p
        if chosen_masks is not None:
            for key in chosen_masks:
                self.masks[key] = self.get_structured_mask(
                    self.net.n_outputs, mask_arr=[key]
                )
        # all neurons are masked
        self.masks["xa_m"] = self.get_structured_mask(
            self.net.n_outputs,
            mask_arr=[
                "xa_m",
            ],
        )

        # set behavioral masks together with the neural mask for
        # either the y (position) or d (velocity) input
        if "xb_y" in self.net.inputs:
            self.masks["xa_m"] = self.get_structured_mask(
                self.net.n_outputs, mask_arr=["xa_m"]
            )

            self.masks["xb_yd"] = self.get_structured_mask(
                self.net.n_outputs, mask_arr=["xb_y"]
            )
        if "xb_d" in self.net.inputs:

            self.masks["xb_yd"] = self.get_structured_mask(
                self.net.n_outputs, mask_arr=["xb_d"]
            )
            self.masks["xa_m"] = self.get_structured_mask(
                self.net.n_outputs, mask_arr=["xa_m"]
            )
        if "xb_d" in self.net.inputs and "xb_y" in self.net.inputs:
            self.masks["xb_yd"] = self.get_structured_mask(
                self.net.n_outputs, mask_arr=["xb_d", "xb_y"]
            )

        # unless specifically specified we take all neuro and all behavior masks
        if chosen_masks is None:
            self.chosen_masks = ["xa_m", "xb_yd"]
        else:
            self.chosen_masks = chosen_masks

        print("Chosen masks: ", self.chosen_masks)

        # ------------------------------------------------
        #                    TRAINING
        # ------------------------------------------------

        last_print = 0
        tot_t = 0

        self.opt_shared_pars = []
        self.opt_session_pars = [[] for _ in range(self.net.n_sessions)]

        # sort parameters into shared and session specific
        # if they start with session they are session specific
        for n, f in self.net.named_parameters():
            if "session" not in n:
                self.opt_shared_pars += [f]
            else:
                # find separation by dot e.g. session_outp.3.0.bias -> 3
                d1 = n.index(".")
                self.opt_session_pars[int(n[d1 + 1 : n.index(".", d1 + 1)])] += [f]

        self.optim_shared = torch.optim.AdamW(
            self.opt_shared_pars, lr=self.lr, weight_decay=self.weight_decay
        )
        self.optims_session = [
            torch.optim.AdamW(p, lr=self.lr, weight_decay=self.weight_decay)
            for p in self.opt_session_pars
        ]

        # sparsity inducing lasso loss
        self.optim_lasso = torch.optim.SGD(
            [self.net.lasso_W], lr=self.lasso_lr, momentum=0
        )

        # This prevents overfitting
        # generate a prior distribution for the weights of the network
        self.gen_wm_prior = torch.distributions.Normal(0, 1)
        self.gen_wmats = []

        # get all weight matrices of the decoder cnn
        for layer in self.net.cnn1d_dec.modules():
            if isinstance(layer, nn.Conv1d):
                self.gen_wmats.append(layer.weight)

        # get all session output weight matrices of the network
        for t in self.net.session_outp:
            for layer in t.modules():
                if isinstance(layer, nn.Conv1d):
                    self.gen_wmats.append(layer.weight)

        while self._iter_count < max_iters:

            t0 = time.time()
            tot_cost = []
            separate_loss = []
            self.net.train()

            for _ in range(self.print_freq):

                batch = data_train.get_train_batch(
                    self.batch_size, batch_T, to_gpu=torch.cuda.is_available()
                )
                for k in self.net.scaling:
                    if self.net.ifdimwise_scaling:
                        for ii in range(batch[k].shape[1]):
                            batch[k][:, ii] = self.net.dimwise_scaling[k + str(ii)][
                                batch["i"]
                            ][0] * (
                                batch[k][:, ii]
                                - self.net.dimwise_scaling[k + str(ii)][batch["i"]][1]
                            )
                    else:
                        batch[k] = self.net.scaling[k][batch["i"]][0] * (
                            batch[k] - self.net.scaling[k][batch["i"]][1]
                        )  # Scale traces for training

                # cross masking possibility but here we use the structured mask
                # with loss computation on the observed data
                cross_mask = get_mask(self.net.n_outputs, self.mask_ratio, off=True)
                assert (
                    cross_mask[0] == 1
                ), "first output should not be masked"  # pylint: disable=unsubscriptable-object

                # only apply masks with probablity of 1-full_mask, i.e sometimes it is all observed
                all_obs = np.random.choice(
                    [True, False], p=[self.full_mask, 1 - self.full_mask]
                )
                # only select randomly between the chosen masks
                self.mask_choice = np.random.choice(self.chosen_masks)

                if all_obs or self._iter_count < self.iter_start_masking:
                    struct_mask = torch.ones_like(cross_mask)
                else:
                    struct_mask = self.masks[self.mask_choice]

                assert (
                    "xa_m" in self.net.inputs
                ), "xa_m should be in inputs otherwise masking is not working"

                # ------------------------------------------------
                # actual run model on batch data and get outputs
                # ------------------------------------------------
                outputs = self.net(
                    batch,
                    struct_mask[: self.net.n_inputs],
                    scale_output=False,
                )

                self.optim_shared.zero_grad()
                self.optims_session[batch["i"]].zero_grad()
                self.optim_lasso.zero_grad()

                r_loss = 0
                r_sep_loss = np.zeros(len(self.net.outputs))
                obs_var_mean = {ki: [] for ki in range(len(self.net.outputs))}
                obs_var_std = {ki: [] for ki in range(len(self.net.outputs))}

                # loop through all outputs and compute the loss
                for ki, k in enumerate(self.net.outputs):
                    # if all observed or in the phase where we do not yet mask
                    if all_obs or self._iter_count < self.iter_start_masking:
                        l = masked_rec_loss(
                            batch[k],
                            outputs[k],
                            loss=self.net.outputs[k],
                            mask=struct_mask[self.net.out_inds[k]],
                            warmup=self.warmup,
                            obs_var=outputs[k + "_noise"]
                            if "xb" in k and self.net.obs_noise
                            else None,
                            nll_beta=self.nll_beta,
                        )
                    else:  # masking loss
                        l = masked_rec_loss(
                            batch[k],
                            outputs[k],
                            loss=self.net.outputs[k],
                            mask=(1 - struct_mask[self.net.out_inds[k]])
                            if self.cross_loss
                            else struct_mask[self.net.out_inds[k]],
                            warmup=self.warmup,
                            obs_var=outputs[k + "_noise"]
                            if "xb" in k and self.net.obs_noise
                            else None,
                            nll_beta=self.nll_beta,
                        )

                    r_loss += torch.mean(l) * self.loss_facs[k]  # mean over batches
                    r_sep_loss[ki] = torch.mean(l)
                    obs_var_mean[ki] = (
                        torch.mean(outputs[k + "_noise"], axis=(0, 2))
                        .detach()
                        .cpu()
                        .numpy()
                        if "xb" in k and self.net.obs_noise
                        else 0
                    )
                    obs_var_std[ki] = (
                        torch.std(outputs[k + "_noise"], axis=(0, 2))
                        .detach()
                        .cpu()
                        .numpy()
                        if "xb" in k and self.net.obs_noise
                        else 0
                    )
                    # track loss of x and y separately - switch off sum reduction across dimensions
                    if k == "xb_y":
                        # cross mask or actual mask can lead to zero loss - check if all inputs are zero - in that case skip the logging
                        # 1-mask if cross mask is used and we are in the masking regime
                        loss_mask = (
                            (1 - struct_mask[self.net.out_inds[k]])
                            if (
                                self.cross_loss
                                and self._iter_count > self.iter_start_masking
                            )
                            else struct_mask[self.net.out_inds[k]]
                        )
                        if all_obs:  # if all obs regular mask - beats cross loss
                            loss_mask = struct_mask[self.net.out_inds[k]]
                        if loss_mask.sum().item() != 0:
                            l2 = masked_rec_loss(
                                batch[k],
                                outputs[k],
                                loss=self.net.outputs[k],
                                mask=loss_mask,
                                warmup=self.warmup,
                                obs_var=outputs[k + "_noise"]
                                if "xb" in k and self.net.obs_noise
                                else None,
                                reduction="none",
                                nll_beta=self.nll_beta,
                            )
                            wandb.log(
                                {"x loss: ": cpu(torch.mean(l2[:, 0]))}, commit=False
                            )
                            wandb.log(
                                {"y loss: ": cpu(torch.mean(l2[:, 1]))}, commit=False
                            )

                if not self.net.vae:
                    z_reg = self.RAE_beta * torch.mean(eval_RAE_z_reg(outputs["z_mu"]))
                else:
                    vae_beta = self.vae_beta
                    # beta param warmup first no KL then increase
                    if self.vae_warmup:
                        vae_beta *= np.clip(self._iter_count / self.vae_warmup, 0, 1)

                    # KL term between posterior and prior
                    # average over batch but sum over latent dimensions
                    z_reg = torch.mean(
                        torch.sum(
                            vae_beta
                            * eval_VAE_prior(
                                outputs["z_mu"],
                                outputs["z_lsig"],
                                p_mu=0.0,
                                p_lsig=np.log(self.prior_sig),
                            ).sum(-1),
                            1,
                        )
                    )

                loss = r_loss + z_reg
                wandb.log({"Train rec loss: ": cpu(r_loss)})
                wandb.log({"KL loss: ": cpu(z_reg)}, commit=False)
                wandb.log({"total loss: ": cpu(loss)}, commit=False)

                if self.lasso_lambda:
                    wmat_reg = sum(
                        [self.gen_wm_prior.log_prob(W).sum() for W in self.gen_wmats]
                    )
                    # lasso loss to prevent overfitting on the weight matrices
                    loss -= wmat_reg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.opt_shared_pars, max_norm=self.grad_norm, norm_type=2
                )
                torch.nn.utils.clip_grad_norm_(
                    self.opt_session_pars[batch["i"]],
                    max_norm=self.grad_norm,
                    norm_type=2,
                )
                self.optim_shared.step()
                self.optims_session[batch["i"]].step()
                self.optim_lasso.step()

                # normalisation lasso loss for lasso_W sparse matrix
                if self.lasso_lambda:
                    # if the lasso matrix exists and column normalization is desired
                    if self.net.lasso_mat and self.net.column_norm:
                        # Calculate the L2 norm for each column in the weight matrix
                        norm = torch.sqrt((self.net.lasso_W**2).sum(0))

                        # Normalize each column of the weight matrix by its L2 norm,s
                        # using a minimum value of 1e-16 for the norms to avoid division by zero
                        self.net.lasso_W.data.div_(torch.clamp_min(norm, 1e-16)[None])

                        # Perform L1 regularization:
                        # Multiply each column of the weight matrix by the difference between
                        # its original norm and a penalty term, which is the learning rate
                        # times the regularization parameter. This difference is clamped to
                        # a minimum of 0 to ensure the norms remain non-negative
                        self.net.lasso_W.data.mul_(
                            torch.clamp_min(
                                norm - self.lasso_lr * self.lasso_lambda, 0
                            )[None]
                        )
                    else:
                        # Calculate the absolute value for each entry in the weight matrix
                        norm = abs(self.net.lasso_W)

                        # Normalize each entry of the weight matrix by its absolute value,
                        # using a minimum value of 1e-16 for the norms to avoid division by zero
                        self.net.lasso_W.data.div_(torch.clamp_min(norm, 1e-16))

                        # Perform L1 regularization:
                        # Multiply each entry of the weight matrix by the difference between
                        # its absolute value and a penalty term, which is the learning rate
                        # times the regularization parameter. This difference is clamped to
                        # a minimum of 0 to ensure the norms remain non-negative
                        self.net.lasso_W.data.mul_(
                            torch.clamp_min(norm - self.lasso_lr * self.lasso_lambda, 0)
                        )

                tot_cost.append(
                    cpu(r_loss + z_reg)
                )  # change to total loss not just rec loss
                separate_loss.append(r_sep_loss)
                self._iter_count += 1

            tot_t += time.time() - t0

            updatetime = 1000 * (tot_t) / (self._iter_count - last_print)
            last_print = self._iter_count
            tot_t = 0

            # ------------------------------------------------
            #                    EVALUATION
            # ------------------------------------------------

            self.col_dict["cost_hist"].at[self._iter_count] = np.mean(tot_cost)
            for ik, k in enumerate(self.net.outputs):
                wandb.log(
                    {
                        f"Train rec loss {k}: ": np.mean(
                            cpu(np.array(separate_loss)[:, ik])
                        )
                    }
                )
                if "xb" in k and self.net.obs_noise:
                    wandb.log({f"Train obs var mean x {k}: ": cpu(obs_var_mean[ik][0])})
                    wandb.log({f"Train obs var mean y {k}: ": cpu(obs_var_mean[ik][1])})
                    wandb.log({f"Train obs var std x {k}: ": cpu(obs_var_std[ik][0])})
                    wandb.log({f"Train obs var std y {k}: ": cpu(obs_var_std[ik][1])})

            # "zmask_l1" logs the sum of the absolute values of the elements in the weight matrix.
            # This is essentially the L1 norm of the weight matrix.
            # It gives a measure of the total "magnitude" of the weights. When using L1 regularization,
            # this value will tend to decrease, as the regularization pushes weights towards zero.
            self.col_dict["zmask_l1"].at[self._iter_count] = np.sum(
                cpu(abs(self.net.lasso_W))
            )
            # "zmask_l0" logs the number of non-zero elements in the weight matrix.
            # This is done by first summing over all dimensions except the last one, and then finding the
            # indices where this sum is non-zero.
            # It gives a measure of the sparsity of the weight matrix. When using L1 regularization,
            # this number will tend to increase, as the regularization pushes more weights to become zero.

            self.col_dict["zmask_l0"].at[self._iter_count] = len(
                np.apply_over_axes(
                    np.sum, cpu(self.net.lasso_W), range(cpu(self.net.lasso_W).ndim - 1)
                ).nonzero()[0]
            )
            self.col_dict["update_time"].at[self._iter_count] = updatetime

            # MAIN EVALUAION FUNCTION
            self.eval_func(data_train, data_val)

            wandb.log(
                {
                    k: self.col_dict[k][self._iter_count]
                    for k in ["cost_hist", "zmask_l0", "zmask_l1", "update_time"]
                }
            )
            # check inputs to model
            try:
                wandb.log(
                    {
                        "Valid LogL/T x_m: ": self.col_dict["df_perf"][
                            "xa_m", 0, "Valid", "Pred", "LogL/T"
                        ][self._iter_count]
                    },
                    commit=False,
                )
            except KeyError as e:
                print(f"KeyError: {e} - keys missing in col_dict")
            try:
                wandb.log(
                    {
                        "Valid RMSE dy pred: ": self.col_dict["df_perf"][
                            "xb_y", 0, "Valid", "Pred", "RMSE"
                        ][self._iter_count]
                    },
                    commit=False,
                )
            except KeyError as e:
                print(f"KeyError: {e} - keys missing in col_dict")
            try:
                wandb.log(
                    {
                        "Train LogL/T x_m: ": self.col_dict["df_perf"][
                            "xa_m", 0, "Train", "Pred", "LogL/T"
                        ][self._iter_count]
                    },
                    commit=False,
                )
            except KeyError as e:
                print(f"KeyError: {e} - keys missing in col_dict")
            try:
                wandb.log(
                    {
                        "Train RMSE dy pred: ": self.col_dict["df_perf"][
                            "xb_y", 0, "Train", "Pred", "RMSE"
                        ][self._iter_count]
                    },
                    commit=False,
                )
            except KeyError as e:
                print(f"KeyError: {e} - keys missing in col_dict")

            if print_output:
                print(
                    f"Cost: {self.col_dict['cost_hist'][self._iter_count]:0.3f}", end=""
                )
                print(f" || Time Upd.: {float(updatetime):0.1f} ms", end="")
                print(f" || BatchNr.: {self._iter_count}", end="")

                # Handle potential exceptions more explicitly
                try:
                    logl_t_value = self.col_dict["df_perf"][
                        "xa_m", 0, "Valid", "Pred", "LogL/T"
                    ][self._iter_count]
                    print(f" || LogL/T x_m: {logl_t_value:0.3f}", end="")
                except KeyError:
                    pass  # Handle missing keys in col_dict

                try:
                    rmse_value = self.col_dict["df_perf"][
                        "xb_y", 0, "Valid", "Pred", "RMSE"
                    ][self._iter_count]
                    print(f" || RMSE dy pred: {rmse_value:0.3f}", end="\n")
                except KeyError:
                    pass  # Handle missing keys in col_dict

            sys.stdout.flush()

            if self.filename:
                self.col_dict["description"] = self.description
                with open(self.filename + "/model.pkl", "wb") as f:
                    torch.save(self, f)
                with open(self.filename + "/model_dicts.pkl", "wb") as f:
                    torch.save(self.col_dict, f)
                with open(wandb.run.dir + "/model_dicts.pkl", "wb") as f:
                    torch.save(self.col_dict, f)

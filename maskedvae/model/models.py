import time
import os
import pickle

import copy
from copy import deepcopy
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from maskedvae.utils.loss import masked_GNLL_loss_fn, masked_MSE_loss_fn
from maskedvae.model.masks import MultipleDimGauss
from maskedvae.model.networks import GLVM_VAE
from maskedvae.utils.utils import (
    ensure_directory,
    reverse_non_all_obs_mask,
    return_posterior_expectation_and_variance_multi_d,
)
from maskedvae.plotting.plotting import test_visualisations_gauss_one_plot


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
        Initialize with a single dict that contains the parameters for the network.
        the chosen dataset, the infernece network model and the training device
        All other parameters for training and evaluation can be changed manually after initialization.

        Parameters
        ----------
        args: dict
            All parameters that should be changed / added to the default attributes of the network class.

        args: dataset
            Dataset class

        args: inference_model
            Inference network with stanard VAE as default

        args: device
            cpu if no gpu detected

        Attributes
        ----------
        TODO: add here

        """
        self.args = args
        self.device = device
        self.dropout = dropout
        print(self.args)

        print("Run model")

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

import argparse
import copy
import random
from argparse import Namespace
from datetime import datetime
from os.path import expanduser
import pickle

import torch
import torch.nn as nn
import wandb
import numpy as np
import yaml
import os
import importlib

from maskedvae.model.models import ModelGLVM
from maskedvae.model.networks import GLVM_VAE
from maskedvae.utils.utils import (
    ensure_directory,
    save_pickle,
    save_yaml,
    save_run_directory,
    csv_to_simple_yaml,
)
from maskedvae.model.masks import MultipleDimGauss


def main(args):
    data_directory = "./data/"
    run_directory = "./runs/rev_glvm/"
    ensure_directory(run_directory)

    with open("./configs/glvm/fit_config_revisions.yml", "r") as f:
        data_conf = yaml.load(f, Loader=yaml.Loader)

    # update the config file
    # the second overwrites the first - comman line will beat config
    args_dict = {**data_conf.__dict__, **args.__dict__}
    args = Namespace(**args_dict)

    assert np.abs(args.fraction_full - 1 / (args.unique_masks + 1)) <= 0.05

    # setup torch and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    method_handle = copy.copy(args.method)

    # shorter testing - fewer C sigma and d values fewer datapoints
    flag = "_shorttest" if args.shorttest else ""
    args.list_len = 1 if args.shorttest else args.list_len

    for c_id in range(args.list_len):  # adjust this such that the noise ID is fixed
        if c_id % 3 != 0:
            continue
        dataset_train_full = torch.load(
            f"{data_directory:s}/rev_glvm/20_dim_1_dim/data_train_{c_id:d}{flag:s}.pt"
        )
        dataset_valid = torch.load(
            f"{data_directory:s}/rev_glvm/20_dim_1_dim/data_valid_{c_id:d}{flag:s}.pt"
        )
        dataset_test = torch.load(
            f"{data_directory:s}/rev_glvm/20_dim_1_dim/data_test_{c_id:d}{flag:s}.pt"
        )

        for train_samples in [1, 10, 30, 60, 100, 1000, 5000, len(dataset_train_full)]:
            # only use a subset of the training data
            dataset_train = copy.deepcopy(dataset_train_full)
            dataset_train.z = dataset_train.z[:train_samples]
            dataset_train.x = dataset_train.x[:train_samples]
            dataset_train.n_samples = train_samples

            args.C = dataset_train.C
            args.d = dataset_train.d
            args.z_prior = np.stack((np.zeros(args.z_dim), np.ones(args.z_dim)))
            args.noise = dataset_train.noise

            print("C ", args.C, "noise ", args.noise)

            print(
                "train ",
                len(dataset_train),
                "valid ",
                len(dataset_valid),
                "test ",
                len(dataset_test),
            )
            # determine the image size MNIST
            args.x_dim = dataset_train.x_dim

            methods = [
                "zero_imputation_mask_concatenated_encoder_only",
                "zero_imputation",
            ]
            # short names for WandB
            method_short = [
                "enc-mask-",
                "zero-",
            ]
            meth_des = [a[:-1] + " " for a in method_short]

            if not args.all:
                print("Not all but just ")
                methods = methods[method_handle : method_handle + 1]  #
                method_short = method_short[method_handle : method_handle + 1]  #
                print(methods)

            logs = {
                method: {
                    "elbo": [],
                    "kl": [],
                    "rec_loss": [],
                    "elbo_val": [],
                    "kl_val": [],
                    "rec_loss_val": [],
                    "masked_rec_loss": [],
                    "masked_rec_loss_val": [],
                    "observed_mse": [],
                    "masked_mse": [],
                    "time_stamp_val": [],
                }
                for method in methods
            }

            args.ts = datetime.now().strftime(
                "%d%m%Y_%H%M" "%S"
            )  # more easily readable

            # Specify the masking generator
            args.generator = MultipleDimGauss
            # Specify which non linearity to use for the networks
            nonlin_mapping = {
                0: nn.Identity(),
                1: nn.ELU(),
                2: nn.ReLU(),
                3: nn.Sigmoid(),
            }
            args.nonlin_fn = nonlin_mapping.get(args.nonlin, nn.Identity())

            # ensure either one or mean imputation if both zero standard zero imputation
            args.mean_impute = not args.one_impute and args.mean_impute

            # ------------------------ loop over methods ----------------------------------------

            for i, method in enumerate(methods):

                # skip the run if all_obs is combined with masked
                if args.all_obs and method != "zero_imputation":
                    continue
                elif not args.all_obs and method == "zero_imputation":
                    continue
                else:
                    print("Running method: ", method, "all obs args: ", args.all_obs)

                # pass the right logs to network
                args.method = method
                name_tag = (
                    f"{train_samples}_sig_{args.noise[0][0]:.2f}_C_{args.C[0][0]:.2f}_"
                )
                args.train_samples = train_samples

                if args.freeze_decoder and args.loss_type != "regular":
                    args.loss_type = "regular"
                    print("The decoder is frozen -> switching loss type to regular...")
                    name_tag = f"_frozen{name_tag}"
                print(name_tag)
                # Initialize the Weights and Biases (wandb) run
                run = wandb.init(
                    project=args.project_name,
                    group=f"{args.exp}{name_tag}",
                    name=f"{method_short[i]}{args.ts}_{name_tag}",
                    reinit=True,
                    config=args,
                    dir=run_directory,
                )

                # Setup the directory for storing figures
                figs_directory = os.path.join(wandb.run.dir, "figs")
                os.makedirs(
                    figs_directory, exist_ok=True
                )  # os.makedirs can create directories and won't throw an error if the directory already exists

                # Update wandb configuration with the figures directory path
                args.fig_dir = figs_directory
                wandb.config.update({"fig_dir": args.fig_dir}, allow_val_change=True)

                print("Masked model...")
                model = ModelGLVM(
                    args=args,
                    dataset_train=dataset_train,
                    dataset_valid=dataset_valid,
                    dataset_test=dataset_test,
                    logs=logs[method],
                    device=device,
                    inference_model=GLVM_VAE,
                    Generator=args.generator,
                    nonlin=args.nonlin_fn,
                    dropout=args.dropout,
                )

                print(" ---------- begin model fit ---------------")
                print(print(method))

                model.fit()
                # run evaluation on test set and store results
                # model.evaluate_test_all_zero_assumption() # compute analytical posterior with zero assumption
                model.evaluate_test_all()
                print(" ---------- end model fit ---------------")
                print("  ")

                model_path = os.path.join(
                    model.args.fig_root, str(model.ts), model.method
                )
                ensure_directory(model_path)

                # save logs and args dictionary
                save_pickle(model.logs, os.path.join(model_path, "logs.pkl"))
                save_pickle(model.args, os.path.join(model_path, "args.pkl"))
                save_yaml(model.args, os.path.join(model_path, "args.yml"))

                model.train_loader = 0
                model.test_loader = 0
                model.validation_loader = 0

                # save the model
                if not model.args.watchmodel:
                    model_filepath = os.path.join(
                        model_path, "model_end_of_training.pt"
                    )
                    torch.save(model, model_filepath)
                    torch.save(
                        model, os.path.join(wandb.run.dir, "model_end_of_training.pt")
                    )
                save_run_directory(
                    method, args, str(model.ts), "./runs/"
                )  # for multiple datasets change filename
                run.finish()

            # ------------------------ end loop over methods ----------------------------------------

            joint_directory_path = os.path.join(model.args.fig_root, str(model.ts))
            ensure_directory(joint_directory_path)
            save_pickle(model.logs, os.path.join(joint_directory_path, "logs.pkl"))
            a = 1


if __name__ == "__main__":
    data_dir = {
        "glvm": "GLVM",
    }
    parser = argparse.ArgumentParser()

    # Run configuration
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--all", type=int, default=1)  # all methods or just one
    parser.add_argument(
        "--list_len", type=int, default=10
    )  # number of datasets to run for multitple datasets set 10
    parser.add_argument("--method", type=int, default=0)  # choose method 0,1,2,3

    # all observed in training but still keep all the test masking conditions
    parser.add_argument("--all_obs", type=int, default=0)
    parser.add_argument(
        "--cross_loss", type=int, default=0
    )  # compute the loss for the masked values
    parser.add_argument(
        "--full_loss", type=int, default=0
    )  # should the loss be counted for both masked and unmasked

    # Network settings
    parser.add_argument("--beta", type=float, default=1.0)  # value for the baseline
    parser.add_argument(
        "--dropout", type=float, default=0.0
    )  # 25)  # value for the baseline
    parser.add_argument(
        "--nonlin", type=int, default=2
    )  # 0: Identity 1: ELU 2: RELU 3 Sigmoid else Identity
    parser.add_argument(
        "--freeze_after_training", type=int, default=0
    )  # train normal decoder and freeze after training

    # imputation methods when passing data to
    parser.add_argument("--one_impute", type=int, default=0)  # impute with one
    parser.add_argument("--val_impute", type=int, default=1)  # impute with value
    parser.add_argument(
        "--baseline", type=float, default=1.0
    )  # value for the baseline used for val_impute
    parser.add_argument("--mean_impute", type=int, default=0)  # impute with mean
    parser.add_argument(
        "--random_impute", type=int, default=0
    )  # set if random values should be used to impute to confuse the network

    # experiment name
    parser.add_argument("--exp", type=str, default="std_glvm")
    parser.add_argument("--task_name", type=str, default="glvm")

    # Training configurations visualise and store plots during run
    parser.add_argument("--visualise", type=int, default=1)  # plot during training
    parser.add_argument(
        "--shorttest", type=int, default=0
    )  # shortens training loop just testsetting

    # offline run
    parser.add_argument("--offline", type=int, default=1)  # offline run wandb

    args = parser.parse_args()
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    main(args)

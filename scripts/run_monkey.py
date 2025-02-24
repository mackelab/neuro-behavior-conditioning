import argparse
import gc
import os
import pickle
import random
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
import yaml
from torch import nn

import wandb
from maskedvae.datasets.datasets import Primate_Reach
from maskedvae.model.models import MonkeyModel
from maskedvae.utils.evaluate import (
    compute_calibration_score,
    get_sampled_masked_outputs,
    run_full_evaluation,
)
from maskedvae.utils.utils import get_mask, make_directory, param_iteration


def main(args):
    # main running loop to start model fitting and evaluation

    # load the config file
    with open(
        os.path.join("./configs/monkey/fit_config.yml"), "r", encoding="utf-8"
    ) as f:
        data_conf = yaml.load(f, Loader=yaml.Loader)
    # update the config file
    args_dict = {**data_conf.__dict__, **args.__dict__}  # commandline overwrites config
    args = Namespace(**args_dict)

    # set all seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # -------------------------------------------#
    #     Set up the dataset and run paths       #
    # -------------------------------------------#
    data_dir = "./data/monkey/"
    temp_dir = "./runs/monkey/"
    os.makedirs(temp_dir, exist_ok=True)

    with open(data_dir + "all_sessions.pkl", "rb") as f:
        data_all_sessions = pickle.load(f)

    # check if the amount of shared sessions is smaller than the actual amount of sessions
    min_shared = min(
        [len(args.run_params["sessions"]), args.run_params["min_shared_sessions"]]
    )
    args.run_params["min_shared_sessions"] = min_shared

    # -------------------------------------------#
    #     Set up the dataset train test valid    #
    # -------------------------------------------#

    PD_train = Primate_Reach(
        data_all_sessions,
        args.run_params["sessions"],
        fps=args.run_params["fps"],
        fr_threshold=args.run_params["fps_thr"],
    )
    PD_train.filt_units(min_sessions=args.run_params["min_shared_sessions"])
    PD_train.plot_behavior()
    PD_train.filt_times_percent_range(per_start=0.0, per_end=0.7)

    PD_test = Primate_Reach(
        data_all_sessions,
        args.run_params["sessions"],
        fps=args.run_params["fps"],
        fr_threshold=args.run_params["fps_thr"],
    )
    PD_test.filt_units(min_sessions=args.run_params["min_shared_sessions"])
    PD_test.filt_times_percent_range(per_start=0.7, per_end=0.8)
    print(PD_test.n_bins)

    PD_valid = Primate_Reach(
        data_all_sessions,
        args.run_params["sessions"],
        fps=args.run_params["fps"],
        fr_threshold=args.run_params["fps_thr"],
    )
    PD_valid.filt_units(min_sessions=args.run_params["min_shared_sessions"])
    PD_valid.filt_times_percent_range(per_start=0.8, per_end=1.0)
    print(PD_valid.n_bins)

    # ------------------------------------#
    #     Set up evaluation parameters    #
    # ------------------------------------#
    eval_params = {
        "sessions": (0, 1),
        "eval_vars": None,
        "t_slice": np.index_exp[0:1000],  # adjust based on fps
        "inp_mask": [],
    }
    eval_params["sessions"] = np.arange(len(args.run_params["sessions"]))
    # get the lowest number of evalution steps
    eval_nslice = min([PD_valid.n_bins[qq] for qq in list(eval_params["sessions"])]) - 1
    # ensure the maximum steps in all possible session test sets
    eval_test_nslice = (
        min([PD_test.n_bins[qq] for qq in list(eval_params["sessions"])]) - 1
    )
    # slice test and validation indices - used at evalution time
    eval_params["t_slice"] = np.index_exp[:eval_nslice]
    eval_params["t_slice_test"] = np.index_exp[:eval_test_nslice]
    print(eval_params["t_slice"], eval_params["t_slice_test"])

    # ------------------------------------#
    #     Set up network parameters       #
    # ------------------------------------#
    net_pars = {}
    for k in args.run_params.keys():
        net_pars[k] = args.run_params[k]
    net_pars["nonlin"] = nn.ELU
    net_pars["vae"] = True
    net_pars["forw_backw"] = args.forw_backw
    net_pars["column_norm"] = False
    net_pars["inp_dims"] = PD_train.n_traces
    net_pars["n_sessions"] = len(PD_train)
    scaling = {
        k: {ses: 0 for ses in range(len(PD_train))}
        for k in net_pars["outputs"]
        if "xb" in k
    }
    dimwise_scaling = {
        k: {ses: 0 for ses in range(len(PD_train))}
        for k in ["xb_d0", "xb_d1", "xb_y0", "xb_y1"]
    }

    # compute scaling on the training set
    for ses in range(len(PD_train)):
        session = PD_train.get_session(ses)
        for k in net_pars["outputs"]:
            if "xb" in k:
                scaling[k][ses] = [1 / np.sqrt(np.var(session[k])), np.mean(session[k])]
                for i in range(session[k].shape[1]):
                    dimwise_scaling[k + str(i)][ses] = [
                        1 / np.sqrt(np.var(session[k][:, i])),
                        np.mean(session[k][:, i]),
                    ]

    net_pars["scaling"] = scaling
    net_pars["dimwise_scaling"] = dimwise_scaling
    net_pars["obs_noise"] = args.obs_noise
    net_pars["mean_obs_noise"] = bool(args.mean_obs_noise)
    net_pars["warmup"] = args.run_params["warmup"]
    net_pars["larger_encoder"] = args.run_params["larger_encoder"]
    net_pars["max_iters"] = args.max_iters

    # If GNLL outputs should model observation noise
    if (
        args.obs_noise
        and np.array(
            [net_pars["outputs"][key] != "gnll" for key in net_pars["outputs"]]
        ).all()
    ):
        raise ValueError("obs_noise is used but none of the outputs is gnll")

    # ------------------------------------#
    #      Initialise the model...        #
    # ------------------------------------#

    model = MonkeyModel(net_pars)
    # print number of parameters
    args.n_params = model.net.get_n_params()
    print("Number of parameters: ", model.net.get_n_params())

    # ------------------------------------#
    #      Model parameter setup          #
    # ------------------------------------#

    model.sessions = args.run_params["sessions"]
    model.vae_beta = args.run_params["vae_beta"]
    model.vae_warmup = args.run_params["vae_warmup"]
    model.prior_sig = 1.0
    model.fps = args.run_params["fps"]
    model.warmup = args.run_params["warmup"]
    model.lasso_lambda = args.run_params["lasso_lambda"]
    model.lasso_lr = args.run_params["lasso_lr"]
    model.min_shared_sessions = args.run_params["min_shared_sessions"]
    model.fps_thr = args.run_params["fps_thr"]
    model.iter_start_masking = args.iter_start_masking
    model.max_iters = args.max_iters
    model.cross_loss = args.cross_loss
    model.loss_facs["xb_y"] = args.run_params["xb_y_loss_facs"]
    model.nll_beta = args.run_params["nll_beta"]
    model.eval_params = eval_params

    # ------------------------------------#
    #     specify storage names           #
    # ------------------------------------#

    args.ts = datetime.now().strftime("%d%m%y_%H%M%S.%f").replace(".", "_")
    if args.run_params["full_mask_prob"] < 1:
        args.masktype = "masked_"
    args.masktag = args.masktype + "_"

    name_tag = ""
    for ses in args.run_params["sessions"]:
        name_tag += f"{ses}_"
    name_tag = (
        f'{name_tag}beta_{args.run_params["vae_beta"]}_lat_'
        f'{args.run_params["n_latents"]}_nll_beta_{args.run_params["nll_beta"]}'
        f'_lasso_mat_{args.run_params["lasso_mat"]}'
    )
    # Eval params
    filename = "model"
    file_dir = f"{filename}_{args.ts}_{args.masktype}_{name_tag[3:]}"
    make_directory(temp_dir + file_dir)

    model.filename = temp_dir + file_dir
    print(model.filename)

    # ------------------------------------#
    #      test model / forward pass      #
    # ------------------------------------#
    batch = PD_train.get_train_batch(10, 100, to_gpu=torch.cuda.is_available())
    for k in model.net.scaling:
        batch[k] = model.net.scaling[k][batch["i"]][0] * (
            batch[k] - model.net.scaling[k][batch["i"]][1]
        )
    out_mask = get_mask(model.net.n_outputs, 1e-8, off=True)  # mask m
    mask = out_mask[: model.net.n_inputs]
    model.net(batch, mask, scale_output=False)

    # ------------------------------------#
    #      Initialise wandb...            #
    # ------------------------------------#

    group_string = (
        f"{args.masktag}_cross_{args.cross_loss}_dimwise_"
        f"{net_pars['ifdimwise_scaling']}_{net_pars['outputs']['xb_y']} "
        f"{args.exp}{name_tag}_{str(args.run_params['full_mask_prob'])}"
        f"_nll_beta{str(args.run_params['nll_beta'])}"
    )

    run = wandb.init(
        project=args.project_name,
        group=group_string,
        name=args.masktag + args.ts + "_" + str(args.seed),
        reinit=True,
        config=args,
        dir=temp_dir,
    )

    print("---------------------------------------")
    print(" ")
    print("  Begin model fitting...  ")
    print(" ")
    print("---------------------------------------")

    model.fit(
        PD_train,
        PD_valid,
        batch_size=args.batch_size,
        batch_T=args.run_params["batch_T"],
        mask_ratio=args.run_params["mask_ratio"],
        max_iters=net_pars["max_iters"],
        learning_rate=args.run_params["lr"],
        print_output=True,
        print_freq=args.run_params["print_freq"],
        full_mask=args.run_params["full_mask_prob"],
    )
    print("---------------------------------------")
    print(" ")
    print("    End model fitting...  ")
    print(" ")
    print("---------------------------------------")

    run.finish()
    run_full_evaluation(
        model,
        data_all_sessions,
        save_path=model.filename + "/eval_df.h5py",
        sessions=np.arange(len(model.sessions)),
        t_slice=eval_params["t_slice_test"],  # change according to fps
        run_latent_ablation=False,
        run_cross_perf=False,
        fr_threshold=args.run_params["fps_thr"],
    )
    del data_all_sessions
    gc.collect()

    # ------------------------------------#
    #    save files / make final evals    #
    # ------------------------------------#

    # save args dictionary
    with open(model.filename + "/args.pkl", "wb") as args_file:
        pickle.dump(args, args_file)

    with open(model.filename + "/args.yml", "w", encoding="utf-8") as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

    with open(model.filename + "/net_pars.yml", "w", encoding="utf-8") as outfile:
        yaml.dump(net_pars, outfile, default_flow_style=False)

    with open(model.filename + "/net_pars.pkl", "wb") as net_file:
        pickle.dump(net_pars, net_file)

    gc.collect()

    compute_calibration_score(
        model,
        PD_valid,
        get_predictions=True,
        sessions=(0,),
        t_slice=eval_params["t_slice"],
        lat_mask=None,
        inp_mask=("xa_s",),
        test_masks=["xb_yd"],
        comp_calibration=True,  # compute calibration for the model
        outputs=None,
        store_samples=True,
        neuro=True,
        n_post_samples=50,  # number of samples from the posterior
        exp_dir=model.filename + "/",
        eval_test_nslice=eval_test_nslice,
        drop_dir=model.filename + "/",
    )

    # sample from the approxmiate posterior and compute test evaluation metrics
    # run analyses for different test_masks
    get_sampled_masked_outputs(
        model,
        PD_test,
        get_predictions=True,
        sessions=(0,),
        t_slice=eval_params["t_slice_test"],
        lat_mask=None,
        inp_mask=("xa_s",),
        test_masks=["xa_m_first_half", "xa_m_last_half", "all_obs", "xb_yd", "xa_m"],
        comp_calibration=True,  # compute calibration for the model
        outputs=None,
        store_samples=True,
        neuro=True,
        n_post_samples=50,  # number of samples from the posterior
        exp_dir=model.filename + "/",
        eval_test_nslice=eval_test_nslice,
        drop_dir=model.filename + "/",
    )

    del model, batch
    del PD_test, PD_train, PD_valid
    gc.collect()

    print(" ")
    print(" ")
    print("           Done with all analyses!            ")
    print(" ")
    print(" ")


if __name__ == "__main__":

    ################################################
    #                                              #
    #           fix and refactor all configs here  #
    #           after revisions keep flexibility   #
    #           for now                            #
    #                                              #
    ################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--latent_size", type=int, default=40)
    parser.add_argument("--exp", default="std")  # experiment run
    # parser.add_argument("--laptop", type=int, default=0)
    parser.add_argument("--masktype", type=str, default="all_obs")
    parser.add_argument("--slurm", type=float, default=0)
    parser.add_argument(
        "--comp_calibration", type=int, default=1
    )  # reduced dimensions in encoder and decoder
    parser.add_argument(
        "--save_files", type=int, default=1
    )  # reduced dimensions in encoder and decoder
    parser.add_argument(
        "--freeze_after_training", type=int, default=0
    )  # train normal decoder and freeze after training
    parser.add_argument(
        "--max_iters", type=int, default=25000
    )  # iterations for training
    parser.add_argument(
        "--iter_start_masking", type=int, default=5000
    )  # when to start masking
    # observation noise per channel not per timestep
    parser.add_argument("--mean_obs_noise", type=int, default=0)
    parser.add_argument("--offline", type=int, default=1)
    parser.add_argument("--cross_loss", type=int, default=0)
    parser.add_argument("--switch_cross_loss", type=int, default=1)
    parser.add_argument("--nll_beta_on", type=int, default=False)

    args = parser.parse_args()
    print(args)
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    now = datetime.now().strftime("%d%m%Y_%H%M%S")  # more easily rea
    args.experiment = f"prim_reach_{now}"

    for seed in [args.seed]:
        for n_latents in [args.latent_size]:
            args.seed = seed
            print("---------------------------------------")
            print(" ")
            print("           seed", seed, n_latents)
            print(" ")
            print("---------------------------------------")

            variable_col = param_iteration()
            variable_col.add("lr", 1e-3)
            variable_col.add(
                "lasso_mat", False
            )  # is it a matrix (initialisation kaiming) or a vector (initialisation 0.1)
            variable_col.add("n_latents", n_latents)  # only 28 for now
            variable_col.add(
                "group_latents", {"z_smdy": n_latents}
            )  # here just one shared latent space not grouped {"z_smdy": 15, "z_m": 15, "z_y": 4, "z_d": 4},
            variable_col.add(
                "outputs", {"xa_m": "poisson", "xb_y": "gnll"}
            )  # {"xa_m": "poisson", "xb_y": "mse"}) # {"xa_m": "poisson", "xb_d": "mse", "xb_y": "mse"}
            variable_col.add("inputs", ["xa_m", "xb_y"])  # ["xa_m","xb_d"] ["xa_m"],
            variable_col.add("fps_thr", 0.5)  # threshold of filtering units
            variable_col.add(
                "layer_pars",
                {
                    "cnn_comb": [1, 80, 5],
                    "cnn_rnn_enc": [1, 80],
                    "cnn_rnn_dec": [1, 50],
                    "cnn_dec": [1, 20, 5],
                },
            )

            variable_col.add("dec_rnn", True)
            variable_col.add("latent_rec", False)  #  True, rnn for the latents
            variable_col.add("vae_beta", 1)  # True VAE
            variable_col.add("mask_ratio", 0.0000000001)
            variable_col.add("vae_warmup", 50)
            variable_col.add("fps", 15.625)  # binning at this framerate
            # lasso scale both for weight priors on Linear nets
            #  and for the lasso on the sparsity of the latent space
            variable_col.add("lasso_lambda", 0.15)
            variable_col.add("lasso_lr", 5e-3)
            # variable_col.add("max_iters", 25000)
            variable_col.add("print_freq", 200)
            variable_col.add("sessions", [35])
            variable_col.add("min_shared_sessions", 4)
            variable_col.add("ifdimwise_scaling", True)
            variable_col.add("full_mask_prob", 1, 0.5)  #  masked, naive
            variable_col.add("larger_encoder", False)  # True
            variable_col.add("nll_beta", 0.3)  # Beta-NLL

            variable_col.add("batch_T", 150)  # sequence length
            variable_col.add("warmup", 50)
            variable_col.add("xb_y_loss_facs", 1)  # weighting of different losses
            all_run_combinations = variable_col.param_product()
            for ir, runs in enumerate(all_run_combinations):
                args.run_params = runs

                print("---------------------------------------")
                print(" ")
                print(f"run_params {ir} / {len(all_run_combinations)}   ")
                print(" ")
                print("           seed", seed, n_latents)
                print(" ")
                print("---------------------------------------")

                main(args)

# python run_monkey.py --offline 0 --cross_loss 0 --latent_size 40 --seed 110 --max_iters 200

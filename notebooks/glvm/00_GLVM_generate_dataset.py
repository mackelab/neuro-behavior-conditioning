# %%
import numpy as np
import random
import torch
import os
from maskedvae.datasets.datasets import generate_dataset


# %%
class data_args(object):
    """args"""

    def __init__(self):
        self.list_len = 10
        self.n_data_dims = 20
        self.shorttest = 0
        self.datasamples = 10000


args = data_args()

# %%

# specify observation noise and C matrix to ensure
# differences when different variables are masked
bag_of_noises = np.arange(1, 10) / 10
bag_of_Cs = np.arange(7, 14) / 10

organised_sign = False
draw_gaussian = True
c_list = []

np.random.seed(10)

for _ in range(args.list_len):
    if draw_gaussian:
        C = np.random.normal(loc=1.1, scale=0.1, size=args.n_data_dims)
    else:
        C = np.random.choice(bag_of_Cs, size=args.n_data_dims, replace=True)
    if organised_sign:
        C[::2] = -1 * C[::2]  # set every second C value to negative
    else:
        sign = np.random.choice([1, -1], size=args.n_data_dims, replace=True)
        C = sign * C
    C = C.reshape(len(C), 1)
    c_list.append(C)

noise_list = []
for _ in range(args.list_len):
    if draw_gaussian:
        noi = np.random.lognormal(
            np.log(0.7), 0.5, size=args.n_data_dims
        )  # noi = np.random.lognormal(np.log(0.5), 0.5, size=args.n_data_dims)
    else:
        noi = np.random.choice(bag_of_noises, size=args.n_data_dims, replace=True)
    noi = noi.reshape(len(noi), 1)
    noise_list.append(noi)


# %%

# generate the datasets for different C and noise values
for c_id, (C, noise) in enumerate(zip(c_list, noise_list)):

    args.x_dim = C.shape[0]
    args.z_dim = C.shape[1]
    args.d = np.reshape(np.ones(args.x_dim), (args.x_dim, 1))
    args.z_prior = np.stack((np.zeros(args.z_dim), np.ones(args.z_dim)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.C = C
    args.noise = noise

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    dataset_train, dataset_valid, dataset_test = generate_dataset(
        split_ratio=9.0 / 10,
        testset=True,
        C=C,
        x_dim=args.x_dim,
        d=args.d,
        noise=noise,
        z_prior=args.z_prior,
        n_samples=args.datasamples,
        z_dim=args.z_dim,
        plot_data=False,
    )
    flag = ""

    # run again with args.shorttest = True to generate a smaller dataset
    if args.shorttest:
        dataset_train, dataset_valid, dataset_test = generate_dataset(
            split_ratio=1.0 / 2,
            testset=True,
            C=C,
            x_dim=args.x_dim,
            d=args.d,
            noise=noise,
            z_prior=args.z_prior,
            n_samples=100,
            z_dim=args.z_dim,
            plot_data=False,
        )
        flag = "_shorttest"

    # %%
    # save the datasets

    def save_dataset(dataset, foldername, name, c_id, flag):
        filepath = f"{foldername}{name}_{c_id}{flag}.pt"
        torch.save(dataset, filepath)

    # save the datasets
    datapath = f"../../data/glvm/{args.x_dim}_dim_{args.z_dim}_dim/"
    os.makedirs(os.path.dirname(datapath), exist_ok=True)

    save_dataset(dataset_train, datapath, "data_train", c_id, flag)
    save_dataset(dataset_valid, datapath, "data_valid", c_id, flag)
    save_dataset(dataset_test, datapath, "data_test", c_id, flag)


# %%

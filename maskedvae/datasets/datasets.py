import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np


class Gaussian_nD_latent(Dataset):
    """Combined dataset for training and test gaussian data starting from an n-D gaussian

    latent variable
    z ~ N(0,1)
    x = Cz + d + epsilon
    n_samples: number of samples
    x_dim: dimension of the data
    z_dim: dimension of the latent variable
    C: linear transformation matrix
    d: offset
    noise: std of the observation noise

    """

    def __init__(
        self,
        n_samples=6000,
        x_dim=2,
        z_dim=2,
        C=np.array([[4.0, 0.8], [0.8, 1.0]]),
        d=np.array([[1.0], [2.5]]),
        noise=np.array([[1], [1]]),
        z_prior=np.array([[0], [1]]),
        plot_data=False,
    ):

        self.n_samples = n_samples
        self.x_dim = x_dim
        self.z_dim = z_dim

        if self.z_dim == 1:
            # note np.random.normal takes std for scale whereas multivar takes covariance -> square it for multivar cov.
            self.z = np.random.normal(
                loc=z_prior[0], scale=z_prior[1], size=(self.z_dim, self.n_samples)
            )
        else:
            mean_prior = z_prior[0]
            cov_prior = z_prior[1] * z_prior[1] * np.identity(self.z_dim)
            self.z = np.random.multivariate_normal(
                mean_prior, cov_prior, self.n_samples
            ).T

        self.C = C
        self.d = d
        self.noise = noise

        assert self.d.shape[0] == self.x_dim, "d does not have correct x dimension"
        assert self.C.shape[0] == self.x_dim, "C does not have correct x dimension"
        assert self.C.shape[1] == self.z_dim, "C does not have correct z dimension"
        assert (
            self.noise.shape[0] == self.x_dim
        ), "noise does not have correct x dimension"

        self.D = (
            self.noise * self.noise * np.identity(self.x_dim)
        )  # observation variances on diagonal
        self.epsilon_d = np.random.multivariate_normal(
            np.squeeze(self.d), self.D, self.n_samples
        ).T

        self.x = self.C @ self.z + self.epsilon_d
        self.x = self.x.T

        cov_xy = np.cov(self.x.T)
        cov_z = np.cov(self.z)

        cov_xy = np.round(cov_xy, 4)  # round it to include it in the plot
        cov_z = np.round(cov_z, 4)

        if plot_data:
            if self.z_dim == 2:

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(self.z.T[:, 0], self.z.T[:, 1], "o", color="midnightblue")
                plt.xlim(-4, 4)
                plt.ylim(-4, 4)
                plt.xlabel(r"latent $z_0$")
                plt.ylabel(r"latent $z_1$")
                ax.set_aspect("equal")
                plt.title(str(self.D) + " empirical " + str(cov_z))
                plt.show()
                plt.clf()
                plt.close("all")
            elif self.z_dim == 1:
                fig = plt.figure()
                plt.hist(self.z[0, :], bins=30, color="midnightblue", density=True)
                plt.xlabel(r"latent $z$")
                plt.ylabel("frequency")
                plt.title(str(1) + " empirical " + str(cov_z))
                plt.show()
                plt.clf()
                plt.close("all")
            if self.x_dim >= 2:
                cov_gt = self.C @ self.C.T * z_prior[1] * z_prior[1] + self.D
                cov_gt = np.round(cov_gt, 4)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(self.x.T[0, :], self.x.T[1, :], "o", color="darkgray")
                plt.xlabel(r"$x_0$")
                plt.ylabel(r"$x_1$")
                plt.title(str(cov_gt) + " empirical " + str(cov_xy))
                plt.show()
                plt.clf()
                plt.close("all")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        latent = {}
        if self.x is not None:
            sample = torch.tensor(self.x.T[:, idx])

        if self.z is not None:
            latent = torch.tensor(self.z[:, idx])

        return sample, latent


def generate_dataset(
    split_ratio=5.0 / 6,
    testset=False,
    novalid=False,
    binarize=0,
    n_samples=6000,
    z_dim=1,
    x_dim=2,
    C=np.array([[4.0, 0.8], [0.8, 1.0]]),
    d=np.array([[1.0], [2.5]]),
    noise=np.array([[1], [1]]),
    z_prior=np.array([[0], [1]]),
    plot_data=False,
):
    """generate a trining and validation dataset for the model"""

    # setup datasets
    tr_split_len = int(np.floor(split_ratio * n_samples))

    dataset_train = Gaussian_nD_latent(
        n_samples=tr_split_len,
        x_dim=x_dim,
        z_dim=z_dim,
        C=C,
        d=d,
        noise=noise,
        z_prior=z_prior,
        plot_data=plot_data,
    )
    dataset_valid = Gaussian_nD_latent(
        n_samples=n_samples - tr_split_len,
        x_dim=x_dim,
        z_dim=z_dim,
        C=C,
        d=d,
        noise=noise,
        z_prior=z_prior,
        plot_data=plot_data,
    )

    if novalid:
        # for testing make them equal
        dataset_valid = dataset_train

    if testset:
        dataset_test = Gaussian_nD_latent(
            n_samples=n_samples - tr_split_len,
            z_dim=z_dim,
            x_dim=x_dim,
            C=C,
            d=d,
            noise=noise,
            z_prior=z_prior,
            plot_data=plot_data,
        )
    else:
        dataset_test = None

    return dataset_train, dataset_valid, dataset_test

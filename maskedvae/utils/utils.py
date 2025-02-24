import os
import pickle
import numpy as np
import yaml
import json
import copy
from copy import deepcopy
import torch
import csv
from torch import FloatTensor
import math
import itertools
from itertools import product as iter_product


class return_args(object):
    """args"""

    def __init__(self, run):
        self.dir = run
        self.fig_root = "figs"
        self.seed = 0
        self.iters = 10000


def save_run_directory(method, args, run_directory, csv_directory):
    """
    Save the run directory in a CSV file for the given method and args
    """
    csv_file = None

    # Check the conditions to separate masked and naive
    if method == "zero_imputation_mask_concatenated_encoder_only" and args.all_obs == 0:
        csv_file = "masked_runs.csv"
    elif method == "zero_imputation" and args.all_obs == 1:
        csv_file = "all_obs_runs.csv"

    if csv_file:
        # Ensure the CSV directory exists
        os.makedirs(csv_directory, exist_ok=True)
        full_csv_path = os.path.join(csv_directory, csv_file)

        # write header if file does not exist
        if not os.path.exists(full_csv_path):
            with open(full_csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["run_directory"])

        # append the run directory to the CSV file
        with open(full_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([run_directory])


def csv_to_simple_yaml(csv_file_path, yaml_file_path):
    """
    Convert a CSV file to a YAML file
    """
    # Read the CSV file
    with open(csv_file_path, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        data = [row[0] for row in reader]  # Collect only the first column from each row

    # Write to a YAML file
    with open(yaml_file_path, mode="w") as yaml_file:
        yaml.safe_dump(data, yaml_file, default_flow_style=False)


def ensure_directory(path):
    """Ensure directory exists and make it if it does not"""
    os.makedirs(path, exist_ok=True)


def save_pickle(object_, filepath):
    """Save object to filepath using pickle"""
    with open(filepath, "wb") as file:
        pickle.dump(object_, file)


def save_yaml(object_, filepath):
    """Save object to filepath using yaml"""
    with open(filepath, "w") as file:
        yaml.dump(object_, file, default_flow_style=False)


def save_json(object_, filepath):
    """Save object to filepath using json"""
    with open(filepath, "w") as file:
        json.dump(object_, file)


def reverse_non_all_obs_mask(original_mask):
    """reverse the non all obs mask"""
    mask = copy.deepcopy(original_mask)
    # find those rows where not all values are observed and flip the mask
    mask[torch.where(torch.sum(mask, axis=1) / mask.shape[1] != 1)[0], :] = (
        1 - mask[torch.where(torch.sum(mask, axis=1) / mask.shape[1] != 1)[0], :]
    )
    return mask


def mse(actual, predicted):
    """Mean squared error that ensures numpy arrays are used"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    differences = np.subtract(actual, predicted)
    squared_differences = np.square(differences)
    return squared_differences.mean()


def KLD_uvg(var1=np.ones(10), mean1=np.zeros(10), var2=np.ones(10), mean2=np.zeros(10)):
    """return the KLD for two univariate gaussians"""
    return 0.5 * np.log(var2 / var1) + 0.5 * (var1 + (mean1 - mean2) ** 2) / var2 - 0.5


def compute_mse(var, gt_var):
    """Compute the mean squared error between var and gt_var."""
    if np.isscalar(var):
        var = np.full_like(gt_var, var)

    # Check if gt_var is a scalar
    if np.isscalar(gt_var):
        gt_var = np.full_like(var, gt_var)

    return np.mean((var - gt_var) ** 2)


def compute_rmse(var, gt_var):
    """Compute the mean squared error between var and gt_var."""
    if np.isscalar(var):
        var = np.full_like(gt_var, var)

    # Check if gt_var is a scalar
    if np.isscalar(gt_var):
        gt_var = np.full_like(var, gt_var)

    return np.sqrt(np.mean((var - gt_var) ** 2))


def return_posterior_expectation_and_variance_multi_d(batch, args, masked_idx=None):
    """p(z|x_obs) with arbitrary masked indices in higher dimensions

    batch: ground truth data that is passed
    args: args dictionary containing info about
            C: loading matrix
            d: bias term
            sigmas: observation noise terms
            z_prior: mean and variance of z

    gt posterior mean = mu_z + RS^-1 (x - C mu_z - d)
    where R and S can be extracted from A Sigma A^T where Sigma
    contains all observation noise variances on the diagonal and A reflects
    the relations of the generative model x = Cz + d

    other option is to replace the observation  noise terms of missing values with -> infty

    masked_idx = [0,1,2,3]
    for idx_m in masked_idx:
        noise_all[idx_m+1]=np.array([10e24])
    #  and then stick to all values not considering masking further down
    masked_idx = []


    """
    if masked_idx is None:
        masked_idx = []
    # Rest of the code...
    # Construct A
    A = np.eye(args.x_dim + 1)
    A[1 : args.x_dim + 1, 0] = args.C.flatten()

    # Construct Sigma
    # prior std as the first entry
    noise_all = [args.z_prior[1]]
    for n in args.noise:
        noise_all.append(n)

    # diagonal of the squared sigma obs
    Sigma = np.diag(np.array(noise_all).flatten() ** 2)

    # A Sigma A^T
    AS = np.matmul(A, Sigma)
    ASAT = np.matmul(AS, A.transpose())

    """
                | Q    R |
        ASAT =  |        |
                |R^T   S |
    """

    # delete the first row and column
    ASAT0 = np.delete(ASAT, 0, 0)  # rows
    ASAT0 = np.delete(ASAT0, 0, 1)  # columns

    # delete i.e. marginalise out the masked indices
    S = np.delete(ASAT0, masked_idx, 0)
    S = np.delete(S, masked_idx, 1)
    # print(S)

    # get R the first row starting after the diagonal Q
    R = ASAT[0, 1:]
    R_del = np.delete(R, masked_idx, 0)
    # print(R, R_del)

    batch_del = np.delete(batch, masked_idx, 1)
    # print(batch_del)
    C_del = np.delete(args.C, masked_idx, 0)
    # print(C_del)
    d_del = np.delete(args.d, masked_idx, 0)
    # print(d_del)

    # get the mean removed batch data (x - C mu_z - d)
    x_centered = batch_del - C_del.flatten() * args.z_prior[0][0] - d_del.flatten()

    # directly invert the matrix
    S_inv = np.linalg.inv(S)

    mu_z_given_x_direct = args.z_prior[0][0] + R_del @ S_inv @ x_centered.T
    cov_z_given_x = ASAT[0][0] - R_del @ S_inv @ R_del.T

    return mu_z_given_x_direct, cov_z_given_x


# ------------------- utils monkey---------------------


def gpu(x):
    """Transforms numpy array or torch tensor torch torch.cuda.FloatTensor"""
    if torch.cuda.is_available():
        return FloatTensor(x).cuda()
    else:
        return FloatTensor(x).cpu()


def cpu(x):
    """Transforms torch tensor into numpy array"""
    if torch.is_tensor(x):
        return x.cpu().detach().numpy()
    else:
        return x


def get_mask(dim, p=0.7, off=True):
    # get a mask with p percent of the elements set to 1
    if off:  # switch off cross masking
        mask = torch.ones(dim)
    else:
        mask = torch.zeros(dim)
        # returns a tensor of size [int(p*dim)] with values sampled from 0 to dim-1
        # if dim is a matrix tensor the output will be [matrix.shape[0], int(p*dim)]
        # with values between 0 and matrix.shape[1]-1
        if p > 0:
            mask[
                torch.multinomial(torch.ones([dim]), int(p * dim), replacement=False)
            ] = 1

    return gpu(mask)  # .cuda()


def rebin(arr, summed_bins=3, skip_last=True):
    """rebin an array arr along the last axis by summing `summed_bins` consecutive elements
    if skip_last is True, the last element of the array is skipped if the array length is not divisible by `summed_bins`
    """
    if arr.ndim == 2:
        # Calculate padding required
        pad_length = (
            summed_bins - (arr.shape[-1] % summed_bins)
            if arr.shape[-1] % summed_bins != 0
            else 0
        )

        # Pad the array with zeros along the last axis
        arr_padded = np.pad(arr, ((0, 0), (0, pad_length)), mode="constant")

        # Reshape into a 3D array where the last dimension has `summed_bins` elements
        arr_reshaped = arr_padded.reshape((arr.shape[0], -1, summed_bins))

        # Sum along the last axis to get the sum of every `summed_bins` consecutive elements
        rebinned = (
            arr_reshaped.sum(axis=-1)[:, :-1]
            if skip_last and pad_length > 0
            else arr_reshaped.sum(axis=-1)
        )
    elif arr.ndim == 1:
        # Calculate padding required
        pad_length = (
            summed_bins - (len(arr) % summed_bins) if len(arr) % summed_bins != 0 else 0
        )

        # Pad the array with zeros
        arr_padded = np.pad(arr, (0, pad_length), mode="constant")

        # Reshape into a 2D array where each row has `summed_bins` elements
        arr_reshaped = arr_padded.reshape(-1, summed_bins)

        # Sum along the second axis to get the sum of every `summed_bins` consecutive elements
        rebinned = (
            arr_reshaped.sum(axis=1)[:-1]
            if skip_last and pad_length > 0
            else arr_reshaped.sum(axis=1)
        )

    return rebinned


def poisson_sample_from_multiple_predictions(
    spike_train, prediction_rate, n_samples=10, axis=0
):
    """sample several spike trains from a prediction rate"""
    spike_train = spike_train.astype(int)
    spike_train_samples = np.random.poisson(
        prediction_rate.repeat(n_samples, axis=axis)
    )
    return spike_train_samples, spike_train, spike_train.mean(axis=-1)


def compute_corr(pred, truth):
    """Calculates the average Pearson correlation
    coefficient for a number of traces. Ignores empty traces

    Args:
        pred: prediction
        truth: ground truth

    Returns: Pearson correlation coefficient

    """
    ccc = []
    for i in range(pred.shape[-1]):
        if truth[:, i].sum() and pred[:, i].sum():
            cc = np.corrcoef(pred[:, i], truth[:, i])[0, 1]
            if not math.isnan(cc):
                ccc.append(cc)
    return np.array(ccc)


def make_directory(dirname):
    """makes a directory if it does not exist"""
    if not (os.path.exists(dirname)):
        os.mkdir(dirname)


def zip_longest_special(*iterables):
    """Contributed by Artur Speiser"""

    def filter(items, defaults):
        return tuple(d if i is sentinel else i for i, d in zip(items, defaults))

    sentinel = object()
    iterables = itertools.zip_longest(*iterables, fillvalue=sentinel)
    first = next(iterables)
    yield filter(first, [None] * len(first))
    for item in iterables:
        yield filter(item, first)


class param_iteration(object):
    """
    Iterate over parameters.
    Contributed by Artur Speiser

    This class provides methods to iterate over parameters by either zipping them or performing a product operation.

    Attributes:
        keys (list): A list of parameter names.
        vals (list): A list of parameter values.

    Methods:
        add(name, *args): Adds a parameter with its corresponding values to the class.
        param_product(): Performs a product operation on the parameters and returns a list of dictionaries, where each dictionary represents a combination of parameter values.
        param_zip(): Zips the parameters and returns a list of dictionaries, where each dictionary represents a combination of parameter values.

    Example usage:
        iter = param_iteration()
        iter.add('param1', 1, 2, 3)
        iter.add('param2', 'a', 'b')
        combinations = iter.param_product()
        for combination in combinations:
            print(combination)
    """

    def __init__(self):

        self.keys = []
        self.vals = []

    def add(self, name, *args):

        self.keys.append(name)
        self.vals.append(args)

    def param_product(self):

        all_params = []
        for values in iter_product(*self.vals):

            params = dict()
            for i, val in zip(self.keys, values):
                params.update({i: val})

            all_params.append(params)

        return all_params

    def param_zip(self):

        all_params = []
        for values in zip_longest_special(*self.vals):

            params = dict()
            for i, val in zip(self.keys, values):
                params.update({i: val})

            all_params.append(params)

        return all_params


def save_as_pickle(data, file_name, directory):
    """Save data to a pickle file."""
    file_path = os.path.join(directory, f"{file_name}.pkl")
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_individual_pickle(file_name, directory):
    """Load data from an individual pickle file."""
    file_path = os.path.join(directory, f"{file_name}.pkl")
    with open(file_path, "rb") as file:
        return pickle.load(file)


def get_standardise_mean_and_std(train_data):
    """
    get mean and std from the training set
    """
    # squeeze the first dimension
    train_data = np.squeeze(train_data)
    # calculate mean and std for each channel in the training data
    train_means = np.mean(train_data, axis=1)
    train_stds = np.std(train_data, axis=1)

    print(train_means.shape)
    print(train_stds.shape)

    return train_means, train_stds


def apply_standardize(train_data, train_means, train_stds):
    """
    apply the standardization to the data
    """
    # squeeze the first dimension
    train_data = np.squeeze(train_data)
    assert train_data.shape[0] == train_means.shape[0] == train_stds.shape[0]
    # standardize the data
    train_data_standardized = (train_data - train_means[:, np.newaxis]) / train_stds[
        :, np.newaxis
    ]

    return train_data_standardized


def get_min_max_scaler_params(train_data):
    """
    get min and max from the training set
    """
    # squeeze the first dimension
    train_data = np.squeeze(train_data)
    # calculate min and max for each channel in the training data
    train_min = np.min(train_data, axis=1)
    train_max = np.max(train_data, axis=1)

    return train_min, train_max


def apply_min_max_scaler(data, train_min, train_max):
    """
    apply the min-max scaling to the data
    """
    # squeeze the first dimension
    data = np.squeeze(data)
    assert data.shape[0] == train_min.shape[0] == train_max.shape[0]
    # apply min-max scaling
    data_scaled = (data - train_min[:, np.newaxis]) / (
        train_max[:, np.newaxis] - train_min[:, np.newaxis]
    )

    return data_scaled


def scale_datasets(
    train_d,
    valid_d,
    test_d,
    train_latent_means,
    valid_latent_means,
    test_latent_means,
    train_spikes_matrices,
    valid_spikes_matrices,
    test_spikes_matrices,
    test_masks,
    standardize=False,
    min_max=True,
):
    """
    Scale the training, validation, and test datasets using either min-max scaling or standardization.

    Args:
        train_d (array): Training data for D.
        valid_d (array): Validation data for D.
        test_d (array): Test data for D.
        train_latent_means (list of arrays): Training latent means.
        valid_latent_means (list of arrays): Validation latent means.
        test_latent_means (list of arrays): Test latent means.
        train_spikes_matrices (list of arrays): Training spikes matrices.
        valid_spikes_matrices (list of arrays): Validation spikes matrices.
        test_spikes_matrices (list of arrays): Test spikes matrices.
        test_masks (array): Mask for test data.
        standardize (bool): If True, apply standardization.
        min_max (bool): If True, apply min-max scaling.

    Returns:
        tuple: Scaled training, validation, and test datasets.
    """

    if min_max:
        # compute the min and max for the training data
        train_min_d, train_max_d = get_min_max_scaler_params(train_d)
        train_min_latent, train_max_latent = get_min_max_scaler_params(
            train_latent_means[0]
        )
        train_min_spikes, train_max_spikes = get_min_max_scaler_params(
            train_spikes_matrices[0]
        )

        train_d = apply_min_max_scaler(train_d, train_min_d, train_max_d)
        valid_d = apply_min_max_scaler(valid_d, train_min_d, train_max_d)
        test_d = apply_min_max_scaler(test_d, train_min_d, train_max_d)

        # apply the min-max scaling to all datasets
        for i in range(len(test_masks)):
            train_latent_means[i] = apply_min_max_scaler(
                train_latent_means[i], train_min_latent, train_max_latent
            )
            valid_latent_means[i] = apply_min_max_scaler(
                valid_latent_means[i], train_min_latent, train_max_latent
            )
            test_latent_means[i] = apply_min_max_scaler(
                test_latent_means[i], train_min_latent, train_max_latent
            )

            train_spikes_matrices[i] = apply_min_max_scaler(
                train_spikes_matrices[i], train_min_spikes, train_max_spikes
            )
            valid_spikes_matrices[i] = apply_min_max_scaler(
                valid_spikes_matrices[i], train_min_spikes, train_max_spikes
            )
            test_spikes_matrices[i] = apply_min_max_scaler(
                test_spikes_matrices[i], train_min_spikes, train_max_spikes
            )

    elif standardize:
        # compute means and stds for the training data
        train_means_d, train_stds_d = get_standardise_mean_and_std(train_d)
        train_means_latent, train_stds_latent = get_standardise_mean_and_std(
            train_latent_means[0]
        )

        train_d = apply_standardize(train_d, train_means_d, train_stds_d)
        valid_d = apply_standardize(valid_d, train_means_d, train_stds_d)
        test_d = apply_standardize(test_d, train_means_d, train_stds_d)

        for i in range(len(test_masks)):
            train_latent_means[i] = apply_standardize(
                train_latent_means[i], train_means_latent, train_stds_latent
            )
            valid_latent_means[i] = apply_standardize(
                valid_latent_means[i], train_means_latent, train_stds_latent
            )
            test_latent_means[i] = apply_standardize(
                test_latent_means[i], train_means_latent, train_stds_latent
            )

            train_spikes_matrices[i] = apply_standardize(
                train_spikes_matrices[i], train_means_latent, train_stds_latent
            )
            valid_spikes_matrices[i] = apply_standardize(
                valid_spikes_matrices[i], train_means_latent, train_stds_latent
            )
            test_spikes_matrices[i] = apply_standardize(
                test_spikes_matrices[i], train_means_latent, train_stds_latent
            )

    return (
        train_d,
        valid_d,
        test_d,
        train_latent_means,
        valid_latent_means,
        test_latent_means,
        train_spikes_matrices,
        valid_spikes_matrices,
        test_spikes_matrices,
    )


def copy_and_shuffle(data, shuffleaxis=1):
    """
    Copy the data and shuffle it along the specified axis.

    Args:
        data (ndarray): Data to shuffle. Can be a multi-dimensional numpy array.
        shuffleaxis (int): Axis along which to shuffle the data.

    Returns:
        ndarray: Shuffled data along the specified axis.
    """
    # Create a copy of the data
    data_copy = data.copy()

    # Check the shape of the data along the specified shuffle axis
    if shuffleaxis >= data.ndim:
        raise ValueError(
            f"shuffleaxis {shuffleaxis} is out of bounds for array of dimension {data.ndim}"
        )

    # Shuffle along the specified axis
    if shuffleaxis == 0:
        # Shuffle rows or elements along the first axis
        np.random.shuffle(data_copy)
    else:
        # Shuffle along any other axis by shuffling the indices along that axis
        indices = np.arange(data_copy.shape[shuffleaxis])
        np.random.shuffle(indices)
        data_copy = np.take(data_copy, indices, axis=shuffleaxis)

    return data_copy


def standardize_fly_data(data):
    """
    standardizes data to mean 0 and std 1 per each feature
     channel and returns mean and std for backtransformation later
    args:
        data (array-like): array in shape(n_data, sequ_length, n_features)
    """
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    standardized_data = (data - mean) / std

    return standardized_data, mean, std


def rescale_fly_from_std(data, mean, std):
    """
    rescales original data with mean 0 and std 1
    args:
        data (array-like): array in shape(n_data, sequ_length, n_features)
        mean: means per channel in shape (n_data, 1, n_features)
        std: standard deviations per channel in shape (n_data, 1, n_features)
    """
    return data * std + mean


def squared_error(y_true, y_predict):
    """compute the squared error between two arrays"""
    return np.square(np.subtract(y_true, y_predict))


if __name__ == "__main__":
    # go to base directory
    csv_to_simple_yaml("./runs/masked_runs.csv", "./data/glvm/runs_masked_glvm.yaml")
    csv_to_simple_yaml("./runs/all_obs_runs.csv", "./data/glvm/runs_all_obs_glvm.yaml")

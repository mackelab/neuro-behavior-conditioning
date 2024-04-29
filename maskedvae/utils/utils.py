import os
import pickle
import numpy as np
import yaml
import json
import copy
from copy import deepcopy
import torch
import csv


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


if __name__ == "__main__":
    # go to base directory
    csv_to_simple_yaml("./runs/masked_runs.csv", "./data/glvm/runs_masked_glvm.yaml")
    csv_to_simple_yaml("./runs/all_obs_runs.csv", "./data/glvm/runs_all_obs_glvm.yaml")

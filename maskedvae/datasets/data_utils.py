import copy
import gc
import os

import numpy as np
import yaml

import h5py
from maskedvae.utils.utils import standardize_fly_data


def get_fly_data_splits(args, min_buffer=10):
    """get the fly walking training, validation and test splits
        by chopping the data into shorter sequences
        separated by buffers after standardizing the data

    Args:
        args (dict): arguments
        min_buffer (int): minimum buffer between sequences

    Returns:
        X_train (nd array): training data
        X_val (nd array): validation data
        X_test (nd array): test data
        mean_rescale (nd array): mean used for rescaling
        std_rescale (nd array): standard deviation used for rescaling
        new_args (dict): updated arguments
    """

    # do not change args in this file - make a copy
    new_args = copy.deepcopy(args)

    with h5py.File("data/fly/Fly_DLC_behavior_tracking.h5", "r") as f:
        data = f["data"][:]

    data_dim = data.shape[1]

    n_seq_slices = int(
        np.floor(data_dim / (args.seq_len + min_buffer))
    )  # how many data slices fit into the original sequence length
    wasted_samples = (
        data_dim - n_seq_slices * args.seq_len
    )  # how many samples are wasted

    # include a buffer to avoid leaking info from training to test samples
    # Savitzky - Golay - Filter was applied with
    # a window length of 7 and polynomial order of two
    buffer = (
        int(wasted_samples / n_seq_slices) - 1
    )  # how many samples are in the buffer

    # for testing on a laptom reduce the data size
    if args.laptop == 1:
        new_args.range_len = 1000
    else:
        new_args.range_len = len(data)

    # -----------------------------------------
    # chopping off data into buffered sequences
    # -----------------------------------------
    data_sampled = []
    data_sampled_with_buffer = []

    for i in range(args.range_len):
        # cut into sequences of length args.seq_len
        for temp in range(n_seq_slices):
            data_sampled.append(
                data[
                    i,
                    (1 + temp) * buffer
                    + temp * args.seq_len : (temp + 1) * (args.seq_len + buffer),
                ]
            )
            if i == 0:
                # store which indices one uses for extracting the samples
                data_sampled_with_buffer.append(
                    [
                        (1 + temp) * buffer + temp * args.seq_len,
                        (temp + 1) * (args.seq_len + buffer),
                    ]
                )

    # free RAM
    del data
    gc.collect()

    # convert to numpy arrays
    data_sampled = np.array(data_sampled)

    # Standardise the dataset - each sequence separately
    # so we do not need to consider the train test valid split
    data_sampled, mean_rescale, std_rescale = standardize_fly_data(data_sampled)

    # -----------------------------------------
    #  split into train and test
    # -----------------------------------------

    new_args.split_index = int(args.train_ratio * len(data_sampled))
    X_train, X_val = (
        data_sampled[: new_args.split_index],
        data_sampled[new_args.split_index :],
    )

    # split the validation set into validation and test set
    if args.testset == 1:
        new_args.split_index_val = int(args.valid_ratio * len(X_val))
        X_test = X_val[new_args.split_index_val :]
        X_val = X_val[: new_args.split_index_val]
    else:
        # full length
        new_args.split_index_val = len(data_sampled) - new_args.split_index
        X_test = None
        print("Warning - No test set used!")

    # free ram
    del data_sampled
    gc.collect()

    return X_train, X_val, X_test, mean_rescale, std_rescale, new_args


def main():
    """Main function run the fly data split function"""

    with open(os.path.join("configs/fly/fit_config.yml"), "r") as f:
        args = yaml.load(f, Loader=yaml.Loader)

    args.seq_len = 48
    args.home = os.path.expanduser("~") + "/"

    X_train, X_val, X_test, _, _, _ = get_fly_data_splits(args)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)


if __name__ == "__main__":
    main()

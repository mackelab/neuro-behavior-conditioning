import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime
from scipy.interpolate import CubicSpline
from maskedvae.utils.utils import gpu


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
    """generate a training and validation dataset for the model"""

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


class Primate_Reach(Dataset):
    """Dataset class for Primate reach data of O'Doherty et al. 2017

    this class was contributed by Artur Speiser

    Combined dataset with neural and behavioral data as well as available group information
    This dataset contains 43 sessions from two monkeys. Trials 0:33 are from 'Indy' and mostly only contain M1 recordings.
    The remaining sessions are from 'Loco' and all have both M1 and S1 data.

    'xa_m' is M1 activity
    'xa_s' is S1 activity
    'xa_j' is all available activity traces (needed ?)
    'xb_y' is finger position (x,y coordinates)
    'xb_d' is finger speed (x,y coordinates)
    'xb_u' is target (x,y coordinates)

    Parameters
    ----------
    data : list of dicts
        all_sessions.pkl
    session_list : list of ints
        Indicate the sessions that should be processed
    fps : float
        Frequency at which to bin the data
    fr_threshold : float
        Cutoff frequency. Channels with lower firing rate are discarded

    Attributes
    ----------
    behave_rec_hz : float
        Frequency at which behaviour data is recorded
    n_channels : int
        Number of channels of the extracellular recordings
    n_units :
        Number of sorted units per channel
    n_traces : dict
        Number of traces for the different neural / behaviour recordings
    fps : float
        Target fps
    fr_threshold : float
        Minimum firing rate
    session_list : list of ints
        Indicates the session stored
    spike_bs : list of np.array
        Binned spike trains
    spike_mask : list of boolean masks
        Boolean mask that indicates which channels have active neurons
    curser_pos : list of np.array
        Curser position
    finger_pos : list of np.array
        Finger position
    curser_pos : list of np.array
        Curser position (same as Finger position but only x,y)
    session_name : list of str
        Name of the session (contains animal and date)
    session_data : list of datetime
        Dates of the sessions
    animal : list of str
        Indy or Loco
    """

    def __init__(self, data, session_list, fps=100, fr_threshold=0.5):

        self.behave_rec_hz = 250  # sampling rate of both neural and behavior data
        self.n_channels = 192
        self.n_units = 5  # sorted into units (up to five, including a "hash" unit),
        self.n_traces = {
            "xa_j": self.n_channels * self.n_units,
            "xa_m": 0.5 * self.n_channels * self.n_units,
            "xa_s": 0.5 * self.n_channels * self.n_units,
            "xb_y": 2,
            "xb_d": 2,
            "xb_u": 2,
        }
        # exactly half the channels are in M1 and half in S1
        self.slices = {
            "xa_j": np.s_[:],
            "xa_m": np.s_[: 0.5 * self.n_channels * self.n_units],
            "xa_s": np.s_[0.5 * self.n_channels * self.n_units :],
        }

        self.fr_threshold = fr_threshold
        self.fps = fps
        self.session_list = session_list

        self.spike_bs = []
        self.spike_mask = []
        self.cursor_pos = []
        self.finger_pos = []
        self.cursor_speed = []
        self.target_pos = []
        self.session_name = []
        self.session_date = []
        self.animal = []
        self.T = []
        self.n_bins = []
        # select the respective sessions
        for n, session in enumerate([data[i] for i in self.session_list]):

            # save the session as a pickle file
            # with open(f'./data/Money_reach_{session["session_name"]}.pkl', 'wb') as f:
            #     pickle.dump(session, f)

            # # load the session
            # with open(f'./data/Money_reach_{session["session_name"]}.pkl', 'rb') as f:
            #     session = pickle.load(f)

            self.session_name.append(session["session_name"])
            self.session_date.append(
                datetime(
                    int(self.session_name[-1][5:9]),
                    int(self.session_name[-1][9:11]),
                    int(self.session_name[-1][11:13]),
                )
            )
            self.animal.append(0 if "indy" in session["session_name"] else 1)
            # spike time range make a list for each channel and unit
            session_spike_ts = [[] for _ in range(self.n_traces["xa_j"])]

            # session data is split into units first then channels
            for c in range(self.n_channels):  # channels
                for u in range(self.n_units):  # units for u 0:4
                    if (
                        len(session["spikes"]) > u
                    ):  # if session contains more units than u
                        if (
                            len(session["spikes"][u]) > c
                        ):  # if session contains more channels than c
                            if (
                                len(session["spikes"][u][c]) == 1
                            ):  # check len == 1 to avoid empty lists
                                # session['spikes][units][channels][0] is spike time array
                                # now list spikes 0-4 for channel 0 then 5-9 for channel 1 etc
                                session_spike_ts[c * self.n_units + u] = session[
                                    "spikes"
                                ][u][c][0]
            # get a session mask for all non-empty channels
            session_mask = [len(s) > 0 for s in session_spike_ts]

            # get the time range of the session
            tmin = np.min(
                [s[0] for s in [session_spike_ts[b] for b in np.where(session_mask)[0]]]
            )
            tmax = np.max(
                [
                    s[-1]
                    for s in [session_spike_ts[b] for b in np.where(session_mask)[0]]
                ]
            )
            T = tmax - tmin  # ifirst spike and last spike
            # behavior data
            tb_min = session["t"][0][0]  # Times where we have behavior data
            tb_max = session["t"][0][-1]
            # bin the spike trains according to the target fps
            bin_ts = np.arange(tmin, tmax, 1 / self.fps)
            min_ind = np.where(bin_ts > tb_min)[0].min()
            max_ind = np.where(bin_ts < tb_max)[0].max()
            bin_ts = bin_ts[min_ind:max_ind]
            # append the time range of the session
            self.T.append(max_ind - min_ind)

            for i in np.where(session_mask)[0]:
                # if overall firing rate is too low, discard the channel
                if (
                    len(session_spike_ts[i]) / T < fr_threshold
                ):  # number of spikes / total time
                    session_spike_ts[i] = []
                    session_mask[i] = False  # set the mask to false

            # bin the spike trains
            session_spike_bs = np.zeros(
                [self.n_channels * self.n_units, max_ind - min_ind], dtype=np.uint8
            )
            # for all channels with active neurons
            for i in np.where(session_mask)[0]:
                session_spike_bs[i] = np.bincount(
                    np.array(
                        ((list(session_spike_ts[i]) + [tmax]) - tmin) * self.fps,
                        dtype="int",
                    )
                )[
                    min_ind:max_ind
                ]  # add tmax at the end to ensure all spiketraces have same length

            # downsampled behaviour: only take between the 8th and 9th samples
            behave_bins = np.array(
                np.arange(len(bin_ts)) * self.behave_rec_hz / self.fps, dtype=int
            )

            self.spike_bs.append(session_spike_bs)
            self.spike_mask.append(session_mask)

            lag = 0
            # cubis spline flow see Jensen et al. 2021 Hennequin lab pGPFA
            query_points = (
                np.arange(len(session["cursor_pos"][0, :]))
                * self.behave_rec_hz
                / self.fps
            )  # measured in ms
            spline = CubicSpline(
                query_points, session["cursor_pos"], axis=1
            )  # fit cubic spline to x
            session["cursor_speed"] = spline(
                query_points + lag, 1
            )  # velocity at time+delay ,1): 1st derivative indicates velocity

            # downsampled behaviour data
            self.cursor_speed.append(session["cursor_speed"][:, behave_bins])
            self.cursor_pos.append(session["cursor_pos"][:, behave_bins])
            self.finger_pos.append(session["finger_pos"][:, behave_bins])
            self.target_pos.append(session["target_pos"][:, behave_bins])
            self.n_bins.append(len(bin_ts))

    def filt_units(self, min_sessions):
        """Filter out all units that are not active in at least min_sessions sessions"""

        self.filt_inds = np.where(
            (np.array(self.spike_mask) * 1).sum(0) >= min_sessions
        )[0]
        for i in range(len(self)):
            self.spike_bs[i] = self.spike_bs[i][self.filt_inds]
            self.spike_mask[i] = list(np.array(self.spike_mask[i])[self.filt_inds])

        self.n_traces["xa_j"] = len(self.filt_inds)
        self.n_traces["xa_m"] = len(
            np.where(self.filt_inds < self.n_channels / 2 * self.n_units)[0]
        )
        self.n_traces["xa_s"] = len(
            np.where(self.filt_inds >= self.n_channels / 2 * self.n_units)[0]
        )
        self.slices["xa_m"] = np.s_[: self.n_traces["xa_m"]]
        self.slices["xa_s"] = np.s_[self.n_traces["xa_m"] :]

    def filt_times_ind(self, start=0, end=None):
        """Filters all time traces by given start and end points"""

        for arr in [
            self.spike_bs,
            self.cursor_pos,
            self.finger_pos,
            self.target_pos,
            self.cursor_speed,
        ]:
            # for i in range(len(arr)):
            for i, iarr in enumerate(arr):
                arr[i] = arr[i][..., start:end]
                self.T[i] = arr[i].shape[-1]

    def filt_times_p(self, percentile=0.8, last=0):
        """Filters all time traces by a given percentile
        Parameters
        ----------
        percentile : float between 0 and 1
            Percentage of the time that should be kept
        last : bool
            When false return beginning, when true return end of traces
        """

        for arr in [
            self.spike_bs,
            self.cursor_pos,
            self.finger_pos,
            self.target_pos,
            self.cursor_speed,
        ]:
            # change to enumerate for i in range(len(arr)):
            for i, iarr in enumerate(arr):
                T = int(arr[i].shape[-1] * percentile)
                if last:
                    arr[i] = arr[i][..., T:]
                else:
                    arr[i] = arr[i][..., :T]
                self.T[i] = arr[i].shape[-1]

        if last:
            self.n_bins = [int(i * (1 - percentile)) for i in self.n_bins]
        else:
            self.n_bins = [int(i * percentile) for i in self.n_bins]

    def filt_times_percent_range(self, per_start=0.3, per_end=0.7):
        """Filters all time traces by a percentile range for each selected session individually
        Parameters
        ----------
        per_start : float between 0 and 1
            Start of the percentage of the time that should be kept
        per_end= : float between 0 and 1
            End of the percentage of the time that should be kept
        """
        assert per_end > per_start, "per_end must be larger than per_start"

        for arr in [
            self.spike_bs,
            self.cursor_pos,
            self.finger_pos,
            self.target_pos,
            self.cursor_speed,
        ]:
            # change to enumerate for i in range(len(arr)):
            for i, iarr in enumerate(arr):
                T_start = int(arr[i].shape[-1] * per_start)
                T_end = int(arr[i].shape[-1] * per_end)
                arr[i] = arr[i][..., T_start:T_end]
                self.T[i] = arr[i].shape[-1]
        self.n_bins = [int(i * (per_end - per_start)) for i in self.n_bins]

    def print_act_units(self):
        """Print the number of active units in each session"""
        for i in range(len(self)):
            print(len(np.where(self.spike_bs[i].sum(-1) > 0)[0]))

    def get_train_batch(self, batch_size=20, T=300, to_gpu=False):
        """Draw a training batch, drawing random session and random starting times
        Parameters
        ----------
        batch_size : int
            Batch size
        T : int
            Length of training sample
        to_gpu : bool
            Whether to load the tensor to the gpu

        Returns
        -------
            batch
                Tensor of size batch_size x T
        """
        batch = {
            "xa_j": [],
            "xa_m": [],
            "xa_s": [],
            "xb_y": [],
            "xb_d": [],
            "xb_u": [],
            "i": [],
            "t": [],
        }
        session = np.random.choice(len(self))
        # session = 1
        for _ in range(batch_size):
            # randomly choose the starting time ensuring it is at least seqlen away from the end
            t_min = np.random.choice(self.spike_bs[session].shape[-1] - T, 1)[0]
            batch["xa_m"].append(
                self.spike_bs[session][: self.n_traces["xa_m"], t_min : t_min + T]
            )
            batch["xa_s"].append(
                self.spike_bs[session][
                    self.n_traces["xa_m"] : self.n_traces["xa_m"]
                    + self.n_traces["xa_s"],
                    t_min : t_min + T,
                ]
            )
            batch["xa_j"].append(self.spike_bs[session][:, t_min : t_min + T])
            batch["xb_y"].append(self.cursor_pos[session][:, t_min : t_min + T])
            batch["xb_d"].append(
                self.cursor_speed[session][:, t_min : t_min + T]
            )  # padded_diff(self.cursor_pos[session][:,t_min:t_min+T]))
            batch["xb_u"].append(self.target_pos[session][:, t_min : t_min + T])
            batch["t"].append(t_min)
        for k in batch.keys():
            batch[k] = np.array(batch[k])
            # gpu now also considers making it a tensor but on cpu
            batch[k] = gpu(batch[k])

        batch["i"] = session
        return batch

    def get_session(self, idx, t_slice=np.index_exp[:], outputs=None, to_gpu=False):
        """return the session data
        ----------
        idx : int
            Indicates session (out of the sessions initially stored)
        t_slice : np.index_ex
            Slice each trace
        outputs : list of str or None
            Indicate which traces to return. If None return all traces.
        to_gpu : bool
            Whether to load the tensor to the gpu
        Returns
        -------
            data
                Dict of traces
        """
        if outputs is None:
            outputs = self.n_traces.keys()

        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = {}

        if "xa_j" in outputs:
            data["xa_j"] = self.spike_bs[idx][None, :, t_slice[0]]
        if "xa_m" in outputs:
            data["xa_m"] = self.spike_bs[idx][None, : self.n_traces["xa_m"], t_slice[0]]
        if "xa_s" in outputs:
            data["xa_s"] = self.spike_bs[idx][
                None,
                self.n_traces["xa_m"] : self.n_traces["xa_m"] + self.n_traces["xa_s"],
                t_slice[0],
            ]
        if "xb_y" in outputs:
            data["xb_y"] = self.cursor_pos[idx][None, :, t_slice[0]]
        if "xb_d" in outputs:
            data["xb_d"] = self.cursor_speed[idx][
                None, :, t_slice[0]
            ]  # padded_diff(self.cursor_pos[idx][None,:,t_slice[0]])
        if "xb_u" in outputs:
            data["xb_u"] = self.target_pos[idx][None, :, t_slice[0]]

        if to_gpu:
            for k in data.keys():
                data[k] = gpu(data[k])

        data["i"] = idx

        return data

    def plot_behavior(self):
        """Plot the finger and target position for each session"""

        print(self.n_traces)
        plt.figure(figsize=(len(self) * 7, 4))
        for t in range(len(self)):
            plt.subplot(1, len(self), t + 1)
            plt.plot(self.cursor_pos[t][0], label="cursor")
            plt.plot(self.finger_pos[t][0], label="finger")
            plt.plot(self.target_pos[t][0], label="target")
            plt.title(self.session_name[t])
            plt.legend()

    def __len__(self):
        """Return the number of sessions"""
        return len(self.session_list)

    def __getitem__(self, idx):
        """Not in use"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return None


def load_train_valid_test_data(
    model,
    data,
    plot=False,
    start_train=0,
    start_test=0.7,
    start_valid=0.8,
    end_valid=1,
    fr_threshold=0.5,
):
    """Load training and validation data (using the same filtering as for training)
    Args:
        model (model class instance): Trained model
        data (dict): all_sessions.pkl
        plot (bool): Whether to plot behaviour
        start_train (float): between 0 and 1
        start_test (float): between 0 and 1
        start_valid (float): between 0 and 1
    Returns:
        PD_train: train data sets
        PD_valid: test data sets
    """
    assert start_test > start_train, "start_test must be larger than start_train"
    assert start_valid >= start_test, "per_end must be larger than per_start"

    print(model.min_shared_sessions)
    print(model.fps)

    PD_train = Primate_Reach(
        data, model.sessions, fps=model.fps, fr_threshold=fr_threshold
    )
    PD_train.filt_units(min_sessions=model.min_shared_sessions)
    if plot:
        PD_train.plot_behavior()
    PD_train.filt_times_percent_range(per_start=start_train, per_end=start_test)

    PD_test = Primate_Reach(
        data, model.sessions, fps=model.fps, fr_threshold=fr_threshold
    )
    PD_test.filt_units(min_sessions=model.min_shared_sessions)
    print(PD_test.n_bins)
    PD_test.filt_times_percent_range(per_start=start_test, per_end=start_valid)
    print(PD_test.n_bins)

    PD_valid = Primate_Reach(
        data, model.sessions, fps=model.fps, fr_threshold=fr_threshold
    )
    PD_valid.filt_units(min_sessions=model.min_shared_sessions)
    if plot:
        PD_valid.plot_behavior()
    PD_valid.filt_times_percent_range(per_start=start_valid, per_end=end_valid)
    print(PD_valid.n_bins)

    return PD_train, PD_test, PD_valid


def test_Prim_Reach(data, sessions, fps=30, plot=True):
    """Test Primate_Reach class
    Args:
        data (dict): all_sessions.pkl
        sessions (list): list of sessions to use
        fps (int): Frames per second
        plot (bool): Whether to plot behaviour
    Returns:
        PD: Primate_Reach class instance
    """

    PD = Primate_Reach(data, sessions, fps=15.625)
    PD.filt_units(min_sessions=1)
    if plot:
        PD.plot_behavior()
    PD.filt_times_p(0.7, last=0)

    PD.get_train_batch()

    PD = Primate_Reach(data, sessions, fps=40)
    PD.filt_units(min_sessions=1)
    if plot:
        PD.plot_behavior()
    PD.filt_times_p(0.7, last=0)

    PD.get_train_batch()
    return PD


def main():
    """Main function read in session data and run test_Prim_Reach"""

    data_dir = "./data/monkey/"

    # load the data from all sessions
    with open(data_dir + "all_sessions.pkl", "rb") as f:
        data = pickle.load(f)

    sessions = [37, 40]
    test_Prim_Reach(data, sessions, fps=40, plot=True)


if __name__ == "__main__":
    main()

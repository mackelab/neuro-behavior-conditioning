import copy

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

# pylint: disable=invalid-name


class GLVM_VAE(nn.Module):
    """ " Gaussian VAE example
    Encoding: p(z|x,m)
    Sampling: reparametrise
    Decoder: p(x|z)

    the networks should know which data is actually missing and which one is not

    combination indicates if the mask embedding should be concatenated or multiplied
    """

    def __init__(
        self,
        input_size,
        latent_size,
        args,
        nonlin=nn.Identity(),
        impute_missing=False,
        init="norm",
        scale=0.1,
        uncertainty=False,
        combination="regular",
        n_hidden=10,
        freeze_gen=False,  # freeze the generative model
        C=None,
        d=None,
        noise_std=None,
        n_masks=2,
        dropout=0.0,
    ):
        print("GLVM_VAE")
        super().__init__()

        assert type(latent_size) == int
        assert type(input_size) == int

        self.latent_size = latent_size
        self.added_std = args.added_std

        # should mask be passed to network
        self.pass_mask = args.pass_mask
        self.pass_mask_decoder = (
            args.pass_mask_decoder
        )  # also to decoder default same as pass_mask

        # uncertainty output predictions learned
        self.uncertainty = uncertainty

        # learned imputation
        self.impute_missing = impute_missing
        self.init = init
        self.scale = scale
        self.impute = ImputeMissing(
            input_size=input_size, init=self.init, scale=self.scale
        )

        # init encoder and decoder
        self.encoder = GLVM_Encoder(
            input_size=input_size,
            latent_size=latent_size,
            masked=self.pass_mask,
            nonlin=nonlin,
            combination=combination,
            n_hidden=n_hidden,
            n_masks=n_masks,
            dropout=dropout,
        )
        self.decoder = GLVM_Decoder(
            input_size=input_size,
            latent_size=latent_size,
            masked=self.pass_mask_decoder,
            nonlin=nonlin,
            uncertainty=self.uncertainty,
            n_hidden=n_hidden,
            freeze_gen=freeze_gen,
            C=C,
            d=d,
            noise_std=noise_std,
        )

    def forward(self, x, m=None):

        if self.impute_missing:
            x = self.impute(x, m)
        # encode
        means, log_var = self.encoder(x, m)
        # reparametrization trick
        z = self.reparameterize(means, log_var)
        # decode
        recon_mean, recon_var = self.decoder(z, m)

        return recon_mean, recon_var, means, log_var, z, None

    def reparameterize(self, mu, log_var):
        """reparameterization trick with added value to std to ensure
        posterior variance is never below added_std^2 often 1e-3 taken"""

        std = torch.exp(0.5 * log_var) + self.added_std
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z, m=None):
        """Decode step of the VAE p(x|z)"""

        recon_mean, recon_var = self.decoder(z, m)

        return recon_mean, recon_var

    def inference(self, x, m=None):
        """Inference step of the VAE p(z|x,m)"""
        if self.impute_missing:
            x = self.impute(x, m)

        means, log_var = self.encoder(x, m)
        return means, log_var


class GLVM_Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        masked,
        nonlin=nn.ReLU(),
        combination="concat",
        n_hidden=10,
        n_masks=2,
        dropout=0.0,
    ):
        super().__init__()
        self.pass_mask = masked

        # set up encoder modules
        self.input_size = input_size
        self.latent_size = latent_size
        self.n_hidden = n_hidden
        self.nonlin = nonlin
        self.n_masks = n_masks

        # dropout contrast to the decoder
        self.dropout_val = dropout
        self.dropout = nn.Dropout(self.dropout_val)

        self.combination = combination
        # implement different methods to combine mask and input
        if self.combination == "concat":
            # concatenation doubles the input size
            self.input_size_concat = int(2 * self.input_size)
        elif self.combination == "multiply" or self.combination == "add":
            # adding and multiplying keep the input size
            self.input_size_concat = self.input_size
        else:
            # setup regular encoder modules
            self.mask_embed_size = self.input_size
            # add mask embedding size to input size
            self.input_size_concat = self.input_size + self.mask_embed_size
            # specify learneable linear mask embedding
            self.mask_embdeding = nn.Linear(self.input_size, self.mask_embed_size)
            # first expand the dimensions linearly
            self.relu_hidden_dim_expansion = nn.Sequential(
                nn.Linear(self.input_size_concat, self.n_hidden), nn.Identity()
            )
            # then reduce the dimensions again non linearly
            self.relu_hidden_dim_reduction = nn.Sequential(
                nn.Linear(self.n_hidden, self.input_size_concat),
                self.nonlin,
                nn.Linear(self.input_size_concat, self.input_size_concat),
                self.nonlin,
            )
            # perform linear mapping to mean and non linear to log var
            self.linear_means = nn.Linear(self.input_size_concat, self.latent_size)
            self.linear_log_var = nn.Sequential(
                nn.Linear(self.input_size_concat, 10),
                self.nonlin,
                nn.Linear(10, self.latent_size),
            )

        if self.combination in ["concat", "multiply", "add"]:

            # additional modules for advanced combination of mask and input
            self.mask_embed_hidden = int(np.round(self.input_size / 2))
            # mask embedding learneable
            self.mask_embed_advanced = nn.Sequential(
                nn.Linear(self.input_size, self.mask_embed_hidden),
                nn.Linear(self.mask_embed_hidden, self.mask_embed_hidden),
                self.nonlin,
                nn.Linear(self.mask_embed_hidden, self.input_size),
            )
            #  input embedding
            self.input_embed_advanced = nn.Sequential(
                nn.Linear(self.input_size, self.n_hidden),
                nn.Linear(self.n_hidden, self.n_hidden),
                self.nonlin,
                nn.Linear(self.n_hidden, self.input_size),
            )

            # linear mapping before the combination with the mask
            self.linear_mean_branch = nn.Linear(self.input_size, self.input_size)
            # different linear mapping for the posterior variance to allow for cutting out x - effects
            self.linear_log_var_branch = nn.Linear(
                self.input_size, self.input_size
            )  # here no softmax as logvae not sigma

            self.mean_branch_transform = nn.Sequential(
                nn.Linear(self.input_size_concat, self.mask_embed_hidden),
                nn.Linear(self.mask_embed_hidden, self.mask_embed_hidden),
                self.nonlin,
                nn.Linear(self.mask_embed_hidden, self.input_size_concat),
            )

            # perform linear read out
            self.linear_means = nn.Linear(self.input_size_concat, self.latent_size)
            self.linear_log_var = nn.Linear(
                self.input_size_concat, self.latent_size
            )  # here no softmax as logvae not sigma

    def forward(self, x, m=None):
        """Computes the forward pass through the Encoder.
        different methods to combine mask and input are supported"""

        if not self.pass_mask:
            # keep the dimensions the same _> also without mask pass 0 -> no information
            m = torch.ones_like(x)

        if self.combination in ["concat", "multiply", "add"]:
            # alternative methods to combine mask and input

            # embed the mask
            m = self.mask_embed_advanced(m)
            # embed input
            x = self.input_embed_advanced(x)

            mx = self.linear_mean_branch(x)
            varx = self.linear_log_var_branch(x)

            # concatenate the mask and the input or multiply
            if self.combination == "concat":
                mx = torch.cat((mx, m), dim=1)
                varx = torch.cat((varx, m), dim=1)
            elif self.combination == "multiply":
                mx = mx * m
                varx = varx * m
            elif self.combination == "add":
                mx = mx + m
                varx = varx + m

            mx = self.mean_branch_transform(mx)
            # map to mean and log var

            means = self.linear_means(mx)
            log_vars = self.linear_log_var(varx)
        else:  # regular encoder method
            # embed the mask first that it can
            m = self.mask_embdeding(m)

            # dropping out in the input layer
            if self.dropout_val != 0.0:
                x = self.dropout(x)
            # concatenate the mask and the input
            x = torch.cat((x, m), dim=1)

            # MLP dim expansion and reduction
            x = self.relu_hidden_dim_expansion(x)
            x = self.relu_hidden_dim_reduction(x)

            # map to mean and log var
            means = self.linear_means(x)
            log_vars = self.linear_log_var(x)

        return means, log_vars


class GLVM_Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        masked,
        nonlin=nn.ReLU(),
        uncertainty=False,
        n_hidden=10,
        freeze_gen=False,
        C=None,
        d=None,
        noise_std=None,
    ):
        super().__init__()

        if (
            freeze_gen
        ):  # if we freeze the generative model ensure the parameters are passed
            assert C is not None, "Provide C for Cz + d"
            assert d is not None, "Provide d for Cz + d"
            if (
                uncertainty
            ):  # if we also estimate the predictive variance ensure the observation noise term is passed
                assert noise_std is not None, "Provide noise_std"

        # layer specifications
        self.pass_mask = masked
        self.input_size = input_size
        self.uncertainty = uncertainty
        self.mask_embed_size = int(np.round(input_size / 2))
        self.input_size_concat = self.input_size + self.mask_embed_size
        self.n_hidden = n_hidden
        self.nonlin = nonlin

        self.freeze_gen = freeze_gen
        if freeze_gen:
            self.C = C
            self.d = d
            self.gt_variance = noise_std * noise_std
        else:
            # mask embedding
            self.mask_embdeding = nn.Linear(self.input_size, self.mask_embed_size)
            # first expand the dimensions linearly
            self.relu_hidden_dim_expansion = nn.Sequential(
                nn.Linear(self.input_size_concat, self.n_hidden), nn.Identity()
            )
            # then reduce the dimensions again non linearly
            self.relu_hidden_dim_reduction = nn.Sequential(
                nn.Linear(self.n_hidden, self.input_size_concat),
                self.nonlin,
                nn.Linear(self.input_size_concat, self.input_size_concat),
                self.nonlin,
            )

            # set up decoder modules
            self.dim_increase = nn.Linear(latent_size, self.input_size)
            # output mean and variance to dataspace softplus to ensure positive noise term
            self.mean_linear = nn.Linear(self.input_size_concat, self.input_size)
            self.var_linear = nn.Sequential(
                nn.Linear(self.input_size_concat, self.input_size), nn.Softplus()
            )

    def forward(self, z, m):

        if self.freeze_gen:
            mean = self.C.float() @ z.T + self.d

            # if we estimate the predictive variance
            if self.uncertainty:
                # set the variance to the ground truth variance
                var = self.gt_variance
                var = var * torch.ones_like(mean)
                return mean.T, var.T
            else:
                return mean.T, None
        else:
            # expand the dimensions
            x = self.dim_increase(z)
            # add in mask after dim increase even if mask not masked just pass zeros
            # in most cases the decoder mask is not passed both for naive and masked conditions
            if not self.pass_mask:
                m = torch.ones_like(x)
            m = self.mask_embdeding(m)

            # concatenate the mask and the input
            x = torch.cat((x, m), dim=1)
            x = self.relu_hidden_dim_expansion(x)
            # expand the dimensions
            x = self.relu_hidden_dim_reduction(x)
            #  linear layer to output
            mean = self.mean_linear(x)
            if self.uncertainty:
                var = self.var_linear(x)
                return mean, var
            else:
                return mean, None


class ImputeMissing(nn.Module):
    def __init__(self, input_size, init="norm", scale=0.1):

        super().__init__()
        # set up encoder modules
        self.input_size = input_size
        # ini
        # self.alpha = nn.Parameter(torch.ones(input_size, requires_grad=True))
        if init == "one":
            self.alpha = nn.Parameter(
                torch.ones(input_size, requires_grad=True).float()
            )
        elif init == "zero":
            self.alpha = nn.Parameter(
                torch.zeros(input_size, requires_grad=True).float()
            )
        elif init == "uni":
            self.alpha = nn.Parameter(
                torch.tensor(
                    scale * np.random.rand(input_size), requires_grad=True
                ).float()
            )
        else:
            self.alpha = nn.Parameter(
                torch.tensor(
                    np.random.normal(0, scale, size=(input_size)), requires_grad=True
                ).float()
            )

    def forward(self, x, m=None):
        """Imputes missing values with learned parameter alpha

        Args:
          x: input vector
          m: mask

        Returns:
          modified x with imputed alpha values instead of 0s for missing values
        """
        x = x + (1 - m) * self.alpha

        return x


# ----------------------------- building blocks -----------------------------


def CNN1D_block(
    n_inputs=50,
    n_filters=5,
    n_layers=1,
    kernel_size=5,
    nonlin=nn.ELU,
    batch_norm=True,
    droput=False,
):
    """Build Sequential CNN block

    Args:
        n_inputs: number of input dimensions to the Convolutional neural net
        n_filters: numbe of convolutional filters
        n_layers: number of layers
        kernel_size: kernel_size of conv kernel
        nonlin: non linearity applied to each layer default to ELU
        batch_norm (bool): batch normalisation
        droput (bool): Dropout applied to each layer

    Returns:
        nn.Sequential(*layers) block with CNN

    """
    layers = []
    for i in range(n_layers):
        layers.append(
            nn.Conv1d(
                n_inputs,
                n_filters,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_filters, track_running_stats=False))
        # if batch_norm: layers.append(PixelNorm())
        if droput:
            layers.append(nn.Dropout(droput))
        layers.append(nonlin())
        n_inputs = n_filters
    return nn.Sequential(*layers)


def RNN_block(n_inputs=32, n_outputs=50, n_layers=1, unit=nn.GRU, forw_backw=True):
    """Prepare RNN block and return Sequential module

    Args:
        n_inputs: number of input dimensions to the recurrent neural net
        n_outputs: number of output dimensions of the recurrent neural net corresponds to number of hidden units
        n_layers: number of layers of rnns
        unit: type of RNN unit defaulted to GRUs
        forw_backw (bool): foward backward or unidirectional default True

    Returns:
        nn.Sequential(*layers): block with RNNs

    """
    layers = []
    layers.append(
        unit(
            input_size=n_inputs,
            hidden_size=n_outputs,
            num_layers=n_layers,
            bidirectional=forw_backw,
        )
    )
    return nn.Sequential(*layers)


def rrn_block(
    input_size=32, n_outputs=50, n_layers=1, unit=nn.GRU, forw_backw=False, bias=True
):
    """Prepare an RNN block and return Sequential module

    Args:
        n_inputs: number of input dimensions to the recurrent neural net
        n_outputs: number of output dimensions of the recurrent neural net corresponds to number of hidden units
        n_layers: number of layers of rnns
        unit: type of RNN unit defaulted to GRUs
        forw_backw (bool): foward backward or unidirectional default True

    Returns:
        nn.Sequential(*layers): block with RNNs

    """
    rnn = unit(
        input_size=input_size,
        hidden_size=n_outputs,
        num_layers=n_layers,
        bidirectional=forw_backw,
        bias=bias,
    )

    return rnn


def linear_block(
    n_inputs=50, n_outputs=5, n_layers=1, nonlin=nn.ELU, batch_norm=True, droput=False
):
    """Build Sequential linear block built by using 1D Conv with a kernelsize of 1

    Args:
        n_inputs: number of input dimensions to the Convolutional neural net
        n_filters: numbe of convolutional filters
        n_layers: number of layers
        kernel_size: kernel_size of conv kernel
        nonlin: non linearity applied to each layer default to ELU
        batch_norm (bool): batch normalisation
        droput (bool): Dropout applied to each layer

    Returns:
        nn.Sequential(*layers) block with CNN

    """
    layers = []
    for i in range(n_layers):
        layers.append(nn.Conv1d(n_inputs, n_outputs, kernel_size=1))
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_outputs, track_running_stats=False))
        # if batch_norm: layers.append(PixelNorm())
        if droput:
            layers.append(nn.Dropout(droput))
        layers.append(nonlin())
        n_inputs = n_outputs
    return nn.Sequential(*layers)


def conv_block_1D(
    n_inputs=50,
    n_filters=5,
    n_layers=1,
    kernel_size=5,
    stride=1,
    groups=1,
    nonlin=nn.ELU,
    batch_norm=True,
    droput=False,
):
    """Build Sequential CNN block

    Args:
        n_inputs: number of input dimensions to the Convolutional neural net
        n_filters: numbe of convolutional filters
        n_layers: number of layers
        kernel_size: kernel_size of conv kernel
        groups: groups of the convolution (1:2D convolution, n_inputs: one filter per input dimension)
        nonlin: non linearity applied to each layer default to ELU
        batch_norm (bool): batch normalisation
        droput (bool): Dropout applied to each layer

    Returns:
        nn.Sequential(*layers) block with CNN

    """
    layers = []
    for i in range(n_layers):
        layers.append(
            nn.Conv1d(
                n_inputs,
                n_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_filters, track_running_stats=False))
        if droput:
            layers.append(nn.Dropout(droput))
        layers.append(nonlin())
        n_inputs = n_filters
    return nn.Sequential(*layers)


def deconv_block_1D(
    n_inputs=50,
    n_filters=5,
    n_layers=1,
    kernel_size=5,
    stride=1,
    groups=1,
    nonlin=nn.ELU,
    batch_norm=True,
    droput=False,
):
    """Build Sequential CNN block

    Args:
        n_inputs: number of input dimensions to the Convolutional neural net
        n_filters: numbe of convolutional filters
        n_layers: number of layers
        kernel_size: kernel_size of conv kernel
        groups: groups of the convolution (1:2D convolution, n_inputs: one filter per input dimension)
        nonlin: non linearity applied to each layer default to ELU
        batch_norm (bool): batch normalisation
        droput (bool): Dropout applied to each layer

    Returns:
        nn.Sequential(*layers) block with CNN

    """
    layers = []
    for i in range(n_layers):
        layers.append(
            nn.ConvTranspose1d(
                n_inputs,
                n_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_filters, track_running_stats=False))
        if droput:
            layers.append(nn.Dropout(droput))
        layers.append(nonlin())
        n_inputs = n_filters
    return nn.Sequential(*layers)


# ----------------------------- Moenky Autoencoder model -----------------------------
class AE_model(nn.Module):

    """Autoencoder model class"""

    def __init__(self, net_pars):

        """
        Initialize with a single dict that contains all the parameters.
        This is useful for easy prototyping.

        Parameters
        ----------
        net_pars: dict
            All parameters that should be changed / added to the default attributes

        Attributes
        ----------

        layer_pars: dict of lists
            Sets the architecture for the network. Each modules is parametrized by [n_layers, number hidden units, kernel size (for cnns)]
        nonlin:
            default nonlinearity for hidden layers
        rnn_unit:
            GRU or LSTM
        forw_backw: bool
            whether to use a bidirectional RNN
        n_latents : int
            number of VAE latents
        vae : bool
            whether to use a VAE or RAE
        lasso_mat : bool
            Whether to use a matrix / or vector to sparsify the latents


        inputs / outputs: list of dict
            Names of inputs / outputs
        n_inputs / n_outputs: list of ints
            Number of traces
        n_group_latents : int
            Total number of latents. Can be different than n_latents when a lasso matrix is used.

        self.out_inds : dict of np.index_exp
            Slices for the different network outputs
        self.lat_inds : dict of np.index_exp
            Slices for the different latent spaces

        self.group_masks : dict of torch.tensors
            Determines which outputs use which part of the latent space as input.
            They are assembled by their names by comparing the letters following the underscore.
            e.g. xa_m would 'see'  z_m and z_msyd but not z_s
        self.lasso_W : torch.tensor
            Vector / Matrix multiplied with latents to induce sparsity

        """
        super().__init__()

        default_attr = dict(
            layer_pars={
                "cnn_comb": [2, 200, 5],
                "cnn_rnn": [2, 200],
                "cnn_dec": [2, 100, 5],
            },
            nonlin=nn.ELU,
            rnn_unit=nn.GRU,
            forw_backw=True,
            lasso_mat=True,
            n_latents=100,
            vae=False,
        )

        default_attr.update(
            net_pars
        )  # Add attributes from net_pars of overwrite those in default_arr
        self.__dict__.update(
            (k, v) for k, v in default_attr.items()
        )  # Adds all attributes to the class

        self.n_inputs = sum([self.inp_dims[x] for x in self.inputs])
        self.n_outputs = sum([self.inp_dims[x] for x in self.outputs])
        self.n_group_latents = sum(self.group_latents.values())
        self.n_factors = self.layer_pars["cnn_dec"][1]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cnn1d_fac_fac = 1 + int(self.forw_backw)

        if not self.lasso_mat:
            self.n_latents = self.n_group_latents

        lat_filters = self.n_latents * (
            1 + int(self.vae)
        )  # Number of units on the VAE latent output, 2 * n_latents for mu and sigma

        self.out_inds = {}
        self.out_noise_inds = {}

        ind = 0
        end_ind = 0
        ind_obs = 0
        end_ind_obs = 0
        # get the indices for the outputs
        # if GNLL, then the indices are different for the xa and x_b
        print("Specify outputs ...... ")
        for k in self.outputs:
            end_ind = ind + self.inp_dims[k]
            print(k, ind, end_ind)
            self.out_inds[k] = np.index_exp[ind:end_ind][0]
            # set ind to end_ind in a safe way
            ind = end_ind

            # counter for observation noise variance for behavioral variables
            if self.obs_noise:
                if "xb" in k:
                    end_ind_obs = ind_obs + self.inp_dims[k]
                    self.out_noise_inds[k] = np.index_exp[ind_obs:end_ind_obs][0]
                    ind_obs = end_ind_obs
        # number of observation noise
        self.n_noise_outputs = end_ind_obs

        # set the latent indices
        # that one knows 0 to 10 corresp to neuro only and 10 to 20 etc
        self.lat_inds = {}
        ind = 0
        for k in self.group_latents:
            self.lat_inds[k] = np.index_exp[ind : ind + self.group_latents[k]][0]
            ind += self.group_latents[k]

        # set the group masks
        self.group_masks = {}
        # for each output, set the group mask
        for k in self.outputs:
            # mask of the corresponding latents
            if torch.cuda.is_available():
                self.group_masks[k] = torch.zeros(self.n_group_latents).type(
                    torch.cuda.LongTensor
                )
            else:
                self.group_masks[k] = torch.zeros(self.n_group_latents)
            for l in self.group_latents:
                # if the output name is contained in the latent name
                # e.g. x_m in z_msyd but not x_m in z_s
                if k.partition("_")[-1] in l.partition("_")[-1]:
                    self.group_masks[k][self.lat_inds[l]] = 1

        # keep the lasso matrix
        if self.lasso_mat:
            # a matrix of size n_group_latents X n_latents
            self.lasso_W = torch.tensor(
                np.zeros([self.n_group_latents, self.n_latents], dtype="float32"),
                requires_grad=True,
                device=self.device,
            )
            torch.nn.init.kaiming_uniform_(self.lasso_W, nonlinearity="linear")
        else:
            # simply an array of 0.1 that gets multiplied with the latents
            self.lasso_W = torch.tensor(
                0.1 * np.ones(self.n_group_latents, dtype="float32"),
                requires_grad=True,
                device=self.device,
            )

        # ENCODER

        self.session_inp = nn.ModuleList(
            [
                CNN1D_block(
                    n_inputs=self.n_inputs,
                    n_layers=1,
                    n_filters=self.n_inputs * 2,
                    kernel_size=1,
                    batch_norm=False,
                    nonlin=nn.Identity,
                ).to(self.device)
                for _ in range(self.n_sessions)
            ]
        )
        self.cnn1d_comb = CNN1D_block(
            n_inputs=self.n_inputs * 2,
            n_layers=self.layer_pars["cnn_comb"][0],
            n_filters=self.layer_pars["cnn_comb"][1],
            kernel_size=self.layer_pars["cnn_comb"][2],
            nonlin=self.nonlin,
        )

        self.rnn_enc = RNN_block(
            n_inputs=self.layer_pars["cnn_comb"][1],
            n_outputs=self.layer_pars["cnn_rnn_enc"][1],
            n_layers=self.layer_pars["cnn_rnn_enc"][0],
            unit=self.rnn_unit,
            forw_backw=self.forw_backw,
        )

        if self.latent_rec:
            self.rnn_cell = torch.nn.GRUCell(
                input_size=self.layer_pars["cnn_rnn_enc"][1] * cnn1d_fac_fac
                + self.n_latents,
                hidden_size=self.layer_pars["cnn_rnn_enc"][1],
            )
            cnn1d_fac_fac = 1

        self.cnn1d_fac = CNN1D_block(
            n_inputs=self.layer_pars["cnn_rnn_enc"][1] * cnn1d_fac_fac,
            n_filters=lat_filters,
            n_layers=1,
            nonlin=nn.Identity,
            kernel_size=1,
            batch_norm=False,
        )

        if self.larger_encoder:

            self.session_inp = nn.ModuleList(
                [
                    CNN1D_block(
                        n_inputs=self.n_inputs,
                        n_layers=1,
                        n_filters=self.n_latents * 3,
                        kernel_size=1,
                        batch_norm=False,
                        nonlin=nn.Identity,
                    ).to(self.device)
                    for _ in range(self.n_sessions)
                ]
            )

            self.non_linear_dim_red_enc = CNN1D_block(
                n_inputs=self.n_latents * 3,
                n_layers=1,
                n_filters=self.n_latents * 2,
                kernel_size=1,
                nonlin=self.nonlin,
            )
            self.cnn1d_comb = CNN1D_block(
                n_inputs=self.n_latents * 2,
                n_layers=self.layer_pars["cnn_comb"][0],  # layers 1
                n_filters=self.layer_pars["cnn_comb"][1],  # output size
                kernel_size=self.layer_pars["cnn_comb"][2],  # kernel size
                nonlin=self.nonlin,
            )

            self.rnn_enc = RNN_block(
                n_inputs=self.layer_pars["cnn_comb"][1],
                n_outputs=self.layer_pars["cnn_rnn_enc"][1],
                n_layers=self.layer_pars["cnn_rnn_enc"][0],
                unit=self.rnn_unit,
                forw_backw=self.forw_backw,
            )

            if self.latent_rec:
                self.rnn_cell = torch.nn.GRUCell(
                    input_size=self.layer_pars["cnn_rnn_enc"][1] * cnn1d_fac_fac
                    + self.n_latents,
                    hidden_size=self.layer_pars["cnn_rnn_enc"][1],
                )
                cnn1d_fac_fac = 1

            self.cnn1d_fac = CNN1D_block(
                n_inputs=self.layer_pars["cnn_rnn_enc"][1] * cnn1d_fac_fac,
                n_filters=lat_filters,
                n_layers=1,
                nonlin=nn.Identity,
                kernel_size=1,
                batch_norm=False,
            )

        # DECODER

        if self.dec_rnn:
            self.rnn_dec = RNN_block(
                n_inputs=self.n_group_latents,
                n_outputs=self.layer_pars["cnn_rnn_dec"][1],
                n_layers=self.layer_pars["cnn_rnn_dec"][0],
                unit=self.rnn_unit,
                forw_backw=False,
            )
            self.cnn1d_dec = CNN1D_block(
                n_inputs=self.layer_pars["cnn_rnn_dec"][1],
                n_filters=self.layer_pars["cnn_dec"][1],
                n_layers=self.layer_pars["cnn_dec"][0],
                kernel_size=self.layer_pars["cnn_dec"][2],
                nonlin=self.nonlin,
            )
        else:
            self.cnn1d_dec = CNN1D_block(
                n_inputs=self.n_group_latents,
                n_filters=self.layer_pars["cnn_dec"][1],
                n_layers=self.layer_pars["cnn_dec"][0],
                kernel_size=self.layer_pars["cnn_dec"][2],
                nonlin=self.nonlin,
            )
        self.session_outp = nn.ModuleList(
            [
                CNN1D_block(
                    n_inputs=self.layer_pars["cnn_dec"][1],
                    n_filters=self.n_outputs,
                    n_layers=1,
                    kernel_size=1,
                    nonlin=nn.Identity,
                    batch_norm=False,
                ).to(self.device)
                for _ in range(self.n_sessions)
            ]
        )

        if self.obs_noise:
            self.session_noise_outp = nn.ModuleList(
                [
                    CNN1D_block(
                        n_inputs=self.layer_pars["cnn_dec"][1],
                        n_filters=self.n_noise_outputs,
                        n_layers=1,
                        kernel_size=1,
                        nonlin=nn.Softplus,
                        batch_norm=False,
                    ).to(self.device)
                    for _ in range(self.n_sessions)
                ]
            )

        for n in range(self.n_sessions):
            self.session_inp[n][0].weight = copy.deepcopy(
                self.session_inp[0][0].weight
            )  # Trial specific input / output matrices are initialized with identical values
            self.session_outp[n][0].weight = copy.deepcopy(
                self.session_outp[0][0].weight
            )
            if self.obs_noise:
                self.session_noise_outp[n][0].weight = copy.deepcopy(
                    self.session_noise_outp[0][0].weight
                )

    def make_sparse(self, z):
        """Multiplies the VAE latents with lasso_W"""
        if self.lasso_mat:  # matrix multiplication
            z = torch.matmul(self.lasso_W, z)
        else:  # here this is just a vector scaling
            z = z * self.lasso_W[None, :, None]
        return z

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, batch, train_mask=None, inp_mask={}):
        """Encoding function

        Parameters
        ----------
        batch : dict
            Batch as provided by Primate_Reach.get_batch or Primate_Reach.get_session
        train_mask : binary tensor
            Zeros out traces of the input. Used during training to enable 'cross training' from one set of traces to the remaining ones.
        inp_mask : dict of 'str'
            Can be used to zero out the given inputs.

        Returns
        _______
            ret_dict : dict of tensors
                Returns the means, sigmas and samples from the VAE latents (before applying any sparsification)

        """
        ret_dict = {}
        # check if index not in inp_mask and then multiply with batch[k]
        # basically if index is in inp_mask, then multiply with 0, else multiply with batch[k]
        h = torch.cat([int(k not in inp_mask) * batch[k] for k in self.inputs], 1)

        if train_mask is not None:
            h = h * train_mask[None, :, None]

        h = self.session_inp[batch["i"]](h)
        bs, ch, T = h.shape
        # 1D conv -> 200 dim
        if self.larger_encoder:
            h = self.non_linear_dim_red_enc(h)
        h = self.cnn1d_comb(h)
        # forward backward GRU -> 200 dim * 2 (forward and backward)
        h = self.rnn_enc(h.permute(2, 0, 1))[0]

        if self.latent_rec:
            z_mu = torch.zeros(T, bs, self.n_latents).to(self.device)
            z_lsig = torch.zeros(T, bs, self.n_latents).to(self.device)
            z = torch.zeros(T, bs, self.n_latents).to(self.device)
            for t in range(1, T):
                ht = self.rnn_cell(torch.cat([h[t], z[t - 1]], -1))
                # go from high dim z sstate to n_latents X 2 (mu and sigma)
                zt = self.cnn1d_fac(ht[:, :, None])[:, :, 0]
                z_mu[t], z_lsig[t] = zt[:, : self.n_latents], zt[:, self.n_latents :]
                # here we only take one sample - could be changed to multiple samples
                z[t] = torch.distributions.Normal(
                    z_mu[t], torch.exp(z_lsig[t])
                ).rsample()

            z_mu = z_mu.permute(1, 2, 0)
            z_lsig = z_lsig.permute(1, 2, 0)
            z = z.permute(1, 2, 0)

        else:  # go directly from the group encoder rnn output to the latent space
            z = self.cnn1d_fac(h.permute(1, 2, 0))
            if self.vae:
                z_mu, z_lsig = z[:, : self.n_latents], z[:, self.n_latents :]
                z = torch.distributions.Normal(z_mu, torch.exp(z_lsig)).rsample()

        ret_dict["z"] = ret_dict["z_mu"] = z  # for vae z_mu gets overwritten
        if self.vae:
            ret_dict["z_mu"] = z_mu
            ret_dict["z_lsig"] = z_lsig

        return ret_dict

    def decode(self, z, i, scale_output=True):
        """Decoding function

        Parameters
        ----------
        z : tensor
            VAE sample provided by the encoding function
        i : int
             Trial index
        scale_output : bool
            Whether to scale the outputs to reverse the scaling used during training.
            When True produces outputs at the original scale.

        Returns
        _______
            ret_dict : dict of tensors
                Returns the network outputs

        """
        ret_dict = {}
        z = self.make_sparse(z)

        for k in self.outputs:

            z_sub = z * self.group_masks[k][None, :, None]
            if self.dec_rnn:
                h = self.rnn_dec(z_sub.permute(2, 0, 1))[0].permute(1, 2, 0)
                h = self.cnn1d_dec(h)
            else:
                h = self.cnn1d_dec(z_sub)
            # smooth factors
            ret_dict[k + "_fac"] = h
            x_rec = self.session_outp[i](h)
            if self.obs_noise and "xb" in k:
                x_rec_noise = self.session_noise_outp[i](h)
                if self.mean_obs_noise:
                    # get the mean across the last dimension and expand it to the same shape as the prediction
                    # do not consider the first warmup time steps in the averaging
                    x_rec_noise = torch.mean(
                        x_rec_noise[:, :, self.warmup :], keepdim=True, dim=-1
                    ).expand_as(
                        x_rec_noise
                    )  # expansion necessary to actually keep dims

            if self.outputs[k] == "poisson":
                # pylint: disable=not-callable
                ret_dict[k] = F.softplus(x_rec[:, self.out_inds[k]])
            else:
                ret_dict[k] = x_rec[:, self.out_inds[k]]
                if self.obs_noise:
                    ret_dict[k + "_noise"] = x_rec_noise[:, self.out_noise_inds[k]]

            if scale_output:
                if self.ifdimwise_scaling:
                    if k in self.scaling:  # if k in self.dimwise_scaling:
                        for ii in range(ret_dict[k].shape[1]):
                            ret_dict[k][:, ii] = (
                                self.dimwise_scaling[k + str(ii)][i][1]
                                + ret_dict[k][:, ii]
                                / self.dimwise_scaling[k + str(ii)][i][0]
                            )  # sigma * z + mu
                            if self.obs_noise and "xb" in k:
                                ret_dict[k + "_noise"][:, ii] = (
                                    ret_dict[k + "_noise"][:, ii]
                                    / self.dimwise_scaling[k + str(ii)][i][0]
                                )  # mutliply std(z) with sigma
                else:
                    if k in self.scaling:
                        # scale back to original scale mu+ z /sima
                        ret_dict[k] = (
                            self.scaling[k][i][1] + ret_dict[k] / self.scaling[k][i][0]
                        )
                        if self.obs_noise and "xb" in k:  # DONE GNLL
                            ret_dict[k + "_noise"] = (
                                ret_dict[k + "_noise"] / self.scaling[k][i][0]
                            )  # mutliply std(z) with sigma

        return ret_dict

    def forward(self, batch, train_mask=None, inp_mask={}, scale_output=True):
        """Runs the encoder and decoder and returns a dictionary containing latents and outputs"""
        ret = {}
        ret.update(self.encode(batch, train_mask, inp_mask))
        ret.update(self.decode(ret["z"], batch["i"], scale_output))

        return ret


def apply_mask(model, batch, train_mask=None, inp_mask={}):
    """apply the mask equivalent to the encode function

    Parameters
    ----------
    batch : dict
        Batch as provided by Primate_Reach.get_batch or Primate_Reach.get_session
    train_mask : binary tensor
        Zeros out traces of the input. Used during training to enable 'cross training' from one set of traces to the remaining ones.
    inp_mask : dict of 'str'
        Can be used to zero out the given inputs.

    Returns
    _______
        ret_dict : dict of tensors
            Returns the means, sigmas and samples from the VAE latents (before applying any sparsification)

    """
    # check if index not in inp_mask and then multiply with batch[k]
    # basically if index is in inp_mask, then multiply with 0, else multiply with batch[k]
    h = torch.cat([int(k not in inp_mask) * batch[k] for k in model.net.inputs], 1)

    if train_mask is not None:
        h = h * train_mask[None, :, None]

    return h

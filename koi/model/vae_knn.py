import torch
import math
from torch import nn

from koi.config.base_config import BaseConfig
from koi.model.base_model import GenerativeModel
from koi.util.utils import dst


class VAEKNN(GenerativeModel):
    def __init__(self, config: BaseConfig, *args, **kwargs):
        GenerativeModel.__init__(self, x_dim=config.x_dim, config=config, **kwargs)
        self.kwargs = kwargs
        self.latent_size = config.latent_size
        self.x_dim = config.x_dim

        self.encoder = self.Encoder(
            config.encoder_layer_sizes, config.latent_size)
        self.decoder = self.Decoder(
            config.decoder_layer_sizes, config.latent_size, self.x_dim,
            last_activation_function=config.last_activation_function)

    # based on https://github.com/timbmg/VAE-CVAE-MNIST/ implementation

    class Encoder(nn.Module):
        def __init__(self, layer_sizes, latent_size):
            super().__init__()

            self.MLP = nn.Sequential()

            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.MLP.add_module(
                    name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

            self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
            self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

        def forward(self, x, c=None):
            x = self.MLP(x)

            means = self.linear_means(x)
            log_vars = self.linear_log_var(x)

            return means, log_vars

    class Decoder(nn.Module):
        def __init__(self, layer_sizes, latent_size, x_dim, last_activation_function=None):
            super().__init__()

            self.MLP = nn.Sequential()
            input_size = latent_size

            for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
                self.MLP.add_module(
                    name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                if i + 1 < len(layer_sizes):
                    self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                else:
                    if last_activation_function == 'sigmoid':
                        self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
                    elif last_activation_function is None:
                        # no need for final activation function for toy examples in R^2
                        pass

            # todo subclass
            self.linear_means = nn.Linear(layer_sizes[-1], x_dim)
            # self.linear_log_vars = nn.Linear(layer_sizes[-1], x_dim)

        def forward(self, z):
            x = self.MLP(z)
            # means = self.linear_means(x)
            # log_vars = self.linear_log_vars(x)

            return x  # means, log_vars

    # from nn.Module class
    def forward(self, x, sample=True):
        if x.dim() > 2:
            x = x.view(-1, self.x_dim)

        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var, sample=sample)
        # recon_x, recon_log_var = self.decoder(z)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var, sample=True):
        if not sample:
            return mu

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z):
        # recon_x, recon_log_var = self.decoder(z)
        recon_x = self.decoder(z)
        return recon_x  # , recon_log_var

    # from GenerativeModel class
    def load_model(self, filepath, *args, **kwargs):
        pass

    def is_cuda(self):
        pass

    def save_model(self, filepath, *args, **kwargs):
        pass

    def loss_function(self, recon_x, x, mean, log_var, kl_weight=torch.tensor(1.), y=torch.tensor(1.),
                      knn_distance=torch.tensor(1.)):

        recon_error = dst(recon_x, x, dst_function=self.config.dst_function)

        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)


        gamma = 0.3
        NLL = torch.mean(1 - torch.exp(-gamma * knn_distance), dim=1) * recon_error * y
        NLL = NLL.sum()

        loss = (NLL + KLD * kl_weight)

        return NLL / y.sum(), KLD / y.sum(), loss / y.sum()

    def not_working_loss_function(self, recon_x, x, mean, log_var, kl_weight=torch.tensor(1.)):

        recon_error = dst(recon_x, x, dst_function=self.config.dst_function)

        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)  # , dim=0)

        # KL_weight, anneal_parameter_beta =   # todo: rimuovere secondo elemento tornato

        # recon_std = torch.exp(0.5 * recon_log_var)

        # gisuto fare .sum() così? o in dim=1? TODO
        # NLL = 0.5 * (dst / var).sum(dim=1) + 0.5 * torch.log(var.sum(dim=1)) + 0.5 * var.size()[1] * torch.log(2 * torch.tensor(math.pi))
        # last term is constant, but i'd like to add it anyway to have the most precise formulation.
        # TODO: ensure this is correct

        N = torch.tensor(x.size()[1])
        # dim=1 is across latent dimension
        # OK NLL = 0.5 * dst.sum(dim=1) + 0.5 * N * torch.log(2 * torch.tensor(math.pi)) + 0.5 * torch.log(N)
        # seems almost ok
        # NLL = 0.5 * (dst / recon_std).sum(dim=1) \
        #      + 0.5 * N * torch.log(2 * torch.tensor(math.pi)) \
        #      + 0.5 * recon_std.sum(dim=1)

        # recon_var = recon_log_var.exp()
        # va solo con latent dim =1
        # NLL = 0.5 * ((x-recon_x)*(x-recon_x) / recon_var).sum(dim=1) \
        #     + 0.5 * N * torch.log(2 * torch.tensor(math.pi)) \
        #     + 0.5 * torch.log(recon_var.sum(dim=1))

        # proposta paper cern, non funziona con latent_dim >1, da valori negativi ma comunque modello corretto
        # per latent_dim = 1
        # NLL = torch.sum((x - recon_x) * (x - recon_x) / (2 * recon_var) + torch.log(
        #     torch.sqrt(2 * torch.tensor(math.pi)) * torch.sqrt(recon_var)), dim=1)
        # NLL = NLL.sum()

        # NLL = 0.5 * dst.sum(dim=1) + 0.5 * N * torch.log(2 * torch.tensor(math.pi)) + 0.5 * torch.log(N)
        NLL = recon_error.sum(dim=1)
        # NLL = NLL.sum()

        # assert NLL > 0
        # TODO NLL = 0.5 * (dst / var).sum(dim=1) + 0.5 * N * torch.log(2 * torch.tensor(math.pi)) + 0.5 * torch.log(
        #             var.sum(dim=1)) perché nll negativa?

        loss = (NLL + KLD * kl_weight)  # * KL_weight)

        return NLL.mean(), KLD.mean(), loss.mean()

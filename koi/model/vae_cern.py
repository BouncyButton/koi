import math

import torch
from torch import nn

from koi.config.base_config import BaseConfig
from koi.model.base_model import GenerativeModel
from koi.model.vae import VAE
from koi.util.utils import dst


class VAECern(VAE):
    # loss based on https://arxiv.org/abs/2010.05531
    # based on https://github.com/timbmg/VAE-CVAE-MNIST/ implementation

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

            self.linear_means = nn.Linear(layer_sizes[-1], x_dim)
            self.linear_log_vars = nn.Linear(layer_sizes[-1], x_dim)

        def forward(self, z):
            x = self.MLP(z)
            means = self.linear_means(x)
            log_vars = self.linear_log_vars(x)

            return means, log_vars

    # from nn.Module class
    def forward(self, x, sample=True):
        if x.dim() > 2:
            x = x.view(-1, self.x_dim)

        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var, sample=sample)
        recon_x, recon_log_var = self.decoder(z)

        return recon_x, recon_log_var, means, log_var, z

    def inference(self, z, scale=1):
        recon_x, recon_log_var = self.decoder(z)
        sampled_x = torch.normal(mean=recon_x, std=torch.exp(0.5 * recon_log_var) * scale)

        return sampled_x

    def inference_mean_logvar(self, z):
        recon_x, recon_log_var = self.decoder(z)
        return recon_x, recon_log_var

    def loss_function_from_paper(self, recon_x, x, mean, log_var, kl_weight=torch.tensor(1.), recon_log_var=None):
        recon_error = dst(recon_x, x, dst_function=self.config.dst_function)

        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)

        recon_var = torch.exp(recon_log_var)
        recon_std = torch.sqrt(recon_var)

        NLL = (0.5 * recon_error / recon_var).sum(dim=1) \
              + torch.log(torch.sqrt(2 * torch.tensor(math.pi)) * recon_std.sum(dim=1)) \
            # + N * torch.log(2 * torch.tensor(math.pi))

        NLL = NLL.sum()

        loss = (NLL + KLD * kl_weight)

        return NLL / x.size(0), KLD / x.size(0), loss / x.size(0)

    def loss_function(self, recon_x, x, mean, log_var, kl_weight=torch.tensor(1.), recon_log_var=None):

        recon_error = dst(recon_x, x, dst_function=self.config.dst_function)

        # why mean AND /x.size(0) on the return?
        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)

        recon_var = torch.exp(recon_log_var)
        N = torch.tensor(x.size()[1])

        # giusto fare .sum() così? o in dim=1? TODO
        # questo da risultati giusti, ma teoricamente mi pare sbagliato
        NLL = (recon_error / recon_var).sum(dim=1) \
              + torch.log(recon_var.sum(dim=1)) \
              + N * torch.log(2 * torch.tensor(math.pi))

        # questo teoricamente mi sembra giusto, ma mi da risultati sbagliati
        # NLL = (recon_error / recon_var + torch.log(recon_var)).sum(dim=1) \
        #      + N * torch.log(2 * torch.tensor(math.pi))
        # TODO ???

        # NLL *= 0.5
        # last term is constant, but i'd like to add it anyway to have the most precise formulation.
        # TODO: ensure this is correct

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
        NLL = NLL.sum()

        # NLL = 0.5 * dst.sum(dim=1) + 0.5 * N * torch.log(2 * torch.tensor(math.pi)) + 0.5 * torch.log(N)

        # assert NLL > 0
        # TODO NLL = 0.5 * (dst / var).sum(dim=1) + 0.5 * N * torch.log(2 * torch.tensor(math.pi)) + 0.5 * torch.log(
        #             var.sum(dim=1)) perché nll negativa?

        loss = (NLL + KLD * kl_weight)  # * KL_weight)

        return NLL / x.size(0), KLD / x.size(0), loss / x.size(0)

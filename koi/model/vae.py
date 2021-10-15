import torch
from dotmap import DotMap
from torch import nn

from koi.model.base_model import GenerativeModel


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, num_labels):
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
    def __init__(self, layer_sizes, latent_size, num_labels, last_activation_function=None):
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
                else:
                    # no need for final activation function for toy examples in R^2
                    pass

    def forward(self, z):
        x = self.MLP(z)
        return x


# should I use multiple inheritance here? imHo yes.

class VAE(GenerativeModel, nn.Module):
    # based on https://github.com/timbmg/VAE-CVAE-MNIST/ implementation

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, x_dim, num_labels=0,
                 last_activation_function=None, **kwargs):
        super(GenerativeModel, self).__init__()
        super(nn.Module, self).__init__()
        self.kwargs = kwargs
        self.latent_size = latent_size
        self.x_dim = x_dim

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, num_labels, last_activation_function=last_activation_function)

    # from nn.Module class
    def forward(self, x, sample=True):
        if x.dim() > 2:
            x = x.view(-1, self.x_dim)

        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var, sample=sample)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var, sample=True):
        if not sample:
            return mu

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z):
        recon_x = self.decoder(z)
        return recon_x

    # from GenerativeModel class
    def load_model(self, filepath, *args, **kwargs):
        pass

    def is_cuda(self):
        pass

    def save_model(self, filepath, *args, **kwargs):
        pass


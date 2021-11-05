import math

import torch

from koi.model.vae import VAE
from koi.util.utils import dst


class VAECorrectLoss(VAE):
    def loss_function(self, recon_x, x, mean, log_var, kl_weight=torch.tensor(1.)):
        recon_error = dst(recon_x, x, dst_function=self.config.dst_function)

        KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)
        N = torch.tensor(x.size(1))

        # by deriving log normal(x;recon_x,I), we obtain the following.
        # after recon error the value can be ignored (no gradient is provided)
        NLL = 0.5 * recon_error.sum(dim=1) + 0.5 * N * torch.log(2 * torch.tensor(math.pi))
        NLL = NLL.sum()

        loss = (NLL + KLD * kl_weight)

        # average over batch
        return NLL / x.size(0), KLD / x.size(0), loss / x.size(0)

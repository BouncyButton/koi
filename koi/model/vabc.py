import torch

from koi.model.vae import VAE
from koi.util.utils import dst


class VABC(VAE):
    def loss_function(self, recon_x, x, mean, log_var,
                      kl_weight=torch.tensor(1.),
                      gamma_weight=torch.tensor(1.),
                      y=torch.tensor(1.)):

        # TODO: perché l2-norm funziona meglio di mse?
        recon_error = dst(recon_x, x, dst_function=self.config.dst_function)
        # recon_error = recon_error.sum(dim=0)

        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # TODO gamma annealing seems to not work yet
        anti_recon_error = torch.log(1 - torch.exp(-recon_error * gamma_weight) + 1e-16)  # * args.gamma_prime

        assert len(anti_recon_error.shape) == 1  # voglio che venga usato nel modo corretto, cioè
        # devo avere n_batch elementi. NON voglio array bidimensionali
        # (es: y ha avuto strani reshape)

        NLL = (y * recon_error - (1 - y) * anti_recon_error)
        NLL = NLL.sum()

        loss = (NLL + KLD * kl_weight)

        return NLL / x.size(0), KLD / x.size(0), loss / x.size(0), -((1 - y) * anti_recon_error).sum() / x.size(0), \
               (y * recon_error).sum() / x.size(0)

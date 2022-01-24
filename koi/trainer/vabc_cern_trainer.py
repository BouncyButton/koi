from itertools import cycle

import torch

from koi.trainer.vabc_trainer import VABCTrainer
from koi.trainer.vae_trainer import VAETrainer


class VABCCernTrainer(VABCTrainer):

    def _run_epoch(self, step, epoch, dataset, **kwargs):
        if dataset is None:
            return

        train = dataset.is_train()
        dlp = dataset.dlp
        dln = dataset.dln
        device = self.device
        x_dim = self.config.x_dim
        nll, kld, loss, anti_rec, rec, kl_weight, gamma_weight = (None,) * 7

        # warning! need to extend to all cases
        assert len(dlp) >= len(dln)

        for iteration, ((x_p, label_p), (x_n, label_n)) in enumerate(zip(dlp, cycle(dln))):
            step += 1
            x_p, label_p = x_p.to(device), label_p.to(device)
            x_p = x_p.reshape(-1, x_dim)
            label_p = label_p.flatten()
            mask_p = torch.ones(label_p.size(0)).to(device).float()  # (label_pu != LABEL).float()

            x_n, label_n = x_n.to(device), label_n.to(device)
            x_n = x_n.reshape(-1, x_dim)
            label_n = label_n.flatten()
            mask_n = torch.zeros(label_n.size(0)).to(device).float()  # (label_n != LABEL).float()

            x = torch.cat((x_p, x_n))
            mask = torch.cat((mask_p, mask_n))

            # shuffling for negative/positive samples
            perm = torch.randperm(x.size(0))
            x = x[perm]
            mask = mask[perm]

            recon_x, recon_log_var, mean, log_var, z = self.model(x, sample=True)
            kl_weight, beta = self.get_current_kl(step)
            gamma_weight = self.get_current_gamma(step)

            nll, kld, loss, rec, anti_rec = self.model.loss_function(
                recon_x, x, mean, log_var, y=mask, kl_weight=kl_weight, gamma_weight=gamma_weight, recon_log_var=recon_log_var)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # todo tensorboard
            print("[{6}] nll: {0:.4f}, kld: {1:.4f}, anti_rec: {2:.4f} rec: {3:.4f} klw: {4:.4f} gammaw: {5: .4f}"
                  .format(nll.item(), kld.item(),
                       anti_rec.item(),
                           rec.item(),
                           kl_weight, gamma_weight, dataset.split))
        self._log(dataset.split, epoch, nll=nll.item(), kld=kld.item(), kl_weight=kl_weight, gamma_weight=gamma_weight,
                  loss=loss.item(),
                  rec=rec.item(), anti_rec=anti_rec.item())

        return step

from itertools import cycle
from typing import Type, Optional

import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

from koi.config.vabc_config import VABCConfig
from koi.dataset.base_dataset import KoiDataset
from koi.model.base_model import GenerativeModel
from koi.model.vabc import VABC
from koi.trainer.vae_trainer import VAETrainer, anneal_parameters, anneal_function


class VABCTrainer(VAETrainer):
    def __init__(self, train: Optional[KoiDataset], val: Optional[KoiDataset],
                 test: Optional[KoiDataset],
                 config: VABCConfig, model_type=VABC,
                 *args,
                 **kwargs):
        super().__init__(train, val, test, config, model_type)
        self.k_gamma, self.x0_gamma = None, None

    def get_current_gamma(self, step):
        gamma_weight = self.config.gamma

        if self.config.abc_annealing:
            # mi interessa che gamma non vada mai sotto il valore che ho scelto.
            # faccio quindi un annealing FINO a warm_up_epochs, poi assume il valore normale.
            # anneal_parameter_gamma = max(anneal_function('logistic', step, self.k_gamma, self.x0_gamma,
            #                                              len(self.train.dlp) * self.config.warm_up_epochs),
            #                              self.config.gamma1)
            # gamma_weight *= anneal_parameter_gamma / self.config.gamma1
            N = self.config.start_gamma_multiplier
            G = self.config.gamma
            v = anneal_function('logistic', step, self.k_gamma, self.x0_gamma,
                                len(self.train.dlp) * self.config.warm_up_epochs)
            v = 1-v  # horizontal flip
            # starts from NG * 1, ends in G, as wanted
            gamma_weight = ((N-1)*G) * v + G

        return gamma_weight

    def _run_training(self, **kwargs):
        step = 0
        epochs = self.config.epochs
        self.k, self.x0 = anneal_parameters(self.train.dlp, epochs=self.config.warm_up_epochs)
        self.k_gamma, self.x0_gamma = anneal_parameters(self.train.dlp, epochs=self.config.warm_up_epochs)

        # kfold = KFold(n_splits=self.config.k_folds, shuffle=True)

        epoch = 0
        for epoch in tqdm(range(epochs)):
            step = self._run_epoch(step, epoch, self.train)
            self._run_epoch(step, epoch, self.val)
        self._run_epoch(step, epoch, self.test)

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

            recon_x, mean, log_var, z = self.model(x, sample=True)
            kl_weight, beta = self.get_current_kl(step)
            gamma_weight = self.get_current_gamma(step)

            nll, kld, loss, anti_rec, rec = self.model.loss_function(
                recon_x, x, mean, log_var, y=mask, kl_weight=kl_weight, gamma_weight=gamma_weight)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # todo tensorboard
            # print("[{6}] nll: {0:.4f}, kld: {1:.4f}, anti_rec: {2:.4f} rec: {3:.4f} klw: {4:.4f} gammaw: {5: .4f}"
            #       .format(nll.item(), kld.item(),
            #               anti_rec.item(),
            #               rec.item(),
            #               kl_weight, gamma_weight, dataset.split))
        self._log(dataset.split, epoch, nll=nll.item(), kld=kld.item(), kl_weight=kl_weight, gamma_weight=gamma_weight,
                  loss=loss.item(),
                  rec=rec.item(), anti_rec=anti_rec.item())

        return step

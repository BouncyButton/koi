import math
from typing import Union, Optional, Type

import numpy as np
import torch
from tqdm import tqdm

from koi.config.base_config import BaseConfig
from koi.dataset.base_dataset import KoiDataset
from koi.model.base_model import GenerativeModel
from koi.model.vae import VAE
from koi.trainer.base_trainer import Trainer
from koi.util.utils import dst


def anneal_function(fn_name, step, k, x0, N=None):
    if fn_name == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif fn_name == 'linear':
        return min(1, step / x0)
    elif fn_name == 'cyclic':
        step = step % N
        return float(1 / (1 + np.exp(-k * (step - x0))))


def anneal_parameters(train_loader, beta0=0.001, beta1=0.999, epochs=4):
    # todo explain how it is derived.
    batches_per_epoch = len(train_loader)
    N = batches_per_epoch * epochs
    k = -np.log(beta0 * (beta1 - 1) / ((beta0 - 1) * beta1)) / N
    x0 = np.log(1 / beta0 - 1) / k
    return k, x0


class VAETrainer(Trainer):
    # i dont like that config is replicated in both model and config.
    def __init__(self, train: Optional[KoiDataset], val: Optional[KoiDataset],
                 test: Optional[KoiDataset],
                 config: BaseConfig,
                 model_type=VAE,
                 *args,
                 **kwargs):
        super().__init__(model_type, train, val, test, config)
        self.k, self.x0 = None, None

        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=config.learning_rate)  # TODO weight_decay=1e-2)
        elif config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=config.learning_rate)  # TODO weight_decay=1e-2)
        else:
            raise NotImplementedError('optim=adam/sgd')

    def get_current_kl(self, step):
        KL_weight = self.config.beta

        anneal_parameter_beta = 1
        if self.config.kl_annealing:
            anneal_parameter_beta = anneal_function('logistic', step, self.k, self.x0)
            KL_weight *= anneal_parameter_beta

        return KL_weight, anneal_parameter_beta

    def _run_training(self, **kwargs):
        step = 0
        epochs = self.config.epochs
        self.k, self.x0 = anneal_parameters(self.train.dlp, epochs=self.config.warm_up_epochs)

        epoch = 0
        for epoch in tqdm(range(epochs)):
            step = self._run_epoch(step, epoch, self.train)
            self._run_epoch(step, epoch, self.val)

        self._run_epoch(step, epoch, self.test)

    def _run_epoch(self, step, epoch, dataset, **kwargs):
        if dataset is None:
            return

        device = self.device
        x_dim = self.config.x_dim
        nll, kld, kl_weight, loss = None, None, None, None
        dl = dataset.dlp  # we *do not* provide negative data to the vae!

        for iteration, (x, label) in enumerate(dl):
            step += 1
            x, label = x.to(device), label.to(device)
            x = x.reshape(-1, x_dim)
            recon_x, mean, log_var, z = self.model(x, sample=True)
            kl_weight, beta = self.get_current_kl(step)
            nll, kld, loss = self.model.loss_function(recon_x, x, mean, log_var, kl_weight=kl_weight)

            if dataset.is_train():
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self._log(dataset.split, epoch, nll=nll.item(), kld=kld.item(), kl_weight=kl_weight, loss=loss.item())
        # print("[{3}] nll: {0:.4f}, kld: {1:.4f}, klw: {2:.4f}".
        # format(nll.item(), kld.item(), kl_weight, dataset.split))
        return step

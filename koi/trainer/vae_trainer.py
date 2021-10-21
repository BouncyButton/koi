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


def kl_anneal_function(anneal_function, step, k, x0, N=None):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)
    elif anneal_function == 'cyclic':
        step = step % N
        return float(1 / (1 + np.exp(-k * (step - x0))))


def find_kl_anneal_parameters(train_loader, beta0=0.001, beta1=0.999, epochs=4):
    # todo explain how it is derived.
    batches_per_epoch = len(train_loader)
    N = batches_per_epoch * epochs
    k = -np.log(beta0 * (beta1 - 1) / ((beta0 - 1) * beta1)) / N
    x0 = np.log(1 / beta0 - 1) / k
    return k, x0


class VAETrainer(Trainer):
    # i dont like that config is replicated in both model and config.
    def __init__(self, model_type: Type[GenerativeModel], train: Optional[KoiDataset], test: Optional[KoiDataset],
                 config: BaseConfig,
                 *args,
                 **kwargs):
        super().__init__(model_type, train, test, config)
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
            anneal_parameter_beta = kl_anneal_function('logistic', step, self.k, self.x0)
            KL_weight *= anneal_parameter_beta

        return KL_weight, anneal_parameter_beta

    def run_training(self, **kwargs):
        step = 0
        epochs = self.config.epochs
        self.k, self.x0 = find_kl_anneal_parameters(self.train.dlp, epochs=self.config.warm_up_epochs)

        for epoch in tqdm(range(epochs)):
            step = self._training_epoch(step, epoch, self.train.dlp)

    def _training_epoch(self, step, epoch, dl):
        device = self.device
        x_dim = self.config.x_dim
        nll, kld, kl_weight = None, None, None

        for iteration, (x, label) in enumerate(dl):
            step += 1
            x, label = x.to(device), label.to(device)
            x = x.reshape(-1, x_dim)
            recon_x, mean, log_var, z = self.model(x, sample=True)
            kl_weight, beta = self.get_current_kl(step)
            nll, kld, loss = self.model.loss_function(recon_x, x, mean, log_var, kl_weight=kl_weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # todo tensorboard
        print("nll: {0:.4f}, kld: {1:.4f}, klw: {2:.4f}".format(nll.item(), kld.item(), kl_weight))
        return step

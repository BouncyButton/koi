from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt

from koi.config.base_config import BaseConfig
from koi.dataset.base_dataset import KoiDataset
from sklearn import cluster, datasets


class MoonsDataset(KoiDataset):
    def __init__(self, N=10000, moon_noise=0.2, label_noise=0.0, noisy_label=0, config=BaseConfig(), **kwargs):
        X, y = datasets.make_moons(n_samples=N, noise=moon_noise, random_state=config.seed)
        # if label_noise > 0:

        super().__init__(X, y, torch.Tensor, config=config, **kwargs)

        self.X_original = deepcopy(X)
        self.y_original = deepcopy(y)
        y[y == noisy_label] = 1 - (np.random.rand(y[y == noisy_label].size) < (1-label_noise))

    def get_positive_data(self):
        return self.X[self.targets == 1]

    def get_negative_data(self):
        return self.X[self.targets == 0]

    def view2d(self, show=True):
        if self.X.ndim != 2:
            raise NotImplementedError()

        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.targets, alpha=0.2)
        plt.legend(*scatter.legend_elements())

        if show:
            plt.show()

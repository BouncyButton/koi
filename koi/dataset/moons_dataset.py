import torch

from koi.dataset.base_dataset import KoiDataset
from sklearn import cluster, datasets


class MoonsDataset(KoiDataset):
    def __init__(self, N=10000, noise=0.2):
        X, y = datasets.make_moons(n_samples=N, noise=noise)
        super().__init__(X, y, torch.Tensor)

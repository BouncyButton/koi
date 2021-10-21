import torch
from torch.utils.data import Dataset, DataLoader, sampler

from koi.config.base_config import BaseConfig


class KoiDataset(Dataset):
    def __init__(self, X, y, transform=torch.Tensor, create_data_loaders=True, config=BaseConfig()):
        self.X = X
        self.targets = y
        self.transform = transform
        self.batch_size = config.batch_size
        self.dl = self.dlp = self.dln = None
        if create_data_loaders:
            self.create_data_loaders()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        point = self.transform(self.X[idx]).reshape(1, -1)
        label = self.transform([self.targets[idx]]).int()

        return point, label

    def get_indexes(self, class_names):
        indexes = []
        for i in range(len(self.targets)):
            if self.targets[i] in class_names:
                indexes.append(i)
        return indexes

    def filter_dataset(self, labels):
        '''
        Filters MNIST dataset using label.
        '''
        idx = self.get_indexes(labels)
        dl = DataLoader(self, batch_size=self.batch_size, sampler=sampler.SubsetRandomSampler(idx))

        return dl

    def create_data_loaders(self, positive=[1], negative=[0]):
        self.dl = DataLoader(
            dataset=self, batch_size=self.batch_size, shuffle=True)
        self.dln = self.filter_dataset(negative)
        self.dlp = self.filter_dataset(positive)

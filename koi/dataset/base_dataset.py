import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, sampler

from koi.config.base_config import BaseConfig


class KoiDataset(Dataset):
    def __init__(self, X, y, transform=torch.Tensor, create_data_loaders=True, config=BaseConfig(), split='train'):
        self.X = X
        self.targets = y
        self.transform = transform
        self.batch_size = config.batch_size
        self.dl = self.dlp = self.dln = None
        self.split = split
        self.config = config
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

    def is_train(self):
        return self.split == 'train'

    def get_correct_split(self, dataset):
        tr_size = int(len(dataset) * (1 - self.config.validation_size))
        val_size = len(dataset) - tr_size

        train_set_tr, val_set_tr = torch.utils.data.random_split(dataset,
                                                                 lengths=[tr_size, val_size],
                                                                 generator=torch.Generator().manual_seed(
                                                                     self.config.seed))
        if self.split == 'train':
            return train_set_tr
        elif self.split == 'val':
            return val_set_tr
        else:
            raise NotImplementedError

    # def create_data_loaders_xv(self, k_folds, positive=[1], negative=[0]):
    #     if not self.is_train():
    #         raise ValueError("Can't cross validate test data")
    #
    #     self.train_folds = []
    #     self.val_folds = []
    #     kcv = KFold(n_splits=k_folds, shuffle=True)
    #     split = kcv.split(self)
    #
    #     for fold, (train_ids, val_ids) in enumerate(split):
    #         self.train_folds.append()

    # dataset = ConcatDataset([self.train, self.test])

    # kfold = KFold(n_splits=self.config.k_folds, shuffle=True)
    # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    # val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    #
    # for fold, (train_ids, val_ids) in enumerate(kfold.split(self.train)):

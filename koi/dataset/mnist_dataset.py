import random
from copy import deepcopy

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from koi.config.base_config import BaseConfig
from koi.dataset.base_dataset import KoiDataset


class MNISTDataset(KoiDataset):
    def __init__(self, label=8, p=0.2, split='train', config=BaseConfig(), create_data_loaders=True, **kwargs):

        # prendo train anche per la validazione (ma mi occupo io di prendere quello che mi serve)
        dataset = MNIST(
            root='data', train=split != 'test', transform=transforms.ToTensor(),
            download=True)

        # filling later, need config, split, etc.
        super().__init__([], [], create_data_loaders=False)

        dataset = self.get_correct_split(dataset)  # retrieve the validation split or the training split.

        X = []
        y = []
        for image, label in dataset:
            X.append(image)
            y.append(label)

        self.X_original = deepcopy(X)
        self.y_original = deepcopy(y)

        # todo very slow! should improve, i.e. using subset.
        for i in range(len(dataset)):
            if dataset[i][1] == label and random.random() < p:
                self.X.append(dataset[i][0])
                self.targets.append(10)
            else:
                self.X.append(dataset[i][0])
                self.targets.append(dataset[i][1])

        if create_data_loaders:
            self.create_data_loaders(positive=[x for x in range(10) if x != label],
                                     negative=[label])

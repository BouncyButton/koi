import numpy as np
import torch
from typing import Optional, Type

from ..config.base_config import BaseConfig
from ..dataset.base_dataset import KoiDataset
from ..model.base_model import GenerativeModel
import random


class Trainer:
    """
    Abstract base class for training any generative model implemented by koi.
    """

    def __init__(self, model: Type[GenerativeModel], train: KoiDataset, test: Optional[KoiDataset],
                 config: BaseConfig):
        self.config = config

        if torch.cuda.is_available() and not self.config.torch_device == 'cpu':
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        # better seed even if cuda is active
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        self.train = train
        self.test = test

        print(config.latent_size)
        self.model = model(config).to(self.device)
        print(self.model.config.latent_size)
        print("Current device: ", self.device)

    def run_training(self, **kwargs):
        r"""Training procedure.
        This method is meant to execute all the training phase. Once the method ends, the
        model should be ready to be used to perform approximate density estimation or sampling.
        """
        raise NotImplementedError()

    def reset_config(self):
        pass

    def init_data(self):
        self.train.create_data_loaders()

import torch
import typing

from dotmap import DotMap

from ..config.base_config import BaseConfig
from ..dataset.base_dataset import KoiDataset
from ..model.base_model import GenerativeModel


class Trainer:
    """
    Abstract base class for training any generative model implemented by koi.
    """
    def __init__(self, model: GenerativeModel, train: KoiDataset, test: KoiDataset, config: BaseConfig):
        self.model = model
        self.train = train
        self.test = test
        self.config = config
        if model.is_cuda():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    def train(self, train_data, **kwargs):
        r"""Training procedure.
        This method is meant to execute all the training phase. Once the method ends, the
        model should be ready to be used to perform approximate density estimation or sampling.
        """
        raise NotImplementedError()

    def reset_config(self):
        pass

    def init_data(self):
        self.train.create_data_loaders()


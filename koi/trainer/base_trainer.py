import os

import numpy as np
import torch
from typing import Optional, Type

from ..config.base_config import BaseConfig
from ..dataset.base_dataset import KoiDataset
from ..model.base_model import GenerativeModel
import random
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Abstract base class for training any generative model implemented by koi.
    For now, we implement validation as holdout set (i.e. a single fold in cross-validation)
    """

    def __init__(self, model: Type[GenerativeModel], train: KoiDataset, val: Optional[KoiDataset],
                 test: Optional[KoiDataset],
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

        # better specify here, so you don't need to give it as parameter.
        train.split = 'train'
        val.split = 'val'
        test.split = 'test'

        self.train = train
        self.val = val
        self.test = test

        self.model = model(config).to(self.device)
        print("Current device: ", self.device)
        print(config.__dict__)
        print(model.__dict__)
        self.writer = dict()
        import time
        t = str(int(time.time()))
        self.writer['train'] = SummaryWriter(os.path.join(self.config.logs_folder, "train/{0}".format(t)))
        self.writer['val'] = SummaryWriter(os.path.join(self.config.logs_folder, "val/{0}".format(t)))
        self.writer['test'] = SummaryWriter(os.path.join(self.config.logs_folder, "test/{0}".format(t)))

    def _log(self, tag, epoch, **kwargs):
        for k, v in kwargs.items():
            self.writer[tag].add_scalar(k, v, epoch)

    def run_training(self, **kwargs):
        r"""Training procedure.
        This method is meant to execute all the training phase. Once the method ends, the
        model should be ready to be used to perform approximate density estimation or sampling.
        """

        self._run_training(**kwargs)
        for k, v in self.writer.items():
            v.flush()
            v.close()

    def _run_training(self, **kwargs):
        r"""Training procedure.
        This method is meant to execute all the training phase. Once the method ends, the
        model should be ready to be used to perform approximate density estimation or sampling.
        """
        raise NotImplementedError()

    def reset_config(self):
        pass

    def init_data(self):
        self.train.create_data_loaders()

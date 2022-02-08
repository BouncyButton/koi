from koi.config.fail_fast_config import FailFastConfig
from koi.config.vabc_mnist_config import MNISTVABCConfig
from koi.config.vae_config import VAEConfig
from koi.config.vae_mnist_config import MNISTVAEConfig
from koi.dataset.mnist_dataset import MNISTDataset
from koi.dataset.moons_dataset import MoonsDataset
from koi.metrics.metrics import generative_negative_error
from koi.model.vabc import VABC
from koi.model.vae import VAE
from koi.config.base_config import BaseConfig
from koi.model.vae_cern import VAECern
from koi.model.vae_correct_loss import VAECorrectLoss
from koi.trainer.base_trainer import Trainer
from koi.trainer.vabc_trainer import VABCTrainer
from koi.trainer.vae_cern_trainer import VAECernTrainer
from koi.trainer.vae_trainer import VAETrainer
from koi.visualizer.mnist import MNISTVisualizer
from koi.visualizer.toy_example import ToyExampleVisualizer


class VAEOnMNISTDataset:
    def __init__(self, test=False):
        config = MNISTVABCConfig()
        # TODO make toy VAE config and make BaseConfig as abstract as possible
        train = MNISTDataset(config=config, split='train')
        val = MNISTDataset(config=config, split='val')
        test = MNISTDataset(config=config, split='test')
        self.trainer = VABCTrainer(model_type=VABC, config=config, train=train, val=val, test=test)

    def run(self):
        self.trainer.run_training()
        # generative_negative_error(self.trainer)  # todo doesnt seem to work properly, fix with cnnt
        v = MNISTVisualizer(self.trainer)
        v.show_samples()


if __name__ == '__main__':
    print('dev')
    ex = VAEOnMNISTDataset()
    ex.run()

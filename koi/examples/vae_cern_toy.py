from koi.config.cern_config import CernConfig
from koi.config.fail_fast_config import FailFastConfig
from koi.config.vae_config import VAEConfig
from koi.dataset.moons_dataset import MoonsDataset
from koi.metrics.metrics import generative_negative_error
from koi.model.vae import VAE
from koi.config.base_config import BaseConfig
from koi.model.vae_cern import VAECern
from koi.model.vae_correct_loss import VAECorrectLoss
from koi.trainer.base_trainer import Trainer
from koi.trainer.vae_cern_trainer import VAECernTrainer
from koi.trainer.vae_trainer import VAETrainer
from koi.visualizer.toy_example import ToyExampleVisualizer


class VAECernOnToyDataset:
    def __init__(self, test=False):
        config = FailFastConfig() if test else CernConfig()
        # TODO make toy VAE config and make BaseConfig as abstract as possible
        train = MoonsDataset(config=config, split='train')
        val = MoonsDataset(N=10000, config=config, split='val')
        test = MoonsDataset(N=1000, config=config, split='test')
        self.trainer = VAECernTrainer(model_type=VAECern, config=config, train=train, val=val, test=test)

    def run(self):
        self.trainer.run_training()
        generative_negative_error(self.trainer)
        v = ToyExampleVisualizer(self.trainer)
        v.show_2d_samples()


if __name__ == '__main__':
    print('dev')
    ex = VAECernOnToyDataset()
    ex.run()

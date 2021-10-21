from koi.config.fail_fast_config import FailFastConfig
from koi.dataset.moons_dataset import MoonsDataset
from koi.model.vae import VAE
from koi.config.base_config import BaseConfig
from koi.model.vae_cern import VAECern
from koi.model.vae_correct_loss import VAECorrectLoss
from koi.trainer.base_trainer import Trainer
from koi.trainer.vae_cern_trainer import VAECernTrainer
from koi.trainer.vae_trainer import VAETrainer
from koi.visualizer.toy_example import ToyExampleVisualizer


class VAEOnToyDataset:
    def __init__(self, test=False):
        config = FailFastConfig() if test else BaseConfig()
        # TODO make toy VAE config and make BaseConfig as abstract as possible
        train = MoonsDataset(config=config)
        test = MoonsDataset(N=1000, config=config)
        self.trainer = VAECernTrainer(model=VAECern, config=config, train=train, test=test)

    def run(self):
        self.trainer.run_training()
        v = ToyExampleVisualizer(self.trainer)
        v.show_2d_samples()


if __name__ == '__main__':
    print('dev')
    ex = VAEOnToyDataset()
    ex.run()

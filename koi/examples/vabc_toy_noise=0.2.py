from koi.config.fail_fast_config import FailFastConfig
from koi.config.vabc_config import VABCConfig
from koi.dataset.moons_dataset import MoonsDataset
from koi.metrics.metrics import generative_negative_error
from koi.model.vae import VAE
from koi.config.base_config import BaseConfig
from koi.model.vabc import VABC
from koi.model.vae_cern import VAECern
from koi.model.vae_correct_loss import VAECorrectLoss
from koi.trainer.base_trainer import Trainer
from koi.trainer.vabc_trainer import VABCTrainer
from koi.trainer.vae_cern_trainer import VAECernTrainer
from koi.trainer.vae_trainer import VAETrainer
from koi.visualizer.toy_example import ToyExampleVisualizer


class VABCOnToyDataset:
    def __init__(self, test=False):
        config = VABCConfig()

        train = MoonsDataset(config=config, split='train', label_noise=0.2)
        val = MoonsDataset(N=1000, config=config, split='val', label_noise=0.2)
        test = MoonsDataset(N=1000, config=config, split='test', label_noise=0.2)
        self.trainer = VABCTrainer(model_type=VABC, config=config, train=train, val=val, test=test)

    def run(self):
        self.trainer.run_training()
        v = ToyExampleVisualizer(self.trainer)
        generative_negative_error(self.trainer)
        v.show_2d_samples()


if __name__ == '__main__':
    print('dev')
    ex = VABCOnToyDataset()
    ex.run()

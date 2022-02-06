from koi.config.cern_config import CernConfig
from koi.config.fail_fast_config import FailFastConfig
from koi.config.vabc_cern_config import VABCCernConfig
from koi.config.vae_config import VAEConfig
from koi.dataset.moons_dataset import MoonsDataset
from koi.metrics.metrics import generative_negative_error
from koi.model.vabc_cern import VABCCern
from koi.model.vae import VAE
from koi.config.base_config import BaseConfig
from koi.model.vae_cern import VAECern
from koi.model.vae_correct_loss import VAECorrectLoss
from koi.trainer.base_trainer import Trainer
from koi.trainer.vabc_cern_trainer import VABCCernTrainer
from koi.trainer.vabc_trainer import VABCTrainer
from koi.trainer.vae_trainer import VAETrainer
from koi.visualizer.toy_example import ToyExampleVisualizer


class VABCCernOnToyDataset:
    def __init__(self, test=False):
        self.config = VABCCernConfig()

    def run(self):
        for i in range(2,10):
            self.config.seed = i

            # TODO make toy VAE config and make BaseConfig as abstract as possible
            train = MoonsDataset(config=self.config, split='train')
            val = MoonsDataset(N=10000, config=self.config, split='val')
            test = MoonsDataset(N=1000, config=self.config, split='test')
            self.trainer = VABCCernTrainer(model_type=VABCCern, config=self.config, train=train, val=val, test=test)

            self.trainer.run_training()
            generative_negative_error(self.trainer, stack=False)
            v = ToyExampleVisualizer(self.trainer)
            v.show_2d_samples()
        # v.gradient_field(positive=True)
        # v.gradient_field(positive=False)


if __name__ == '__main__':
    print('dev')
    ex = VABCCernOnToyDataset()
    ex.run()

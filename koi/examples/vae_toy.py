from koi.config.fail_fast_config import FailFastConfig
from koi.config.vae_config import VAEConfig
from koi.dataset.moons_dataset import MoonsDataset
from koi.metrics.metrics import generative_negative_error
from koi.model.vae import VAE
from koi.config.base_config import BaseConfig
from koi.model.vae_cern import VAECern
from koi.model.vae_correct_loss import VAECorrectLoss
from koi.trainer.base_trainer import Trainer

from koi.trainer.vae_trainer import VAETrainer
from koi.visualizer.toy_example import ToyExampleVisualizer


class VAEOnToyDataset:
    def __init__(self, test=False):
        self.config = FailFastConfig() if test else VAEConfig()
        self.test = test

    def run(self):

        errs = []
        for i in range(3,13):

            self.config.seed = i
            train = MoonsDataset(config=self.config, split='train')
            val = MoonsDataset(N=10000, config=self.config, split='val')
            test = MoonsDataset(N=1000, config=self.config, split='test')
            self.trainer = VAETrainer(model_type=VAE, config=self.config, train=train, val=val, test=test)
            self.trainer.run_training()
            v = ToyExampleVisualizer(self.trainer)
            v.show_2d_samples()
            # v.kde_estimation()
            #v.gradient_field(positive=True)
            #v.gradient_field(positive=False)

            errs.append(generative_negative_error(self.trainer, N1=100, stack=False))
        print("="*80)
        print(sum(errs)/len(errs))



if __name__ == '__main__':
    print('dev')
    ex = VAEOnToyDataset()
    ex.run()

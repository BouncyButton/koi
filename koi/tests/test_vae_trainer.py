from koi.config.cern_config import CernConfig
from koi.config.fail_fast_config import FailFastConfig
from koi.dataset.base_dataset import KoiDataset
from koi.dataset.moons_dataset import MoonsDataset
from koi.model.vae import VAE
from koi.model.vae_cern import VAECern
from koi.model.vae_correct_loss import VAECorrectLoss
from koi.trainer.vae_cern_trainer import VAECernTrainer
from koi.trainer.vae_trainer import VAETrainer
from koi.visualizer.toy_example import ToyExampleVisualizer


def test_vae_trainer():
    config = FailFastConfig()
    train = MoonsDataset(N=10, config=config)
    test = MoonsDataset(N=10, config=config)
    val = MoonsDataset(N=10, config=config)
    trainer = VAETrainer(train=train, val=val, test=test, config=config)
    trainer.run_training()
    v = ToyExampleVisualizer(trainer)
    v.show_2d_samples(test=True)


def test_vae_correct_loss_trainer():
    config = FailFastConfig()
    train = MoonsDataset(N=10, config=config)
    test = MoonsDataset(N=10, config=config)
    val = MoonsDataset(N=10, config=config)
    trainer = VAETrainer(train=train, val=val, test=test, config=config)
    trainer.run_training()
    v = ToyExampleVisualizer(trainer)
    v.show_2d_samples(test=True)


def test_vae_cern_trainer():
    config = CernConfig()
    config.epochs = 1
    train = MoonsDataset(N=10, config=config)
    test = MoonsDataset(N=10, config=config)
    val = MoonsDataset(N=10, config=config)
    trainer = VAETrainer(train=train, val=val, test=test, config=config)
    trainer.run_training()
    v = ToyExampleVisualizer(trainer)
    v.show_2d_samples(test=True)

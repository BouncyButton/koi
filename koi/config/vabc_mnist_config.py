from koi.config.vabc_config import VABCConfig
from koi.config.vae_config import VAEConfig


class MNISTVABCConfig(VABCConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x_dim = 784
        self.latent_size = 2
        self.encoder_layer_sizes = [784, 300, 100]  # todo remove first
        self.decoder_layer_sizes = [100, 300, 784]
        self.torch_device = 'cuda'
        self.dst_function = 'l2-norm'
        self.epochs = 30
        self.kl_annealing = True
        #self.abc_annealing = True
        self.warm_up_epochs = 10
        self.batch_size = 80
        self.gamma = 0.05
        self.beta = 1/5
        self.last_activation_function = 'sigmoid'
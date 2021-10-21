from dotmap import DotMap

from koi.config.base_config import BaseConfig


class FailFastConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kl_annealing = True
        self.seed = 0
        self.epochs = 4
        self.warm_up_epochs = 2
        self.batch_size = 80
        self.learning_rate = 0.001
        self.encoder_layer_sizes = [2, 4, 8]
        self.decoder_layer_sizes = [8, 4, 2]
        self.latent_size = 2
        self.beta = 1 # / 20
        self.gamma0 = 0.8
        self.gamma1 = 0.2
        self.gamma_prime = 1
        self.x_dim = 2
        self.last_activation_function = None #'sigmoid'
        self.dst_function = 'mse'

        # overwrite with any kwargs provided the current configuration
        # from https://stackoverflow.com/questions/8187082/
        # TODO can we avoid repeating this in any subclass?
        for k, v in kwargs.items():
            setattr(self, k, v)

from koi.config.base_config import BaseConfig


class VAEKNNConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma_prime = 1
        self.start_gamma_multiplier = 4
        self.encoder_layer_sizes = [2, 10, 10]  # todo remove first
        self.decoder_layer_sizes = [10, 10, 2]
        self.latent_size = 2
        self.kl_annealing = True
        self.abc_annealing = False
        self.epochs = 150
        self.warm_up_epochs = 10
        self.beta = 1/2
        self.gamma = 9
        self.batch_size = 80
        self.dst_function = 'l2-norm'
        # overwrite with any kwargs provided the current configuration
        # from https://stackoverflow.com/questions/8187082/
        for k, v in kwargs.items():
            setattr(self, k, v)

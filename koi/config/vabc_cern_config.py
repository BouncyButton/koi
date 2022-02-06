from koi.config.base_config import BaseConfig


class VABCCernConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layer_sizes = [2, 20, 40]  # todo remove first
        self.decoder_layer_sizes = [40, 20, 2]
        self.latent_size = 1
        self.kl_annealing = True
        self.abc_annealing = False
        self.epochs = 50
        self.warm_up_epochs = 10
        self.gamma = 0.1
        self.dst_function = 'mse'
        self.beta = 0.5

        # overwrite with any kwargs provided the current configuration
        # from https://stackoverflow.com/questions/8187082/
        for k, v in kwargs.items():
            setattr(self, k, v)

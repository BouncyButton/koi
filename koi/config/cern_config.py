from koi.config.base_config import BaseConfig


class CernConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layer_sizes = [2, 25, 50]  # todo remove first
        self.decoder_layer_sizes = [50, 25, 2]
        self.latent_size = 1
        self.kl_annealing = True
        self.epochs = 150
        self.warm_up_epochs = 20

        # overwrite with any kwargs provided the current configuration
        # from https://stackoverflow.com/questions/8187082/
        for k, v in kwargs.items():
            setattr(self, k, v)

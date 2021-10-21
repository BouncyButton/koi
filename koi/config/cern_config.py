from dotmap import DotMap

from koi.config.base_config import BaseConfig


class CernConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layer_sizes = [2, 10]  # todo remove first
        self.decoder_layer_sizes = [10, 2]
        self.latent_size = 1

        # overwrite with any kwargs provided the current configuration
        # from https://stackoverflow.com/questions/8187082/
        for k, v in kwargs.items():
            setattr(self, k, v)

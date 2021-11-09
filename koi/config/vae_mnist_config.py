from koi.config.vae_config import VAEConfig


class MNISTVAEConfig(VAEConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x_dim = 784
        self.encoder_layer_sizes = [784, 300, 100]  # todo remove first
        self.decoder_layer_sizes = [100, 300, 784]
        self.torch_device = 'cuda'
        self.dst_function = 'l2-norm'

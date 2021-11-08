from koi.config.base_config import BaseConfig


class VAEConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = 'adam'
        self.torch_device = 'cpu'
        self.kl_annealing = False
        self.seed = 0
        self.epochs = 50
        self.warm_up_epochs = self.epochs // 2
        self.batch_size = 128
        self.learning_rate = 0.001
        self.encoder_layer_sizes = [2, 10]  # todo remove first
        self.decoder_layer_sizes = [10, 2]
        self.latent_size = 2
        self.print_every = 1000
        self.beta = 1 # 1 / 20
        self.gamma0 = 0.8
        self.gamma1 = 0.2
        self.gamma_prime = 1
        self.fig_root = 'figures'
        self.conditional = False
        self.x_dim = 2
        self.last_activation_function = None  # 'sigmoid'
        self.dst_function = 'mse'

        # overwrite with any kwargs provided the current configuration
        # from https://stackoverflow.com/questions/8187082/
        for k, v in kwargs.items():
            setattr(self, k, v)

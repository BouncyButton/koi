from dotmap import DotMap


class BaseConfig(DotMap):
    def __init__(self):
        self.seed = 0
        self.epochs = 150
        self.warm_up_epochs = 50
            'batch_size': 80,
            'learning_rate': 0.001,
            'encoder_layer_sizes': [2, 10, 10],
            'decoder_layer_sizes': [10, 10, 2],
            'latent_size': 2,
            'print_every': 1000,
            'beta': 1 / 20,
            'gamma0': 0.8,
            'gamma1': 0.2,
            'gamma_prime': 1,
            'fig_root': 'figures',
            'conditional': False
        })



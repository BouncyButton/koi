from koi.model.vae import VAE


class VAEOnToyDataset:
    def __init__(self):
        self.model = VAE(x_dim=x_dim,
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        last_activation_function=last_activation_function,
        num_labels=0).to(device)
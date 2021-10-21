import torch
from matplotlib import pyplot as plt

from koi.model.vae_cern import VAECern
from koi.trainer.base_trainer import Trainer


class ToyExampleVisualizer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.dataset = trainer.train

    def show_2d_samples(self, test=False, N=500):
        cfg = self.model.config
        device = self.trainer.device
        X, y = self.dataset.X, self.dataset.targets

        z = torch.randn([N, cfg.latent_size]).to(device)
        fig, ax = plt.subplots()

        show_var = isinstance(self.model, VAECern)

        if show_var:
            recon_mu, recon_log_var = self.model.inference(z)
            sampled_x = torch.normal(mean=recon_mu, std=torch.exp(0.5 * recon_log_var))
            recon_mu = recon_mu.cpu().detach().numpy()
            # show mu
            plt.scatter(recon_mu[:, 0], recon_mu[:, 1], c='red', alpha=0.8, marker='+', zorder=3)

        else:
            # we are sampling using the mean parameter and discarding variance (that is infact not modeled)
            sampled_x = self.model.inference(z)

        sampled_x = sampled_x.cpu().detach().numpy()
        # show dataset
        # TODO i'd like to find a printing-friendly color combination, i need external feedback
        colors = ['purple', 'white']
        for i in range(2):
            plt.scatter(X[:, 0][y == i], X[:, 1][y == i], c=colors[i], edgecolors='k', alpha=0.2)

        # show samples
        plt.scatter(sampled_x[:, 0], sampled_x[:, 1], c='green', alpha=0.8, marker='x')

        # compose custom legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='0',
                                      markerfacecolor='purple', markersize=10),
                           plt.Line2D([0], [0], marker='o', color='k', label='1',
                                      markerfacecolor='yellow', markersize=10),
                           plt.Line2D([], [], marker='x', color='green', label='samples',
                                      markerfacecolor='green', markersize=8, linestyle='None')]
        ax.legend(handles=legend_elements)
        if not test:
            plt.show()

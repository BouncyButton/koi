import numpy as np
import torch
from matplotlib import pyplot as plt

from koi.model.vabc_cern import VABCCern
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


        show_var = isinstance(self.model, VAECern) or isinstance(self.model, VABCCern)

        fig, ax = plt.subplots(figsize=(4.5, 3))

        if show_var:
            recon_mu, recon_log_var = self.model.inference_mean_logvar(z)
            sampled_x = torch.normal(mean=recon_mu, std=torch.exp(0.5 * recon_log_var))
            recon_mu = recon_mu.cpu().detach().numpy()
            # show mu
            ax.scatter(recon_mu[:, 0], recon_mu[:, 1], c='red', alpha=0.8, marker='+', zorder=3)

        else:
            # we are sampling using the mean parameter and discarding variance (that is infact not modeled)
            # sampled_x = self.model.inference(z)

            recon_mu = self.model.inference(z)
            sampled_x = torch.normal(mean=recon_mu, std=0.5*torch.tensor(self.trainer.config.beta))
            recon_mu = recon_mu.cpu().detach().numpy()
            # show mu
            ax.scatter(recon_mu[:, 0], recon_mu[:, 1], c='red', alpha=0.8, marker='+', zorder=3)


        sampled_x = sampled_x.cpu().detach().numpy()
        # show dataset
        # TODO i'd like to find a printing-friendly color combination, i need external feedback
        colors = ['purple', 'white']
        for i in range(2):
            ax.scatter(X[:, 0][y == i], X[:, 1][y == i], c=colors[i], edgecolors='k', alpha=0.2)

        # show samples
        ax.scatter(sampled_x[:, 0], sampled_x[:, 1], c='green', alpha=0.8, marker='x')

        # compose custom legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='0',
                                      markerfacecolor='purple', markersize=10),
                           plt.Line2D([0], [0], marker='o', color='k', label='1',
                                      markerfacecolor='yellow', markersize=10),
                           plt.Line2D([], [], marker='x', color='green', label='samples',
                                      markerfacecolor='green', markersize=8, linestyle='None')]
        ax.legend(handles=legend_elements)
        if not test:
            plt.show(block=True)

        fig.savefig("test1200t.pdf", format='pdf', dpi=1200, bbox_inches='tight', transparent=True)

    def kde_estimation(self):
        cfg = self.model.config
        device = self.trainer.device
        X, y = self.dataset.X, self.dataset.targets
        model = self.model

        z = torch.randn([10000, cfg.latent_size]).to(device)
        sampled_x = model.inference(z).cpu().detach().numpy()

        # plt.legend(*scatter.legend_elements())

        x = sampled_x[:, 0]
        y = sampled_x[:, 1]

        # Define the borders
        deltaX = (max(x) - min(x)) / 10
        deltaY = (max(y) - min(y)) / 10
        xmin = -2  # min(X_dataset[:,0])# - deltaX
        xmax = +3  # max(X_dataset[:,0])# + deltaX
        ymin = -1.25  # min(X_dataset[:,0])# - deltaY
        ymax = 1.75  # max(X_dataset[:,0])# + deltaY

        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        import scipy.stats as st

        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        # ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.title('2D Gaussian Kernel density estimation for sampled $x$. (N=10000)')
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.07)
        # plt.scatter(sampled_x[:,0], sampled_x[:,1], c='green', alpha=0.2, marker='x')
        plt.show()

        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('PDF')
        ax.set_title('Surface plot of Gaussian 2D KDE for sampled $x$. (N=10000)')
        fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
        ax.view_init(60, 35)
        plt.show()

        z = torch.randn([100000, cfg.latent_size]).to(device)
        sampled_x = model.inference(z).cpu().detach().numpy()
        my_cmap = plt.cm.get_cmap("jet").copy()
        my_cmap.set_under('w', 1)
        fig, ax = plt.subplots()
        plt.title('2D histogram for sampled $x$ (N=100000)')
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.1)
        h = plt.hist2d(x, y, bins=80, cmap=my_cmap, vmin=1, alpha=0.7)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        plt.colorbar(h[3])
        plt.show()

    def gradient_field(self, gradient_min_x=-2, gradient_max_x=2.5, gradient_min_y=-1, gradient_max_y=1.5, gradient_step=0.2, positive=True, title='moons', **kwargs):
        X = self.dataset.X
        y = self.dataset.targets
        import itertools
        l = list(itertools.product(np.arange(gradient_min_x,gradient_max_x,gradient_step), np.arange(gradient_min_y,gradient_max_y,gradient_step)))
        grid_numpy = np.array(l)

        grid = torch.Tensor(grid_numpy).to(self.trainer.device)

        # https://discuss.pytorch.org/t/derivative-of-model-outputs-w-r-t-input-features/95814/2
        grid.requires_grad = True
        recon_x, mean, log_var, z = self.model(grid, sample=False)

        label = torch.ones(grid.size(0)) if positive else torch.zeros(grid.size(0))

        nll, kld, loss, anti_rec, rec = self.trainer.model.loss_function(
                        recon_x, grid, mean, log_var, y=label.to(self.trainer.device))

        loss.backward(retain_graph = True)
        # print(grid.grad)
        gradients = grid.grad.detach().cpu().numpy()
        if positive:
            gradients *= -1
        #
        z = torch.randn([1000, self.trainer.config.latent_size]).to(self.trainer.device)
        sampled_x = self.model.inference(z).cpu().detach().numpy()

        fig,ax = plt.subplots()
        # plt.figure(figsize=(8, 10))

        plt.title('Negative loss gradients given {0}, with samples.'.format('y=1 (positive)' if positive else 'y=0 (negative)'))
        # recon_x = recon_x.cpu().detach().numpy()

        colors = ['purple', 'white']
        for i in range(2):
            plt.scatter(X[:, 0][y == i], X[:, 1][y == i], c=colors[i], edgecolors='k', alpha=0.2)

        # show samples
        plt.scatter(sampled_x[:, 0], sampled_x[:, 1], c='green', alpha=0.8, marker='x')

        plt.scatter(sampled_x[:,0], sampled_x[:,1], c='green', alpha=0.6, marker='x')
        legend_elements = [plt.Line2D([], [], marker='o', color='w', label='0',
                                markerfacecolor='purple', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='1',
                                markerfacecolor='yellow', markersize=10),
        plt.Line2D([], [], marker='x', color='green', label='samples',
                                markerfacecolor='green', markersize=8, linestyle = 'None'),
        plt.Line2D([], [], marker=r'$\rightarrow$', color='black', markersize=12, markerfacecolor='black', linestyle='None', label='gradient $-\partial \mathcal{L} / \partial x$')]

        ax.legend(handles=legend_elements)

        plt.quiver(grid_numpy[:,0],grid_numpy[:,1],gradients[:,0],gradients[:,1],alpha=0.9)
        plt.axis('equal')
        plt.show(block=True)


import numpy as np
import torch
from matplotlib import pyplot as plt


class MNISTVisualizer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.dataset = trainer.train

    def show_samples(self, test=False, size=3, N=20):
        # samples from latent
        step = 1
        plt.figure(figsize=(12, 12))
        for i in np.arange(-size, +size, size / N * 2):
            for j in np.arange(-size, +size, size / N * 2):
                plt.subplot(N, N, step)
                sampled_x = self.model.inference(torch.Tensor([i, j]).to(self.trainer.device))
                plt.imshow(sampled_x.reshape(28, 28).cpu().data.numpy())
                plt.axis('off')
                step += 1
        plt.show()

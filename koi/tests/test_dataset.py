import os
import pytest


def test_moons_dataset():
    from koi.dataset.moons_dataset import MoonsDataset
    md = MoonsDataset()
    md.view2d(show=False)


def test_noise_dataset():
    from koi.dataset.moons_dataset import MoonsDataset
    md = MoonsDataset(label_noise=0.2)
    md.view2d(show=False)


if __name__ == '__main__':
    test_noise_dataset()

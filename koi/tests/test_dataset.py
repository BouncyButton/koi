import os
import pytest


def test_moons_dataset():
    from koi.dataset.moons_dataset import MoonsDataset
    md = MoonsDataset()
    md.view2d(show=False)

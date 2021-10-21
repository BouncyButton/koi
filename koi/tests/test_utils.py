import pytest

from koi.util.utils import dst
import torch


def test_dst():
    x = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    y = torch.tensor([[1., 0., 0.], [2., 3., 4.]])

    assert (dst(x, y, dst_function='mse') == torch.tensor([[1., 0., 0.], [1., 4., 9.]])).all()
    assert (dst(x, y, dst_function='squared-euclidean') == torch.tensor([[1., 0., 0.], [1., 4., 9.]])).all()


if __name__ == '__main__':
    test_dst()

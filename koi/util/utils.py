import torch
import torch.nn.functional as F


def dst(recon_x, x, dst_function='mse'):
    # TODO move to tests
    assert recon_x.size(0) == x.size(0)
    assert recon_x.size(1) == x.size(1), "recon_x={0} different from x={1}".format(recon_x.size(1), x.size(1))
    x_dim = x.size(1)
    # those views are really necessary?
    x = x.view(-1, x_dim)
    mu = recon_x.view(-1, x_dim)

    if dst_function == 'mse':
        MSE = F.mse_loss(mu, x, reduction='none')
        d = MSE

    elif dst_function == 'bce':
        BCE = torch.nn.functional.binary_cross_entropy(mu, x, reduction='none') # .sum(dim=1)
        d = BCE

    elif dst_function == 'l2-norm':
        l2_norm = torch.linalg.norm(mu - x, keepdim=True)  #, dim=1)  # todo eh mannaggia c'era dim=1
        d = l2_norm

    elif dst_function == 'squared-euclidean':
        # it's the same compared to mse
        squared_euclidean = (x - mu) * (x - mu)
        d = squared_euclidean

    else:
        raise NotImplementedError()

    return d



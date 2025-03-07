import torch

from .misc import eps


def normalize_single_axis(d):
    # d is a arbitrary-shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d - dmin) / (eps + (dmax - dmin))
    return d


def normalize(d):
    # d is B x ???. 
    # normalize within each element of the batch
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single_axis(d[b])
    return out


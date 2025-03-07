import numpy as np
import torch

from .misc import DEVICE


def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0 * (grid_y / float(Y - 1)) - 1.0
    grid_x = 2.0 * (grid_x / float(X - 1)) - 1.0

    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_y, grid_x


def meshgrid2d(B, Y, X, stack=False, norm=False, device=DEVICE):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


# Reference:
#   https://github.com/xingyizhou/CenterNet/blob/a5a0483beb0f9e8705f6dc67d8817275369cfa7e/src/lib/utils/image.py#L126

def draw_umich_gaussian(heatmap, center, sigma, k=1):
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    gaussian = torch.tensor(gaussian2D((diameter, diameter), sigma=sigma))

    x, y = int(center[0]), int(center[1])
    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)

    masked_heatmap = heatmap[y - top  : y + bottom, 
                             x - left : x + right]

    masked_gaussian = gaussian[radius - top  : radius + bottom, 
                               radius - left : radius + right]

    if min(masked_gaussian.shape) > 0 \
    and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap # masked_heatmap ???


def gaussian2D(shape, sigma=1):
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


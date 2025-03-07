import torch

from .misc import DEVICE


def normalize_grid3d(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_z = 2.0 * (grid_z / float(Z - 1)) - 1.0
    grid_y = 2.0 * (grid_y / float(Y - 1)) - 1.0
    grid_x = 2.0 * (grid_x / float(X - 1)) - 1.0

    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_z, grid_y, grid_x


def meshgrid3d(B, Y, Z, X, stack=False, norm=False, device=DEVICE):
    '''
    returns a meshgrid sized B x Y x Z x X
    '''
    grid_z = torch.linspace(0.0, Z - 1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, 1, Z, 1])
    grid_z = grid_z.repeat(B, Y, 1, X)

    grid_y = torch.linspace(0.0, Y - 1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, Y, 1, 1])
    grid_y = grid_y.repeat(B, 1, Z, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Y, Z, 1)

    if norm:
        grid_y, grid_z, grid_x = normalize_grid3d(grid_y, grid_z, grid_x, Y, Z, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_y, grid_z, grid_x


def gridcloud3d(B, Y, Z, X, norm=False, device=DEVICE):
    # we want to sample for each location in the grid
    grid_y, grid_z, grid_x = meshgrid3d(B, Y, Z, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    # B x N -> B x N x 3
    xyz = torch.stack([x, y, z], dim=2)
    return xyz



import torch

from .intrinsics import split_intrinsics


def pixels2camera(x, y, z, fx, fy, x0, y0):
    """
    arguments:
        x, y, z can be images or pointclouds
            x and y are locations in pixel coordinates, 
            z is depth in meters
        fx, fy, x0, y0 are camera intrinsics
    
    return: xyz, sized B x N x 3
    """
    B = x.shape[0]

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    x = torch.reshape(x, [B, -1])
    y = torch.reshape(y, [B, -1])
    z = torch.reshape(z, [B, -1])

    # unproject
    x = (z / fx) * (x - x0)
    y = (z / fy) * (y - y0)

    xyz = torch.stack([x, y, z], dim=2) # B x N x 3
    return xyz


def camera2pixels(xyz, pix_T_cam, eps: float = 1e-4):
    """
    arguments: 
        xyz is shaped B x H*W x 3

    return: 
        xy is shaped B x H*W x 2
    """
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])
    x = torch.reshape(x, [B, -1])
    y = torch.reshape(y, [B, -1])
    z = torch.reshape(z, [B, -1])

    # z = torch.clamp(z, min=eps)
    z[z <= 0] = z[z <= 0].clamp(max=-eps)
    z[z >= 0] = z[z >= 0].clamp(min=eps)

    x = (x * fx) / z + x0
    y = (y * fy) / z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy


def xyd2pointcloud(xyd, pix_T_cam):
    """
    arguments: 
        xyd is like a pointcloud but in pixel coordinates;
            xy comes from a meshgrid bounded by H and W, 
            d comes from a depth map
    """
    B, N, C = xyd.shape
    assert (C == 3)
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(xyd[:, :, 0], xyd[:, :, 1], xyd[:, :, 2], fx, fy, x0, y0)
    return xyz


import torch


def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    x0 = K[:, 0, 2]
    y0 = K[:, 1, 2]
    return fx, fy, x0, y0


def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=fx.dtype, device=fx.device)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = x0
    K[:, 1, 2] = y0
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0
    return K


def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx * sx
    fy = fy * sy
    x0 = x0 * sx
    y0 = y0 * sy
    K = merge_intrinsics(fx, fy, x0, y0)
    return K


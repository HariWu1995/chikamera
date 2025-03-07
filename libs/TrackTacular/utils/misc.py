import torch


eps = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wrap2pi(rad_angle):
    # puts the angle into the range [-pi, pi]
    return torch.atan2(torch.sin(rad_angle), 
                       torch.cos(rad_angle))


def eye_4x4(B, device=DEVICE):
    I = torch.eye(4, device=device).view(1, 4, 4).repeat([B, 1, 1])
    return I


def apply_4x4(RT, xyz):
    B, N, _ = xyz.shape # N 8 4
    ones = torch.ones_like(xyz[:, :, 0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)

    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:, :, :3]
    return xyz2


def sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=eps, max=1-eps)


def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)


def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert (B == B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B * S] + otherdims)
    return tensor


def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert (BS % B == 0)
    otherdims = shapelist[1:]
    S = int(BS / B)
    tensor = torch.reshape(tensor, [B, S] + otherdims)
    return tensor


def reduce_masked_mean(x, mask, dim=None, keepdim: bool = False):
    """
    x and mask are the same shape, or at least broadcastably
    """
    for (a, b) in zip(x.size(), mask.size()):
        assert (a == b)
    # assert(x.size() == mask.size())

    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (denom + eps)
    return mean


def img_transform(img, resize_dims, crop):
    img = img.resize(resize_dims, Image.NEAREST)
    img = img.crop(crop)
    return img



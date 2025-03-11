import torch

from .misc import pack_seqdim, unpack_seqdim, matmul2, eye_4x4, apply_4x4


# parallel version
def safe_inverse(a):  
    B, _, _ = a.shape
    inv = a.clone()

    # inverse of rotation matrix
    r_transpose = a[:, :3, :3].transpose(1, 2)

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])
    return inv


def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3, 1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)

    bottom_row = a[3:4, :]  # = [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4)
    inv = torch.cat([inv, bottom_row], 0)
    return inv


def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t


def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert (C == 3)
    assert (D == 3)
    assert (B == B2)
    assert (D2 == 3)

    t = t.view(B, 3)

    rt = eye_4x4(B, device=t.device)
    rt[:, :3, :3] = r
    rt[:, :3, 3] = t
    return rt


def merge_rtlist(rlist, tlist):
    B, N, D, E = list(rlist.shape)
    assert (D == 3)
    assert (E == 3)

    B, N, F = list(tlist.shape)
    assert (F == 3)

    rlist_ = pack_seqdim(rlist, B)
    tlist_ = pack_seqdim(tlist, B)

    rtlist_ = merge_rt(rlist_, tlist_)
    rtlist = unpack_seqdim(rtlist_, B)
    return rtlist


def split_lrtlist(lrtlist):
    # splits a B x N x 19 tensor
    # into B x N x 3 (lens)
    # and B x N x 4 x 4 (rts)
    B, N, D = list(lrtlist.shape)
    assert (D == 19)

    len_list        = lrtlist[:, :, :3].reshape(B, N, 3)
    ref_T_objs_list = lrtlist[:, :, 3:].reshape(B, N, 4, 4)

    return len_list, ref_T_objs_list


def merge_lrtlist(lenlist, rtlist):
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4
    # merges these into a B x N x 19 tensor
    B, N, D = list(lenlist.shape)
    assert (D == 3)

    B2, N2, E, F = list(rtlist.shape)
    assert (B == B2)
    assert (N == N2)
    assert (E == 4 and F == 4)

    rtlist = rtlist.reshape(B, N, 16)
    lrtlist = torch.cat([lenlist, rtlist], axis=2)
    return lrtlist


def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    B, N, D = list(lrtlist_X.shape)
    assert (D == 19)

    B2, E, F = list(Y_T_X.shape)
    assert (B2 == B)
    assert (E == 4 and F == 4)

    lenlist, rtlist_X = split_lrtlist(lrtlist_X)

    Y_T_Xs = Y_T_X.unsqueeze(1).repeat(1, N, 1, 1)
    Y_T_Xs_ = Y_T_Xs.view(B * N, 4, 4)

    rtlist_X_ = rtlist_X.reshape(B * N, 4, 4)
    rtlist_Y_ = matmul2(Y_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.reshape(B, N, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y


def apply_4x4_to_lrt(Y_T_X, lrt_X):
    B, D = list(lrt_X.shape)
    assert (D == 19)

    B2, E, F = list(Y_T_X.shape)
    assert (B2 == B)
    assert (E == 4 and F == 4)

    return apply_4x4_to_lrtlist(Y_T_X, lrt_X.unsqueeze(1)).squeeze(1)


def get_xyzlist_from_lenlist(lenlist):
    B, N, D = list(lenlist.shape)
    assert (D == 3)
    lx, ly, lz = torch.unbind(lenlist, axis=2)

    xs = torch.stack([lx / 2.,  lx / 2., -lx / 2., -lx / 2.,  lx / 2.,  lx / 2., -lx / 2., -lx / 2.], axis=2)
    ys = torch.stack([ly / 2.,  ly / 2.,  ly / 2.,  ly / 2., -ly / 2., -ly / 2., -ly / 2., -ly / 2.], axis=2)
    zs = torch.stack([lz / 2., -lz / 2., -lz / 2.,  lz / 2.,  lz / 2., -lz / 2., -lz / 2.,  lz / 2.], axis=2)

    # B x N x 8 -> B x N x 8 x 3
    xyzlist = torch.stack([xs, ys, zs], axis=3)
    return xyzlist


def get_xyzlist_from_lrtlist(lrtlist, include_clist=False):
    B, N, D = list(lrtlist.shape)
    assert (D == 19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist = get_xyzlist_from_lenlist(lenlist)
    # xyzlist is B x N x 8 x 3

    rtlist_  =  rtlist.reshape(B * N, 4, 4)
    xyzlist_ = xyzlist.reshape(B * N, 8, 3)

    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 8, 3)

    if include_clist:
        clist_cam = get_clist_from_lrtlist(lrtlist).unsqueeze(2)
        xyzlist_cam = torch.cat([xyzlist_cam, clist_cam], dim=2)
    return xyzlist_cam


def get_clist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert (D == 19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist = torch.zeros((B, N, 1, 3), device=lrtlist.device)
    # xyzlist is B x N x 8 x 3

    rtlist_  =  rtlist.reshape(B * N, 4, 4)
    xyzlist_ = xyzlist.reshape(B * N, 1, 3)

    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam

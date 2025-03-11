import torch
import torch.nn as nn

from .ops.attn import VanillaSelfAttention, SpatialCrossAttention as SpatialXAttention
from .ops.encoder import Encoder_res101, Encoder_res50, Encoder_res18, Encoder_eff, Encoder_swin_t
from .ops.decoder import Decoder

from ..utils.misc import pack_seqdim, unpack_seqdim, apply_4x4
from ..utils.misc3d import gridcloud3d
from ..utils.conversion import camera2pixels


# no radar / lidar integration
class Bevformer(nn.Module):

    def __init__(self, Y, Z, X,
                 rand_flip: bool = False,
                 latent_dim: int = 128,
                 feat2d_dim: int = 128,
                num_classes: int = None,
                     z_sign: int = 1,
               encoder_type: str = 'swin_t',
        ):
        super(Bevformer, self).__init__()

        assert (encoder_type in ['res101','res50','res34','res18','effb0','effb4','swin_t']), \
            f'encoder_type = {encoder_type} is not supported!'
        self.encoder_type = encoder_type

        self.Y, self.Z, self.X = Y, Z, X
        self.z_sign = z_sign

        self.use_radar = False
        self.use_lidar = False
        self.rand_flip = rand_flip

        self.feat2d_dim = feat2d_dim
        self.latent_dim = latent_dim

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float().cuda()
        self.std  = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float().cuda()

        # Encoder
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == 'res34':
            self.encoder = Encoder_res34(self.feat2d_dim)
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        elif encoder_type == 'effb4':
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEVFormer self & cross attention layers
        self.bev_keys = nn.Linear(feat2d_dim, latent_dim)
        self.bev_queries     = nn.Parameter(0.1 * torch.randn(latent_dim, Y, X))  # C, Y, X
        self.bev_queries_pos = nn.Parameter(0.1 * torch.randn(latent_dim, Y, X))  # C, Y, X
        
        num_layers = 6
        self.num_layers = num_layers

        self.self_attn_layers = nn.ModuleList([VanillaSelfAttention(dim=latent_dim) for _ in range(num_layers)])
        self.cross_attn_layers = nn.ModuleList([SpatialXAttention(dim=latent_dim) for _ in range(num_layers)])

        ffn_dim = 512
        self.ffn_layers = nn.ModuleList([nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.ReLU(), 
                                                       nn.Linear(ffn_dim, latent_dim)) for _ in range(num_layers)])

        self.norm1_layers = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self.norm3_layers = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])

        self.bev_temporal = nn.Sequential(
                    nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), 
                    nn.ReLU(),
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        # Decoder
        self.decoder = Decoder(in_channels=latent_dim, n_classes=num_classes, feat2d=feat2d_dim,)

        # Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.tracking_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, prev_bev=None):
        """
        Notation:
            B = batch size, 
            S = number of cameras, 
            C = 3, 
            H = img height, 
            W = img width

        Arguments:
                 rgb_cams: (B,S,C,H,W)
               pix_T_cams: (B,S,4,4)
            cams_T_global: (B,S,4,4)
             ref_T_global: (B,4,4)
                 vox_util: vox util object
        """
        B, S, C, H, W = rgb_cams.shape
        assert (C == 3)

        B0 = B * S
        device = rgb_cams.device

        # reshape tensors
        __p = lambda x:   pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)

        rgb_cams_      = __p(rgb_cams)          # B*S, 3, H, W
        pix_T_cams_    = __p(pix_T_cams)        # B*S, 4, 4
        cams_T_global_ = __p(cams_T_global)     # B*S, 4, 4
        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S, 4, 4

        ref_T_cams = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams)  # B*S,4,4

        # rgb encoder
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,128,H/8,W/8

        _, C, Hf, Wf = feat_cams_.shape
        feat_cams = __u(feat_cams_)  # B,S,C,Hf,Wf

        Y, Z, X = self.Y, self.Z, self.X

        # compute the image locations (no flipping for now)
        xyz_mem_ = gridcloud3d(B0, Y, Z, X, norm=False, device=device)  # B0, Z*Y*X, 3
        xyz_ref_ = vox_util.Mem2Ref(xyz_mem_, Y, Z, X, assert_cube=False)
        xyz_cams_ = apply_4x4(cams_T_ref_, xyz_ref_)
        xy_cams_ = camera2pixels(xyz_cams_, pix_T_cams_)  # B0, N, 2

        # bev coords project to pixel level and normalized  S,B,Y*X,Z,2
        ref_pts_cam = xy_cams_.reshape(B, S, Y, Z, X, 2).permute(1, 0, 2, 4, 3, 5).reshape(S, B, Y * X, Z, 2)
        ref_pts_cam[..., 0:1] = ref_pts_cam[..., 0:1] / float(W)
        ref_pts_cam[..., 1:2] = ref_pts_cam[..., 1:2] / float(H)
    
        cam_x = xyz_cams_[..., 2].reshape(B, S, Y, Z, X, 1).permute(1, 0, 2, 4, 3, 5)\
                                 .reshape(S, B, Y * X, Z, 1)

        bev_mask = ((ref_pts_cam[..., 1:2] >= 0.0)
                  & (ref_pts_cam[..., 1:2] <= 1.0)
                  & (ref_pts_cam[..., 0:1] <= 1.0)
                  & (ref_pts_cam[..., 0:1] >= 0.0)
                  & (self.z_sign * cam_x >= 0.0)).squeeze(-1)  # S,B,Y*X,Z

        # self-attention prepare -> B, Y*X, C
        bev_queries = self.bev_queries.clone().unsqueeze(0)\
                                                 .repeat(B, 1, 1, 1)\
                                                .reshape(B, self.latent_dim, -1) \
                                                .permute(0, 2, 1)

        bev_queries_pos = self.bev_queries_pos.clone().unsqueeze(0)\
                                                         .repeat(B, 1, 1, 1) \
                                                        .reshape(B, self.latent_dim, -1)\
                                                        .permute(0, 2, 1)

        # cross-attention prepare
        bev_keys = feat_cams.reshape(B, S, C, Hf * Wf).permute(1, 3, 0, 2)  # S, Hf*Wf, B, C
        bev_keys = self.bev_keys(bev_keys)

        spatial_shapes = bev_queries.new_zeros([1, 2]).long()
        spatial_shapes[0, 0] = Hf
        spatial_shapes[0, 1] = Wf

        for i in range(self.num_layers):
            # self attention within the features (B, Y*X, C)
            bev_queries = self.self_attn_layers[i](bev_queries, self.Y, self.X, bev_queries_pos)

            # normalize (B, Y*X, C)
            bev_queries = self.norm1_layers[i](bev_queries)

            # cross attention into the images
            bev_queries = self.cross_attn_layers[i](bev_queries, bev_keys, bev_keys,
                                                    query_pos=bev_queries_pos,
                                                    ref_pts_cam=ref_pts_cam,
                                                spatial_shapes=spatial_shapes,
                                                    bev_mask=bev_mask)

            # normalize (B, N, C)
            bev_queries = self.norm2_layers[i](bev_queries)

            # feedforward layer (B, N, C) + residual
            bev_queries += self.ffn_layers[i](bev_queries)

            # normalize (B, N, C)
            bev_queries = self.norm3_layers[i](bev_queries)

        feat_bev = bev_queries.permute(0, 2, 1).reshape(B, self.latent_dim, self.Y, self.X)

        if prev_bev is None:
            prev_bev = feat_bev

        feat_bev = torch.cat([feat_bev, prev_bev], dim=1)
        feat_bev = self.bev_temporal(feat_bev)

        # bev decoder
        out_dict = self.decoder(feat_bev, feat_cams_)

        return out_dict

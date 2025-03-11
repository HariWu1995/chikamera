import torch
import torch.nn as nn

from .attn_ms import MSDeformAttn, MSDeformAttn3D


class VanillaSelfAttention(nn.Module):

    def __init__(self, dim: int = 128, dropout: float = 0.5):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=4, n_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, Y, X, query_pos=None):
        """
        query    : (B, Y*X, C)
        query_pos: (B, Y*X, C)
        """
        inp_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        # Y, X = 200, 200

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, Y - 0.5, Y, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X - 0.5, X, dtype=torch.float, device=query.device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / Y
        ref_x = ref_x.reshape(-1)[None] / X

        reference_points = torch.stack((ref_y, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2)  # (B, Y*X, 1, 2)

        input_spatial_shapes = query.new_zeros([1, 2]).long()
        input_spatial_shapes[0, 0] = Y
        input_spatial_shapes[0, 1] = X
        input_level_start_index = query.new_zeros([1, ]).long()

        queries = self.deformable_attention(query=query,
                                            reference_points=reference_points,
                                            input_flatten=query.clone(),
                                            input_spatial_shapes=input_spatial_shapes,
                                            input_level_start_index=input_level_start_index)
        queries = self.output_proj(queries)
        return self.dropout(queries) + inp_residual


# From https://github.com/zhiqi-li/BEVFormer
class SpatialCrossAttention(nn.Module):

    def __init__(self, dim: int = 128, dropout: float = 0.5):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim, num_heads=4, num_levels=1, num_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, 
                    query_pos=None, ref_pts_cam=None, 
                    spatial_shapes=None, bev_mask=None):
        """
        query: bev_queries          (B, Y*X, C)
          key: bev_keys             (S, Hf*Wf, B, C)
        value: bev_keys             (S, Hf*Wf, B, C)
        query_pos                   (B, Y*X, C)
        ref_pts_cam                 (S, B, Y*X, Z, 2) normalized
        spatial_shapes              (1,2) [[Hf,Wf]]
        bev_mask                    (S. B, Y*X, Z)
        """
        inp_residual = query
        slots = torch.zeros_like(query)  # (B, Y*X, C)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape   # N=Y*X
        S, M, _, _ = key.shape  # M=Hf*Wf
        D = ref_pts_cam.size(3)  # Z

        """
        Traverse the S cameras, 
        take the index of the valid coordinates of the BEV projection to each feature map
        find out which cam having most valid BEV projection coords and get the max_len
        """
        # for i, mask_per_img in enumerate(bev_mask):
        #     # if once valid through Z-axis, query it
        #     index_query = mask_per_img.sum(-1)
        #     indexes.append(index_query)
        max_len = bev_mask.sum(dim=-1).gt(0).sum(-1).max()

        # for each batch and cam reconstruct the query and reference_points
        queries_rebatch =       query.new_zeros([B, S, max_len, self.dim])
        ref_pts_rebatch = ref_pts_cam.new_zeros([B, S, max_len, D, 2])

        for j in range(B):
            for i, ref_pts in enumerate(ref_pts_cam):
                index_query = bev_mask[i, j].sum(-1).nonzero().squeeze(-1)
                queries_rebatch[j, i, :len(index_query)] =   query[j, index_query]
                ref_pts_rebatch[j, i, :len(index_query)] = ref_pts[j, index_query]

        # take feature map as key and value of attention module
        key   =   key.permute(2, 0, 1, 3).reshape(B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(B * S, M, C)

        level_start_index = query.new_zeros([1, ]).long()

        ref_pts_rebatch = ref_pts_rebatch.view(B * S, max_len, D, 2)
        # ref_pts_rebatch = torch.mean(ref_pts_rebatch, dim=-2, keepdim=True)
        
        queries = self.deformable_attention(
                        query = queries_rebatch.view(B * S, max_len, self.dim),
                          key = key,
                        value = value,
             reference_points = ref_pts_rebatch,
               spatial_shapes = spatial_shapes,
            level_start_index = level_start_index).view(B, S, max_len, self.dim)

        for j in range(B):
            for i in range(S):
                # slots (B, Y*X, C)
                index_query = bev_mask[i, j].sum(-1).nonzero().squeeze(-1)
                slots[j, index_query] += queries[j, i, :len(index_query)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)

        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual



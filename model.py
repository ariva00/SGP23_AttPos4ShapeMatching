import torch
import torch.nn as nn
from x_transformers import Encoder
from point_gaussian import GaussianAttention


class EncoderPointTransfomer(nn.Module):
    def __init__(
            self,
            dim=512,
            heads=8,
            gaussian_heads=0,
            inf_gaussian_heads=0,
            sigma=[],
            dim_head=64,
            custom_layers=None,
            force_cross_attn=False,
            force_self_attn=False,
            depth=6,
            infer_sigma=False
            ) -> None:
        super(EncoderPointTransfomer, self).__init__()

        self.gaussian_heads = gaussian_heads
        self.inf_gaussian_heads = inf_gaussian_heads
        self.force_cross_attn = force_cross_attn
        self.force_self_attn = force_self_attn

        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head_custom = dim_head,
            attn_dim_head = dim_head,
            pre_norm=False,
            residual_attn=True,
            rotary_pos_emb=True,
            rotary_emb_dim = dim_head,
            custom_layers=custom_layers,
            gauss_gaussian_heads=gaussian_heads + inf_gaussian_heads,
            infer_sigma=infer_sigma
        )

        self.infer_sigma = infer_sigma

        self.gauss_attn = GaussianAttention(sigma if not infer_sigma else [])

        self.linear_in = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, dim)
        )

        self.linear_out = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
    
    def forward(self, x:torch.Tensor, sep_idx=None, mask_head=[], return_hiddens=False):
        if sep_idx is None:
            sep_idx = x.shape[1] // 2
        dim1 = sep_idx
        dim2 = sep_idx + 1

        if self.gaussian_heads and not self.infer_sigma:
            shape1_gaussian_attn = self.gauss_attn(x[:, :dim1])
            shape2_gaussian_attn = self.gauss_attn(x[:, dim2:])
        
        
        points = x if self.infer_sigma else None

        x = self.linear_in(x)
        attn_mask = torch.ones((8, x.shape[1], x.shape[1]), device=x.device) if self.force_cross_attn or self.force_self_attn or mask_head else None
        fixed_attn = torch.zeros((x.shape[0], self.gaussian_heads + self.inf_gaussian_heads, x.shape[1], x.shape[1]), device=x.device) if (self.gaussian_heads or self.inf_gaussian_heads) and not self.infer_sigma else None
        if (self.gaussian_heads or self.inf_gaussian_heads) and not self.infer_sigma:
            if self.gaussian_heads:
                fixed_attn[:, self.inf_gaussian_heads:, :dim1, :dim1] = shape1_gaussian_attn
                fixed_attn[:, self.inf_gaussian_heads:, dim2:, dim2:] = shape2_gaussian_attn
            if self.inf_gaussian_heads:
                fixed_attn[:, :self.inf_gaussian_heads, :dim1, :dim1] = 1
                fixed_attn[:, :self.inf_gaussian_heads, dim2:, dim2:] = 1

        if self.force_self_attn:
            attn_mask[-self.force_self_attn:, :dim1, dim2:] = 0
            attn_mask[-self.force_self_attn:, dim2:, :dim1] = 0
        if self.force_cross_attn:
            attn_mask[:self.force_cross_attn, :dim1, :dim1] = 0
            attn_mask[:self.force_cross_attn, dim2:, dim2:] = 0

        if mask_head:
            attn_mask[mask_head, :, :] = 0

        if attn_mask is not None:
            attn_mask = attn_mask.type(torch.bool)

        if return_hiddens:
            x, hiddens = self.encoder(x, gaussian_attn=fixed_attn, shape_sep_idx=dim1, attn_mask=attn_mask, return_hiddens=True, points=points)
        else:
            x = self.encoder(x, gaussian_attn=fixed_attn, shape_sep_idx=dim1, attn_mask=attn_mask, points=points)
        x = self.linear_out(x)

        if return_hiddens:
            return x, hiddens
        return x

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class GlobalAttention3D(nn.Module):
    """
    3D Global Self-Attention (Standard ViT Attention)
    Unlike Swin, this calculates attention across ALL tokens in the feature map.
    Perfect for bottleneck layers where feature map size is small.
    """

    def __init__(
        self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (Batch, Num_Tokens, Dim)
        B, N, C = x.shape

        #
        # QKV projection
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: (B, Heads, N, Head_Dim)

        # Attention Score: (Q @ K.T) * scale
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Weighted Sum: Attn @ V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock3D(nn.Module):
    """
    3D Vision Transformer Block
    Structure: Norm -> Global Attention -> Norm -> MLP
    """

    def __init__(
        self,
        in_channels,
        mlp_ratio=4.0,
        num_heads=8,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = in_channels

        # 1. Global Attention
        self.norm1 = norm_layer(self.dim)
        self.attn = GlobalAttention3D(
            self.dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )

        # 2. Stochastic Depth (Drop Path)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 3. MLP (Feed Forward Network)
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, self.dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # Input: (B, C, D, H, W) -> Output: (B, C, D, H, W)
        b, c, d, h, w = x.shape

        # Flatten: (B, C, D, H, W) -> (B, D*H*W, C)
        # 注意：ViT 需要 (Batch, Tokens, Channels) 的形狀
        x_flat = x.flatten(2).transpose(1, 2)

        # Block 1: Attention
        x_flat = x_flat + self.drop_path(self.attn(self.norm1(x_flat)))

        # Block 2: MLP
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))

        # Reshape back: (B, D*H*W, C) -> (B, C, D, H, W)
        x_out = x_flat.transpose(1, 2).view(b, c, d, h, w)

        return x_out

import torch
import torch.nn.functional as F
from timm.layers import DropPath
from torch import nn


class CrossAttention(nn.Module):
    """Multi-head cross-attention module.

    Args:
        dim: Input feature dimension (must equal ``num_heads * head_dim``).
        num_heads: Number of attention heads.
        qkv_bias: Whether to add learnable biases to Q, K, V projections.
        qk_scale: Override for the attention scale factor. Defaults to
            ``head_dim ** -0.5``.
        attn_drop: Dropout probability applied to attention weights.
        proj_drop: Dropout probability applied to the output projection.
        attn_head_dim: Per-head dimension. Defaults to ``dim // num_heads``.
        out_dim: Output dimension. Defaults to ``dim``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: int | None = None,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads if attn_head_dim is None else attn_head_dim
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q = F.linear(x, self.q.weight, self.q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        k = F.linear(k, self.k.weight, self.k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(v, self.v.weight, self.v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj_drop(self.proj(x))


class AttentiveBlock(nn.Module):
    """Transformer block with cross-attention between query and key-value inputs."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        attn_head_dim: int | None = None,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            out_dim=out_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        pos_q: torch.Tensor | int,
        pos_k: torch.Tensor | int,
        bool_masked_pos=None,
        rel_pos_bias=None,
    ) -> torch.Tensor:
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        return self.cross_attn(x_q, k=x_k, v=x_v)


class AttentionPoolingBlock(AttentiveBlock):
    """Attention-based pooling: compresses a sequence to a single vector via
    cross-attention from the sequence mean to all tokens."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = x.mean(1, keepdim=True)
        x = super().forward(x_q, x, pos_q=0, pos_k=0)
        return x.squeeze(1)

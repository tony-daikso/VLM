import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from totalfm.attention_pooling import AttentionPoolingBlock
from totalfm.position_encoding import PositionEmbeddingLearned3d


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )
        attn = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.to_out(out)


class _Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _PreNorm(dim, _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        _PreNorm(dim, _FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """3D Vision Transformer for CT volume encoding.

    Accepts volumetric input of shape ``(B, C, H, W, D)`` and returns
    embeddings of shape ``(B, out_dim)``.

    Args:
        image_size: Spatial size (H and W) of the input volume.
        image_patch_size: Spatial patch size.
        frames: Depth (D) of the input volume.
        frame_patch_size: Depth patch size.
        dim: Transformer hidden dimension.
        depth: Number of Transformer layers.
        heads: Number of attention heads.
        mlp_dim: Feed-forward hidden dimension.
        channels: Number of input channels (1 or 3).
        dim_head: Per-head dimension (typically ``dim // heads``).
        dropout: Dropout probability inside Transformer layers.
        emb_dropout: Dropout probability applied to patch embeddings.
        out_dim: Output embedding dimension after the projection head.
        reduction: Pooling strategy. Must be ``"attn_pool"``.
    """

    def __init__(
        self,
        *,
        image_size: int,
        image_patch_size: int,
        frames: int,
        frame_patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        out_dim: int = 768,
        reduction: str = "attn_pool",
    ) -> None:
        super().__init__()
        assert image_size % image_patch_size == 0, (
            "Image size must be divisible by patch size."
        )
        assert frames % frame_patch_size == 0, (
            "Frame count must be divisible by frame patch size."
        )

        self.patch_height = image_patch_size
        self.patch_width = image_patch_size
        self.frame_patch_size = frame_patch_size

        patch_dim = channels * image_patch_size * image_patch_size * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)",
                p1=image_patch_size,
                p2=image_patch_size,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionEmbeddingLearned3d(
            dim // 3,
            image_size // image_patch_size,
            image_size // image_patch_size,
            frames // frame_patch_size,
        )

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.reduction = reduction

        assert reduction == "attn_pool", (
            f"Unsupported reduction '{reduction}'. Only 'attn_pool' is supported."
        )
        self.pool = AttentionPoolingBlock(
            dim=dim,
            num_heads=heads,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            out_dim=dim,
        )

        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim, bias=False))

    def forward(self, video: torch.Tensor, _: None) -> torch.Tensor:
        """Forward pass.

        Args:
            video: Input tensor of shape ``(B, C, H, W, D)``.
            _: Unused (attention mask placeholder for API compatibility).

        Returns:
            Embedding tensor of shape ``(B, out_dim)``.
        """
        B, C, H, W, D = video.shape

        x = self.to_patch_embedding(video)
        pos = self.pos_embedding(
            B,
            H // self.patch_height,
            W // self.patch_width,
            D // self.frame_patch_size,
            x,
        )
        x = self.emb_dropout(x + pos)
        x = self.transformer(x)
        x = self.pool(x)
        return self.proj(x)

import torch
from einops import rearrange
from torch import nn


def _make_indices(n_steps: int, n_ref: int, device: torch.device) -> torch.Tensor:
    """Sample n_steps indices uniformly from [0, n_ref-1]."""
    idx = torch.linspace(0, n_ref - 1, steps=n_steps, device=device)
    return torch.round(idx).long().clamp_(0, n_ref - 1)


class PositionEmbeddingLearned3d(nn.Module):
    """Learned absolute 3D position embedding for (H, W, D) patch grids.

    Args:
        num_pos_feats: Embedding dimension per axis. The full position embedding
            dimension is ``3 * num_pos_feats``.
        h_patch_num: Maximum number of patches along the height axis.
        w_patch_num: Maximum number of patches along the width axis.
        d_patch_num: Maximum number of patches along the depth axis.
    """

    def __init__(
        self,
        num_pos_feats: int = 256,
        h_patch_num: int = 16,
        w_patch_num: int = 16,
        d_patch_num: int = 64,
    ) -> None:
        super().__init__()
        self.h_patch_num = h_patch_num
        self.w_patch_num = w_patch_num
        self.d_patch_num = d_patch_num
        self.row_embed = nn.Embedding(h_patch_num, num_pos_feats)
        self.col_embed = nn.Embedding(w_patch_num, num_pos_feats)
        self.dep_embed = nn.Embedding(d_patch_num, num_pos_feats)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.dep_embed.weight)

    def forward(
        self,
        B: int,
        h: int,
        w: int,
        d: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute position embeddings for a (B, h*w*d, C) patch sequence.

        Args:
            B: Batch size.
            h: Number of patches along height.
            w: Number of patches along width.
            d: Number of patches along depth.
            x: Patch tensor (used only to obtain the device).

        Returns:
            Position embedding of shape ``(B, h*w*d, 3*num_pos_feats)``.
        """
        i = _make_indices(h, self.h_patch_num, x.device)
        j = _make_indices(w, self.w_patch_num, x.device)
        k = _make_indices(d, self.d_patch_num, x.device)

        x_emb = self.row_embed(i).unsqueeze(1).unsqueeze(2).repeat(1, w, d, 1)
        y_emb = self.col_embed(j).unsqueeze(0).unsqueeze(2).repeat(h, 1, d, 1)
        z_emb = self.dep_embed(k).unsqueeze(0).unsqueeze(1).repeat(h, w, 1, 1)

        pos = torch.cat([x_emb, y_emb, z_emb], dim=-1)           # (h, w, d, 3*F)
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1, 1)              # (B, h, w, d, 3*F)
        pos = rearrange(pos, "b h w d c -> b (h w d) c")
        return pos

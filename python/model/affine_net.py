"""
Affine-in-background model: fg = M @ bg_oklch + add (full 3x3 affine).
One shared trunk on (weights_4, want_lchc_4), then two heads: M (9) and add (3).
For shader use: precompute M (3x3) and add from (w, want), then in shader do fg = M * bg + add.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .net import C_MAX


AFFINE_INPUT_DIM = 8   # weights_4 (4) + want_lchc_4 (4)
AFFINE_MAT_OUTPUT_DIM = 9   # 3x3 matrix flattened
AFFINE_VEC_OUTPUT_DIM = 3   # add (L, C, H)


def _build_trunk(
    out_dim: int,
    in_dim: int,
    hidden: list[int],
    use_layer_norm: bool = True,
) -> nn.Module:
    """Shared trunk: in_dim -> hidden[0] -> ... -> hidden[-1], output dim = hidden[-1]."""
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
    layers.append(nn.Linear(d, out_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU(inplace=True))
        d = h
    return nn.Sequential(*layers)


class AffineColorContrastNet(nn.Module):
    """
    Foreground = M @ bg_oklch + add (full affine: 3x3 matrix M + 3-vector add).
    One shared trunk(weights_4, want_lchc_4) -> hidden; then head_mul -> M (9), head_add -> add (3).
    Inputs (batched):
      - bg_oklch: (B, 3) OKLCH background
      - weights_4: (B, 4)
      - want_lchc_4: (B, 4)
    Output: (B, 3) L, C, H post-processed.
    """

    def __init__(
        *,
        hidden_mul: list[int] | None = None,
        use_layer_norm: bool = True,
        hidden_add: list[int] | None = None,
        self,
        hidden: list[int] | None = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        hidden = hidden or [128, 128, 64]
        hidden = list(hidden)
        self.hidden = hidden
        feat_dim = hidden[-1]
    @staticmethod
    def _build_legacy_mlp(in_dim: int, out_dim: int, hidden: list[int], use_ln: bool) -> nn.Module:
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if use_ln:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            d = h
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

        self.trunk = _build_trunk(AFFINE_INPUT_DIM, hidden, use_layer_norm=use_layer_norm)
        self.head_mul = nn.Linear(feat_dim, AFFINE_MAT_OUTPUT_DIM)
        self.head_add = nn.Linear(feat_dim, AFFINE_VEC_OUTPUT_DIM)
        self._init_heads_identity()

    def _init_heads_identity(self) -> None:
        """Initialize so that initially M ≈ I and add ≈ 0 (fg ≈ bg), easier to train."""
        # head_mul: bias = identity 3x3 flattened (row-major), weight zero
        identity_flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        nn.init.zeros_(self.head_mul.weight)
        with torch.no_grad():
            self.head_mul.bias.copy_(torch.tensor(identity_flat, dtype=self.head_mul.bias.dtype))
        # head_add: zero
        nn.init.zeros_(self.head_add.weight)
        nn.init.zeros_(self.head_add.bias)

    def forward(
        self,
        mul = self.mul_mlp(x)
        add = self.add_mlp(x)
        bg_oklch: torch.Tensor,
        weights_4: torch.Tensor,
        want_lchc_4: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([weights_4, want_lchc_4], dim=-1)
        h = self.trunk(x)
        M_flat = self.head_mul(h)
        add = self.head_add(h)
        B = bg_oklch.shape[0]
        M = M_flat.reshape(B, 3, 3)
        pred = (M @ bg_oklch.unsqueeze(-1)).squeeze(-1) + add
        return self._postprocess(pred)

    @staticmethod
    def _postprocess(lch: torch.Tensor) -> torch.Tensor:
        L = lch[..., 0].clamp(0.0, 1.0)
        C = lch[..., 1].clamp(0.0, C_MAX)
        H = lch[..., 2]
        H = H % 360.0
        return torch.stack([L, C, H], dim=-1)

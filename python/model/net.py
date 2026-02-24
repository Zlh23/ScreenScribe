"""
Single MLP: concat(bg_oklch, weights_4, want_lchc_4) -> hidden -> (L, C, H).
Post-process: L/C clamp, H in [0, 360).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# L in [0,1], C in [0, 0.4] (OKLCH chroma typical max), H in [0, 360)
C_MAX = 0.4

INPUT_DIM = 11  # 3 + 4 + 4
OUTPUT_DIM = 3  # L, C, H


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden: list[int],
    use_layer_norm: bool = True,
) -> nn.Module:
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU(inplace=True))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ColorContrastNet(nn.Module):
    """
    Inputs (all batched, last dim = features):
      - bg_oklch: (B, 3) OKLCH background
      - weights_4: (B, 4) w_contrast, w_L, w_C, w_H
      - want_lchc_4: (B, 4) wantL, wantC, wantH, wantContrast
    Output:
      - lch: (B, 3) L, C, H (post-processed: L in [0,1], C in [0, C_MAX], H in [0, 360))
    """

    def __init__(
        self,
        hidden: list[int] | None = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        hidden = hidden or [128, 128, 64]
        self.mlp = _build_mlp(INPUT_DIM, OUTPUT_DIM, hidden, use_layer_norm=use_layer_norm)

    def forward(
        self,
        bg_oklch: torch.Tensor,
        weights_4: torch.Tensor,
        want_lchc_4: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([bg_oklch, weights_4, want_lchc_4], dim=-1)
        lch = self.mlp(x)
        return self._postprocess(lch)

    @staticmethod
    def _postprocess(lch: torch.Tensor) -> torch.Tensor:
        L = lch[..., 0].clamp(0.0, 1.0)
        C = lch[..., 1].clamp(0.0, C_MAX)
        H = lch[..., 2]
        H = H % 360.0
        return torch.stack([L, C, H], dim=-1)

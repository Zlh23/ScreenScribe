"""
Weighted loss: w_contrast*L_contrast + w_L*L_L + w_C*L_C + w_H*L_H.
Expects pred_contrast to be from differentiable APCA(pred_fg_srgb, bg_srgb) so gradient flows.
Note: APCA is polarity-sensitive (light-on-dark is negative). Here we match contrast *magnitude*,
so the model can satisfy targets on both light and dark backgrounds.
With scale=True (default), each term is normalized to ~[0,1] so equal weights = equal importance.
"""
from __future__ import annotations

import torch

PI = 3.141592653589793

# Typical ranges for normalizing loss terms to comparable scale
CONTRAST_SCALE = 75.0   # APCA Lc typical range
C_MAX_SQ = 0.4 ** 2    # C in [0, 0.4], squared err max
L_H_MAX = 2.0          # 1 - cos in [0, 2]


def weighted_loss(
    pred_lch: torch.Tensor,
    want_lch: torch.Tensor,
    want_contrast: torch.Tensor,
    pred_contrast: torch.Tensor,
    weights_4: torch.Tensor,
    scale: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    pred_lch, want_lch: (B, 3) L, C, H
    want_contrast: (B,) target APCA Lc
    pred_contrast: (B,) predicted APCA Lc (from differentiable apca_contrast). May be signed.
    weights_4: (B, 4) w_contrast, w_L, w_C, w_H
    scale: if True, normalize each term to ~[0,1] so weighted sum is balanced.
    Returns (loss, components): loss is scalar mean over batch; components has
    keys "L_contrast", "L_L", "L_C", "L_H" with batch-mean scalars.
    """
    w_c, w_l, w_c_, w_h = weights_4[..., 0], weights_4[..., 1], weights_4[..., 2], weights_4[..., 3]
    L_contrast = (pred_contrast.abs() - want_contrast).abs()
    L_L = (pred_lch[..., 0] - want_lch[..., 0]).pow(2)
    L_C = (pred_lch[..., 1] - want_lch[..., 1]).pow(2)
    h_diff = (pred_lch[..., 2] - want_lch[..., 2]) * (PI / 180.0)
    L_H = 1.0 - torch.cos(h_diff)
    if scale:
        L_contrast = L_contrast / CONTRAST_SCALE
        L_C = L_C / C_MAX_SQ
        L_H = L_H / L_H_MAX
    per_sample = w_c * L_contrast + w_l * L_L + w_c_ * L_C + w_h * L_H
    loss = per_sample.mean()
    components = {
        "L_contrast": L_contrast.mean(),
        "L_L": L_L.mean(),
        "L_C": L_C.mean(),
        "L_H": L_H.mean(),
    }
    return loss, components

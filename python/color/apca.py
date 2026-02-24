"""
Differentiable APCA-W3 contrast in PyTorch.
Input: sRGB in [0,1] (e.g. from oklch_to_srgb), shape (..., 3).
Output: Lc contrast per sample (polarity-sensitive), shape (...).
"""
from __future__ import annotations

import torch

# APCA-W3 constants (0.0.98G-4g-sRGB)
Ntx = 0.57
Nbg = 0.56
Rtx = 0.62
Rbg = 0.65
B_thrsh = 0.022
B_clip = 1.414
W_scale = 1.14
W_offset = 0.027
W_clamp = 0.1

# sRGB to Ys coefficients (R,G,B in 0..1: Ys = R^2.4*0.2126729 + G^2.4*0.7151522 + B^2.4*0.0721750)
_YS_COEF = (0.2126729, 0.7151522, 0.0721750)
_S_TRC = 2.4


def _srgb_to_ys(srgb: torch.Tensor) -> torch.Tensor:
    """sRGB in [0,1], last dim 3 -> Ys. Differentiable."""
    r, g, b = srgb[..., 0], srgb[..., 1], srgb[..., 2]
    r = r.clamp(min=1e-8)
    g = g.clamp(min=1e-8)
    b = b.clamp(min=1e-8)
    ys = (
        torch.pow(r, _S_TRC) * _YS_COEF[0]
        + torch.pow(g, _S_TRC) * _YS_COEF[1]
        + torch.pow(b, _S_TRC) * _YS_COEF[2]
    )
    return ys


def _soft_clip(yc: torch.Tensor) -> torch.Tensor:
    """
    f_softclp(Y_c): 0 if Y_c<0; Y_c + (B_thrsh - Y_c)^B_clip if Y_c < B_thrsh; else Y_c.
    Differentiable: use smooth gradient at boundaries. For Y_c < 0 return 0 (grad 0).
    """
    yc = yc.clamp(min=0.0)
    # Y_c < B_thrsh: add (B_thrsh - Y_c)^B_clip. At Y_c = B_thrsh derivative of (B_thrsh - y)^B_clip is -B_clip*0 = 0.
    delta = (B_thrsh - yc).clamp(min=0.0)
    add = torch.pow(delta + 1e-10, B_clip)
    out = torch.where(yc < B_thrsh, yc + add, yc)
    return out


def apca_contrast(
    txt_srgb: torch.Tensor,
    bg_srgb: torch.Tensor,
) -> torch.Tensor:
    """
    APCA Lc contrast between text (foreground) and background.
    txt_srgb, bg_srgb: sRGB in [0,1], shape (..., 3). Same leading shape.
    Returns Lc shape (...). Positive = dark text on light bg; negative = light on dark.
    Differentiable.
    """
    ys_txt = _srgb_to_ys(txt_srgb)
    ys_bg = _srgb_to_ys(bg_srgb)
    Y_txt = _soft_clip(ys_txt)
    Y_bg = _soft_clip(ys_bg)

    # S_apc: normal polarity (Y_bg > Y_txt): (Y_bg^Nbg - Y_txt^Ntx) * W_scale
    #         reverse (Y_bg < Y_txt): (Y_bg^Rbg - Y_txt^Rtx) * W_scale
    # Differentiable soft switch to avoid branch gradient break
    Y_txt_safe = Y_txt.clamp(min=1e-10)
    Y_bg_safe = Y_bg.clamp(min=1e-10)
    s_normal = (torch.pow(Y_bg_safe, Nbg) - torch.pow(Y_txt_safe, Ntx)) * W_scale
    s_reverse = (torch.pow(Y_bg_safe, Rbg) - torch.pow(Y_txt_safe, Rtx)) * W_scale
    # Blend by polarity: when Y_bg > Y_txt use normal, else reverse. Soft: sigmoid((Y_bg - Y_txt)*large)
    blend = torch.sigmoid((Y_bg - Y_txt) * 100.0)
    S_apc = blend * s_normal + (1.0 - blend) * s_reverse

    # Lc: 0 if |S_apc| < W_clamp; (S_apc - W_offset)*100 if S_apc > 0; (S_apc + W_offset)*100 if S_apc < 0
    abs_s = S_apc.abs()
    lc_pos = (S_apc - W_offset) * 100.0
    lc_neg = (S_apc + W_offset) * 100.0
    # Soft clamp: when abs_s < W_clamp output 0
    active = torch.sigmoid((abs_s - W_clamp) * 50.0)
    lc = active * (torch.where(S_apc > 0, lc_pos, lc_neg))
    return lc

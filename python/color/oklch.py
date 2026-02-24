"""
Differentiable OKLCH <-> sRGB in PyTorch.
OKLCH: L in [0,1], C >= 0, H in degrees [0, 360).
Input/output tensors support arbitrary batch dimensions; last dim is (L,C,H) or (R,G,B).
"""
from __future__ import annotations

import torch

# OKLab -> linear sRGB (Björn Ottosson 2021-01-25)
# l'_m'_s' from L,a,b then cube, then matrix to RGB
_OKLAB_TO_LINEAR_RGB = torch.tensor(
    [
        [+4.0767416621, -3.3077115913, +0.2309699292],
        [-1.2684380046, +2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, +1.7076147010],
    ],
    dtype=torch.float32,
)
# L,a,b -> l'_m'_s' (inverse of M2 applied to L,a,b gives l',m',s')
_OKLAB_TO_LMS_PRIME = torch.tensor(
    [
        [1.0, +0.3963377774, +0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=torch.float32,
)

# sRGB linear -> sRGB (gamma encoding); threshold 0.0031308
_SRGB_LIN_THRESH = 0.0031308
_SRGB_GAMMA = 1.0 / 2.4


def _linear_srgb_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Differentiable: linear sRGB in [0,1] -> sRGB in [0,1]."""
    # For gradient: use smooth blend near threshold to avoid undefined derivative
    # Standard: out = 12.92 * x if x <= 0.0031308 else 1.055 * x^(1/2.4) - 0.055
    linear = linear.clamp(min=1e-8)  # avoid 0^gamma
    encoded = torch.where(
        linear <= _SRGB_LIN_THRESH,
        12.92 * linear,
        1.055 * torch.pow(linear, _SRGB_GAMMA) - 0.055,
    )
    return encoded.clamp(0.0, 1.0)


def _srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """sRGB in [0,1] -> linear sRGB in [0,1]. Differentiable."""
    srgb = srgb.clamp(0.0, 1.0)
    return torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        torch.pow((srgb + 0.055) / 1.055, 2.4),
    )


def oklch_to_srgb(lch: torch.Tensor, lch_last_dim: bool = True) -> torch.Tensor:
    """
    OKLCH -> sRGB. Differentiable.
    lch: ... x 3 with (L, C, H). L in [0,1], C >= 0, H in degrees.
    Returns ... x 3 sRGB in [0, 1]. Batch dimensions preserved.
    """
    device = lch.device
    dtype = lch.dtype
    if _OKLAB_TO_LMS_PRIME.device != device or _OKLAB_TO_LMS_PRIME.dtype != dtype:
        M2inv = _OKLAB_TO_LMS_PRIME.to(device=device, dtype=dtype)
        M3 = _OKLAB_TO_LINEAR_RGB.to(device=device, dtype=dtype)
    else:
        M2inv = _OKLAB_TO_LMS_PRIME
        M3 = _OKLAB_TO_LINEAR_RGB

    L = lch[..., 0:1]  # ... x 1
    C = lch[..., 1:2].clamp(min=0.0)
    H_deg = lch[..., 2:3]
    H_rad = H_deg * (3.141592653589793 / 180.0)
    a = C * torch.cos(H_rad)
    b = C * torch.sin(H_rad)
    # ... x 3
    lab = torch.cat([L, a, b], dim=-1)
    # l'_m'_s' = M2inv @ lab
    lms_p = (lab.unsqueeze(-2) @ M2inv.T).squeeze(-2)  # ... x 3
    lms = lms_p ** 3
    linear_rgb = (lms.unsqueeze(-2) @ M3.T).squeeze(-2)
    linear_rgb = linear_rgb.clamp(min=0.0)
    srgb = _linear_srgb_to_srgb(linear_rgb)
    return srgb.clamp(0.0, 1.0)


def srgb_to_oklch(srgb: torch.Tensor) -> torch.Tensor:
    """
    sRGB in [0,1], last dim 3 -> OKLCH (L, C, H). L in [0,1], C >= 0, H in [0, 360).
    Batch dimensions preserved. Differentiable.
    linalg.inv 不支持 float16，故内部用 float32 计算再转回输入 dtype。
    """
    device = srgb.device
    out_dtype = srgb.dtype
    srgb = srgb.float()
    linear = _srgb_to_linear(srgb)
    M2inv = _OKLAB_TO_LMS_PRIME.to(device=device, dtype=torch.float32)
    M3 = _OKLAB_TO_LINEAR_RGB.to(device=device, dtype=torch.float32)
    # linear_rgb = lms @ M3.T  =>  lms = linear_rgb @ inv(M3.T)
    inv_M3T = torch.linalg.inv(M3.T)
    lms = linear @ inv_M3T
    lms = lms.clamp(min=1e-10)
    lms_p = torch.pow(lms, 1.0 / 3.0)
    # lms_p = lab @ M2inv.T  =>  lab = lms_p @ inv(M2inv.T)
    inv_M2invT = torch.linalg.inv(M2inv.T)
    lab = lms_p @ inv_M2invT
    L = lab[..., 0:1]
    a = lab[..., 1:2]
    b = lab[..., 2:3]
    C = torch.sqrt(a * a + b * b + 1e-12)
    H_rad = torch.atan2(b, a)
    H_deg = H_rad * (180.0 / 3.141592653589793)
    H_deg = torch.where(C < 1e-8, torch.zeros_like(H_deg), H_deg)
    out = torch.cat([L, C, (H_deg % 360.0)], dim=-1)
    return out.to(out_dtype)

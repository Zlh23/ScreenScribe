"""
Synthetic sampling of (bg_oklch, wantL, wantC, wantH, wantContrast, w_contrast, w_L, w_C, w_H).
No ground-truth foreground; used for training with weighted loss.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ContrastDataset(Dataset):
    """
    Each sample: bg_oklch (3), weights_4 (4), want_lchc_4 (4) = (wantL, wantC, wantH, wantContrast).
    Ranges: L/C/H in OKLCH bounds; wantContrast e.g. 15–90; weights non-negative (e.g. sum to 1).
    """

    def __init__(
        self,
        size: int,
        *,
        seed: int | None = None,
        l_range: tuple[float, float] = (0.0, 1.0),
        c_range: tuple[float, float] = (0.0, 0.4),
        h_range: tuple[float, float] = (0.0, 360.0),
        contrast_range: tuple[float, float] = (15.0, 90.0),
        prob_one_weight_zero: float = 0.3,
        prob_only_one_weight_nonzero: float = 0.3,
    ):
        self.size = size
        self.l_range = l_range
        self.c_range = c_range
        self.h_range = h_range
        self.contrast_range = contrast_range
        self.prob_one_zero = prob_one_weight_zero
        self.prob_only_one = prob_only_one_weight_nonzero
        if seed is not None:
            torch.manual_seed(seed)
        self._bg_oklch = None
        self._weights_4 = None
        self._want_lchc_4 = None
        self._populate()

    def _populate(self) -> None:
        # Background OKLCH: random in range (CPU for DataLoader)
        bg_l = torch.rand(self.size) * (self.l_range[1] - self.l_range[0]) + self.l_range[0]
        bg_c = torch.rand(self.size) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        bg_h = torch.rand(self.size) * (self.h_range[1] - self.h_range[0]) + self.h_range[0]
        self._bg_oklch = torch.stack([bg_l, bg_c, bg_h], dim=1)

        # Want L, C, H, Contrast
        want_l = torch.rand(self.size) * (self.l_range[1] - self.l_range[0]) + self.l_range[0]
        want_c = torch.rand(self.size) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        want_h = torch.rand(self.size) * (self.h_range[1] - self.h_range[0]) + self.h_range[0]
        want_contrast = (
            torch.rand(self.size) * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]
        )
        self._want_lchc_4 = torch.stack([want_l, want_c, want_h, want_contrast], dim=1)

        # Weights: non-negative, normalize to sum to 1 per sample
        # With prob_one_zero: one of the four weights is 0 (others renormalized)
        # With prob_only_one: exactly one weight is 1, the rest 0
        # Otherwise: all four random, sum to 1
        w = torch.rand(self.size, 4).clamp(min=1e-6)
        w = w / w.sum(dim=1, keepdim=True)

        u = torch.rand(self.size)
        one_zero = u < self.prob_one_zero
        only_one = (u >= self.prob_one_zero) & (u < self.prob_one_zero + self.prob_only_one)

        if only_one.any():
            j = torch.randint(0, 4, (self.size,))
            w_only = torch.zeros(self.size, 4)
            w_only[torch.arange(self.size), j] = 1.0
            w[only_one] = w_only[only_one]

        if one_zero.any():
            idx = torch.where(one_zero)[0]
            j = torch.randint(0, 4, (idx.shape[0],))
            w[idx, j] = 0
            s = w[idx].sum(dim=1, keepdim=True)
            s = s.clamp(min=1e-8)
            w[idx] = w[idx] / s

        self._weights_4 = w

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._bg_oklch[idx],
            self._weights_4[idx],
            self._want_lchc_4[idx],
        )

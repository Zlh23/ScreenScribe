"""
Load trained ColorContrastNet and predict foreground OKLCH + actual APCA contrast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

# Allow running as script from project root or python/
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPT_DIR))

from model.net import ColorContrastNet
from color.oklch import oklch_to_srgb
from color.apca import apca_contrast


def load_model(checkpoint_path: str | Path, device: torch.device | None = None) -> ColorContrastNet:
    """Load ColorContrastNet from checkpoint. Returns model in eval mode."""
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    hidden = ckpt["hidden"]
    state = ckpt["model_state_dict"]
    use_ln = "mlp.1.weight" in state and state["mlp.1.weight"].dim() == 1
    model = ColorContrastNet(hidden=hidden, use_layer_norm=use_ln)
    model.load_state_dict(state, strict=True)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def predict(
    model: ColorContrastNet,
    bg_oklch: torch.Tensor,
    want_l: float | torch.Tensor,
    want_c: float | torch.Tensor,
    want_h: float | torch.Tensor,
    want_contrast: float | torch.Tensor,
    w_contrast: float | torch.Tensor,
    w_l: float | torch.Tensor,
    w_c: float | torch.Tensor,
    w_h: float | torch.Tensor,
    *,
    return_contrast: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    One or batched prediction.
    bg_oklch: (3,) or (B, 3) OKLCH background.
    want_* / w_*: scalars or (B,) tensors. Will be broadcast to match batch.
    Returns dict with "l", "c", "h" (tensors or scalars) and optionally "contrast".
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    was_1d = bg_oklch.dim() == 1
    if was_1d:
        bg_oklch = bg_oklch.unsqueeze(0)
    bg_oklch = bg_oklch.to(device=device, dtype=torch.float32)
    B = bg_oklch.shape[0]

    def _to_tensor(x: float | torch.Tensor) -> torch.Tensor:
        if isinstance(x, (int, float)):
            return torch.full((B,), float(x), device=device, dtype=torch.float32)
        t = x.to(device=device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        return t

    want_lchc = torch.stack(
        [_to_tensor(want_l), _to_tensor(want_c), _to_tensor(want_h), _to_tensor(want_contrast)],
        dim=1,
    )
    weights_4 = torch.stack(
        [_to_tensor(w_contrast), _to_tensor(w_l), _to_tensor(w_c), _to_tensor(w_h)],
        dim=1,
    )

    with torch.no_grad():
        pred_lch = model(bg_oklch, weights_4, want_lchc)
        pred_srgb = oklch_to_srgb(pred_lch)
        bg_srgb = oklch_to_srgb(bg_oklch)
        actual_apca = apca_contrast(pred_srgb, bg_srgb)

    out: dict[str, Any] = {
        "l": pred_lch[..., 0].cpu(),
        "c": pred_lch[..., 1].cpu(),
        "h": pred_lch[..., 2].cpu(),
    }
    if return_contrast:
        out["contrast"] = actual_apca.cpu()

    if was_1d:
        out["l"] = out["l"].squeeze(0).item()
        out["c"] = out["c"].squeeze(0).item()
        out["h"] = out["h"].squeeze(0).item()
        if "contrast" in out:
            out["contrast"] = out["contrast"].squeeze(0).item()

    return out

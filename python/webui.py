"""
Simple Gradio Web UI to test the color contrast model.
Default: python webui.py  (等价于 --checkpoint_dir checkpoints --port 7860)
无 checkpoint 时也会启动，界面会提示先训练。
"""
from __future__ import annotations

import argparse
import contextlib
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np

from model.net import ColorContrastNet
from model.affine_net import AffineColorContrastNet
from color.oklch import oklch_to_srgb, srgb_to_oklch
from color.apca import apca_contrast
from loss import weighted_loss

# 全图逐像素推理的默认 batch 大小（GPU 上可适当调大）
DEFAULT_BATCH_SIZE = 65536


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """sRGB (R,G,B 0-255) -> hex for display."""
    return f"#{min(255, max(0, int(r))):02x}{min(255, max(0, int(g))):02x}{min(255, max(0, int(b))):02x}"


def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]
    if ckpt.get("model_type") == "AffineColorContrastNet":
        model = AffineColorContrastNet(hidden=ckpt["hidden"])
    else:
        hidden = ckpt["hidden"]
        use_ln = "mlp.1.weight" in state and state["mlp.1.weight"].dim() == 1
        model = ColorContrastNet(hidden=hidden, use_layer_norm=use_ln)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model


def _list_checkpoints(checkpoint_dir: str, default_relative: str) -> tuple[list[str], str]:
    """Return (choices, default). Choices are relative paths like 'ColorContrastNet/ckpt_final.pt'.
    Scans checkpoint_dir for class-name subdirs; if none, falls back to *.pt in root (legacy)."""
    root = Path(checkpoint_dir)
    if not root.exists():
        return ([], default_relative)

    def sort_key(item: tuple[str, str]) -> tuple[str, int, int]:
        rel_path = item[1]
        name = Path(rel_path).name
        if "final" in name:
            return (item[0], 0, 0)
        m = re.search(r"epoch_(\d+)", name)
        return (item[0], 1, -(int(m.group(1)) if m else 0))

    collected: list[tuple[str, str]] = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        pt_files = sorted(subdir.glob("*.pt"), key=lambda p: (0 if "final" in p.name else 1, -_epoch_num(p)))
        for p in pt_files:
            collected.append((subdir.name, f"{subdir.name}/{p.name}"))
    if not collected:
        pt_in_root = sorted(root.glob("*.pt"), key=lambda p: (0 if "final" in p.name else 1, -_epoch_num(p)))
        collected = [("", p.name) for p in pt_in_root]
    if not collected:
        return ([], default_relative)
    choices = [rel for _, rel in sorted(collected, key=sort_key)]
    default = default_relative if default_relative in choices else choices[0]
    return (choices, default)


def _epoch_num(p: Path) -> int:
    m = re.search(r"epoch_(\d+)", p.name)
    return int(m.group(1)) if m else 0


def _full_ckpt_path(checkpoint_dir: str, relative_choice: str) -> str:
    return str(Path(checkpoint_dir) / relative_choice)


def run_batch_image(
    model: ColorContrastNet,
    img: np.ndarray,
    want_l: float,
    want_c: float,
    want_h: float,
    want_contrast: float,
    w_contrast: float,
    w_l: float,
    w_c: float,
    w_h: float,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, float]:
    """
    对整图每个像素用当前背景色做一次前景预测，GPU 批量推理。
    img: (H, W, 3) numpy，0~255 或 0~1；返回 (H, W, 3) 0~255，以及耗时（秒）。
    """
    if img is None or img.size == 0:
        return None, 0.0
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 3 or img.shape[-1] != 3:
        return None, 0.0
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    H, W = img.shape[0], img.shape[1]
    N = H * W
    pixels = img.reshape(-1, 3)
    t0 = time.perf_counter()
    use_fp16 = device == "cuda" or (isinstance(device, str) and device.startswith("cuda"))
    dtype = torch.float16 if use_fp16 else torch.float32
    pixels_gpu = torch.from_numpy(pixels).to(device=device, dtype=dtype)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16) if use_fp16 else contextlib.nullcontext():
            bg_oklch = srgb_to_oklch(pixels_gpu)
            want_lchc_4 = torch.tensor(
                [[want_l, want_c, want_h, want_contrast]],
                device=device,
                dtype=dtype,
            ).expand(N, 4)
            weights_4 = torch.tensor(
                [[w_contrast, w_l, w_c, w_h]],
                device=device,
                dtype=dtype,
            ).expand(N, 4)
            pred_list = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                b_bg = bg_oklch[start:end]
                b_want = want_lchc_4[start:end]
                b_weights = weights_4[start:end]
                b_pred = model(b_bg, b_weights, b_want)
                pred_list.append(b_pred)
            pred_lch = torch.cat(pred_list, dim=0)
            pred_srgb = oklch_to_srgb(pred_lch)
    out = pred_srgb.float().cpu().numpy()
    out = np.clip(out, 0.0, 1.0).reshape(H, W, 3)
    elapsed = time.perf_counter() - t0
    return (out * 255.0).astype(np.uint8), elapsed


def create_ui(checkpoint_dir: str, default_checkpoint: str, device: str) -> "gr.Blocks":
    import gradio as gr

    choices, default_ckpt = _list_checkpoints(checkpoint_dir, default_checkpoint)
    if not choices:
        choices = ["（暂无 checkpoint，请先训练）"]
        default_ckpt = choices[0]
        initial_model = None
        initial_ckpt = ""
    else:
        initial_model = _load_model(_full_ckpt_path(checkpoint_dir, default_ckpt), device)
        initial_ckpt = default_ckpt

    def run(
        state,
        ckpt_choice,
        bg_r, bg_g, bg_b,
        want_l, want_c, want_h, want_contrast,
        w_contrast, w_l, w_c, w_h,
    ):
        model, last_ckpt = state
        if ckpt_choice and ckpt_choice != "（暂无 checkpoint，请先训练）" and Path(_full_ckpt_path(checkpoint_dir, ckpt_choice)).exists():
            if ckpt_choice != last_ckpt:
                model = _load_model(_full_ckpt_path(checkpoint_dir, ckpt_choice), device)
                state = (model, ckpt_choice)
            else:
                state = (model, last_ckpt)
        else:
            state = (model, last_ckpt)
            if model is None:
                return state, "**暂无 checkpoint**：请先运行 `python train.py`，将模型保存到 `checkpoints/ColorContrastNet/` 下。", ""
        if model is None:
            return state, "**暂无 checkpoint**：请先训练并保存模型。", ""

        bg_srgb_01 = torch.tensor(
            [[bg_r / 255.0, bg_g / 255.0, bg_b / 255.0]],
            dtype=torch.float32,
            device=device,
        )
        bg_oklch = srgb_to_oklch(bg_srgb_01)
        weights = torch.tensor([[w_contrast, w_l, w_c, w_h]], dtype=torch.float32, device=device)
        want = torch.tensor([[want_l, want_c, want_h, want_contrast]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_lch = model(bg_oklch, weights, want)
            pred_srgb = oklch_to_srgb(pred_lch)
            bg_srgb = bg_srgb_01
            contrast = apca_contrast(pred_srgb, bg_srgb).abs().item()
        pr, pg, pb = (pred_srgb[0].clamp(0, 1) * 255).round().int().tolist()
        bg_hex = _rgb_to_hex(bg_r, bg_g, bg_b)
        fg_hex = _rgb_to_hex(pr, pg, pb)
        ckpt_name = Path(ckpt_choice or last_ckpt).name
        text = (
            f"**模型** {ckpt_name}  \n"
            f"**预测前景 (sRGB)** R={pr} G={pg} B={pb}  "
            f"**APCA 对比度** {contrast:.1f}  "
            f"(目标 {want_contrast})"
        )
        html = f"""
        <div style="display:flex; gap:1rem; align-items:center; flex-wrap:wrap;">
          <div style="width:120px; height:80px; background:{bg_hex}; border:1px solid #333; border-radius:8px;"></div>
          <span>背景</span>
          <div style="width:120px; height:80px; background:{fg_hex}; border:1px solid #333; border-radius:8px;"></div>
          <span>前景</span>
          <div style="width:280px; height:80px; background:{bg_hex}; color:{fg_hex}; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.2rem;">
            示例文字 Aa
          </div>
          <span>预览</span>
        </div>
        """
        return state, text, html

    def run_loss_tab(
        state,
        ckpt_choice,
        bg_r, bg_g, bg_b,
        want_l, want_c, want_h, want_contrast,
        w_contrast, w_l, w_c, w_h,
        sim_l, sim_c, sim_h,
    ):
        bg_srgb_01 = torch.tensor(
            [[bg_r / 255.0, bg_g / 255.0, bg_b / 255.0]],
            dtype=torch.float32,
            device=device,
        )
        bg_oklch = srgb_to_oklch(bg_srgb_01)
        want_lch = torch.tensor([[want_l, want_c, want_h]], dtype=torch.float32, device=device)
        want_contrast_t = torch.tensor([want_contrast], dtype=torch.float32, device=device)
        weights_4 = torch.tensor([[w_contrast, w_l, w_c, w_h]], dtype=torch.float32, device=device)
        sim_lch = torch.tensor([[sim_l, sim_c, sim_h]], dtype=torch.float32, device=device)
        with torch.no_grad():
            sim_srgb = oklch_to_srgb(sim_lch)
            bg_srgb = bg_srgb_01
            sim_contrast = apca_contrast(sim_srgb, bg_srgb).abs()
            loss_sim = weighted_loss(sim_lch, want_lch, want_contrast_t, sim_contrast, weights_4)[0].item()
        out = f"**模拟输出** L={sim_l:.3f} C={sim_c:.3f} H={sim_h:.1f}° → **Loss = {loss_sim:.6f}**  \n"

        model, last_ckpt = state
        if ckpt_choice and ckpt_choice != "（暂无 checkpoint，请先训练）" and Path(_full_ckpt_path(checkpoint_dir, ckpt_choice)).exists():
            if ckpt_choice != last_ckpt:
                model = _load_model(_full_ckpt_path(checkpoint_dir, ckpt_choice), device)
                state = (model, ckpt_choice)
            if model is not None:
                with torch.no_grad():
                    pred_lch = model(bg_oklch, weights_4, torch.cat([want_lch, want_contrast_t.unsqueeze(1)], dim=1))
                    pred_srgb = oklch_to_srgb(pred_lch)
                    pred_contrast = apca_contrast(pred_srgb, bg_srgb).abs()
                    loss_model = weighted_loss(pred_lch, want_lch, want_contrast_t, pred_contrast, weights_4)[0].item()
                pred_contrast_val = apca_contrast(pred_srgb, bg_srgb).abs().item()
                pr, pg, pb = (pred_srgb[0].clamp(0, 1) * 255).round().int().tolist()
                out += f"**模型预测 (sRGB)** R={pr} G={pg} B={pb} (APCA {pred_contrast_val:.1f}) → **Loss = {loss_model:.6f}**"
            else:
                out += "**模型预测**：未加载模型。"
        else:
            out += "**模型预测**：请先在「预测」页选择有效 checkpoint 或先训练保存模型。"
        return state, out

    with gr.Blocks(title="Color Contrast 测试", css=".block { max-width: 900px; }") as demo:
        gr.Markdown("## 前景色预测：背景 + 期望 LCH + 目标对比度 + 权重")
        state = gr.State((initial_model, initial_ckpt))
        with gr.Tabs():
            with gr.TabItem("预测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ckpt_dropdown = gr.Dropdown(
                            choices=choices,
                            value=default_ckpt,
                            label="测试用 checkpoint",
                            allow_custom_value=False,
                        )
                        gr.Markdown("### 背景 (sRGB)")
                        bg_r = gr.Slider(0, 255, value=235, step=1, label="R")
                        bg_g = gr.Slider(0, 255, value=235, step=1, label="G")
                        bg_b = gr.Slider(0, 255, value=235, step=1, label="B")
                        gr.Markdown("### 期望 (want)")
                        want_l = gr.Slider(0, 1, value=0.25, step=0.01, label="want L")
                        want_c = gr.Slider(0, 0.4, value=0.12, step=0.01, label="want C")
                        want_h = gr.Slider(0, 360, value=200, step=1, label="want H °")
                        want_contrast = gr.Slider(15, 90, value=60, step=1, label="目标对比度 (APCA)")
                        gr.Markdown("### 误差权重 (偏好)")
                        w_contrast = gr.Slider(0, 1, value=0.25, step=0.05, label="w_contrast")
                        w_l_s = gr.Slider(0, 1, value=0.25, step=0.05, label="w_L")
                        w_c_s = gr.Slider(0, 1, value=0.25, step=0.05, label="w_C")
                        w_h = gr.Slider(0, 1, value=0.25, step=0.05, label="w_H")
                        btn = gr.Button("预测", variant="primary")
                    with gr.Column(scale=1):
                        out_text = gr.Markdown("")
                        out_html = gr.HTML("")
                inputs_list = [
                    state,
                    ckpt_dropdown,
                    bg_r, bg_g, bg_b,
                    want_l, want_c, want_h, want_contrast,
                    w_contrast, w_l_s, w_c_s, w_h,
                ]
                outputs_list = [state, out_text, out_html]
                trigger_components = [
                    ckpt_dropdown,
                    bg_r, bg_g, bg_b,
                    want_l, want_c, want_h, want_contrast,
                    w_contrast, w_l_s, w_c_s, w_h,
                ]
                for comp in trigger_components:
                    comp.change(fn=run, inputs=inputs_list, outputs=outputs_list)
                btn.click(fn=run, inputs=inputs_list, outputs=outputs_list)
                demo.load(fn=run, inputs=inputs_list, outputs=outputs_list)
            with gr.TabItem("Loss 测试"):
                gr.Markdown("输入相同参数 + **模拟前景 (L,C,H)**，查看该输出的 loss；并对比当前模型的预测及其 loss。")
                with gr.Row():
                    with gr.Column(scale=1):
                        ckpt_dropdown_loss = gr.Dropdown(
                            choices=choices,
                            value=default_ckpt,
                            label="checkpoint",
                            allow_custom_value=False,
                        )
                        gr.Markdown("### 背景 (sRGB)")
                        bg_r_loss = gr.Slider(0, 255, value=235, step=1, label="R")
                        bg_g_loss = gr.Slider(0, 255, value=235, step=1, label="G")
                        bg_b_loss = gr.Slider(0, 255, value=235, step=1, label="B")
                        gr.Markdown("### 期望 (want)")
                        want_l_loss = gr.Slider(0, 1, value=0.25, step=0.01, label="want L")
                        want_c_loss = gr.Slider(0, 0.4, value=0.12, step=0.01, label="want C")
                        want_h_loss = gr.Slider(0, 360, value=200, step=1, label="want H °")
                        want_contrast_loss = gr.Slider(15, 90, value=60, step=1, label="目标对比度 (APCA)")
                        gr.Markdown("### 误差权重 (偏好)")
                        w_contrast_loss = gr.Slider(0, 1, value=0.25, step=0.05, label="w_contrast")
                        w_l_loss = gr.Slider(0, 1, value=0.25, step=0.05, label="w_L")
                        w_c_loss = gr.Slider(0, 1, value=0.25, step=0.05, label="w_C")
                        w_h_loss = gr.Slider(0, 1, value=0.25, step=0.05, label="w_H")
                        gr.Markdown("### 模拟前景 (OKLCH)")
                        sim_l = gr.Slider(0, 1, value=0.3, step=0.01, label="模拟 L")
                        sim_c = gr.Slider(0, 0.4, value=0.1, step=0.01, label="模拟 C")
                        sim_h = gr.Slider(0, 360, value=200, step=1, label="模拟 H °")
                        btn_loss = gr.Button("计算 Loss", variant="primary")
                    with gr.Column(scale=1):
                        out_loss_text = gr.Markdown("")
                loss_inputs = [
                    state,
                    ckpt_dropdown_loss,
                    bg_r_loss, bg_g_loss, bg_b_loss,
                    want_l_loss, want_c_loss, want_h_loss, want_contrast_loss,
                    w_contrast_loss, w_l_loss, w_c_loss, w_h_loss,
                    sim_l, sim_c, sim_h,
                ]
                btn_loss.click(fn=run_loss_tab, inputs=loss_inputs, outputs=[state, out_loss_text])
                for comp in [ckpt_dropdown_loss, bg_r_loss, bg_g_loss, bg_b_loss, want_l_loss, want_c_loss, want_h_loss, want_contrast_loss, w_contrast_loss, w_l_loss, w_c_loss, w_h_loss, sim_l, sim_c, sim_h]:
                    comp.change(fn=run_loss_tab, inputs=loss_inputs, outputs=[state, out_loss_text])
            with gr.TabItem("Test"):
                gr.Markdown("上传或**粘贴图片**（如从剪切板），对每个像素用该像素颜色作为背景做前景色预测，输出整图与耗时（GPU 批量推理）。")
                with gr.Row():
                    with gr.Column(scale=1):
                        ckpt_test = gr.Dropdown(
                            choices=choices,
                            value=default_ckpt,
                            label="checkpoint",
                            allow_custom_value=False,
                        )
                        img_in = gr.Image(
                            label="输入图片（可粘贴）",
                            type="numpy",
                            height=300,
                        )
                        gr.Markdown("### 期望与权重（全图统一）")
                        want_l_t = gr.Slider(0, 1, value=0.25, step=0.01, label="want L")
                        want_c_t = gr.Slider(0, 0.4, value=0.12, step=0.01, label="want C")
                        want_h_t = gr.Slider(0, 360, value=200, step=1, label="want H °")
                        want_contrast_t = gr.Slider(15, 90, value=60, step=1, label="目标对比度 (APCA)")
                        w_contrast_t = gr.Slider(0, 1, value=0.25, step=0.05, label="w_contrast")
                        w_l_t = gr.Slider(0, 1, value=0.25, step=0.05, label="w_L")
                        w_c_t = gr.Slider(0, 1, value=0.25, step=0.05, label="w_C")
                        w_h_t = gr.Slider(0, 1, value=0.25, step=0.05, label="w_H")
                        btn_test = gr.Button("处理", variant="primary")
                    with gr.Column(scale=1):
                        img_out = gr.Image(label="输出图片", height=300)
                        out_time = gr.Markdown("")
                test_inputs = [
                    state,
                    ckpt_test,
                    img_in,
                    want_l_t, want_c_t, want_h_t, want_contrast_t,
                    w_contrast_t, w_l_t, w_c_t, w_h_t,
                ]

                def run_test(
                    state_val,
                    ckpt_choice,
                    image,
                    want_l, want_c, want_h, want_contrast,
                    w_contrast, w_l, w_c, w_h,
                ):
                    model_obj, last_ckpt = state_val
                    if ckpt_choice and ckpt_choice != "（暂无 checkpoint，请先训练）" and Path(_full_ckpt_path(checkpoint_dir, ckpt_choice)).exists():
                        if ckpt_choice != last_ckpt:
                            model_obj = _load_model(_full_ckpt_path(checkpoint_dir, ckpt_choice), device)
                            state_val = (model_obj, ckpt_choice)
                    if model_obj is None:
                        return state_val, None, "**暂无 checkpoint**：请先训练并保存模型后再使用 Test。"
                    out_img, elapsed = run_batch_image(
                        model_obj,
                        image,
                        want_l, want_c, want_h, want_contrast,
                        w_contrast, w_l, w_c, w_h,
                        device,
                    )
                    if out_img is None:
                        return state_val, None, "未收到有效图片，请上传或粘贴后再点「处理」。"
                    time_text = f"**处理耗时**：{elapsed:.3f} 秒（像素数 {out_img.shape[0] * out_img.shape[1]:,}，GPU 批量推理）"
                    return state_val, out_img, time_text

                btn_test.click(
                    fn=run_test,
                    inputs=test_inputs,
                    outputs=[state, img_out, out_time],
                )

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Color Contrast Web UI（默认 checkpoint_dir=checkpoints, port=7860）")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="扫描 .pt 的根目录，默认 checkpoints")
    p.add_argument("--checkpoint", type=str, default="ColorContrastNet/ckpt_final.pt", help="默认选中的 checkpoint 相对路径")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备，Test 页批量推理会使用此设备")
    p.add_argument("--port", type=int, default=7860, help="服务端口，默认 7860")
    p.add_argument("--share", action="store_true", help="创建公网链接")
    args = p.parse_args()
    import gradio as gr
    demo = create_ui(args.checkpoint_dir, args.checkpoint, args.device)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

# ML Color Contrast

根据 [DESIGN_ML_COLOR_CONTRAST.md](DESIGN_ML_COLOR_CONTRAST.md) 实现：背景 + 期望 LCH + 目标对比度 + 误差权 → 前景 OKLCH。

## 安装

使用清华 PyPI 源加速（推荐）：

**Windows（PowerShell 或 cmd）：**

```powershell
cd python
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Linux / macOS：**

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Windows 若 venv 一直创建不出目录**：双击运行 `setup_venv.bat` 看输出（会用绝对路径建 `venv` 并列出是否成功）。若仍失败，可**不建 venv**：双击 `run_without_venv.bat`，用当前 Python 直接装依赖，之后在该目录下用 `python train.py` / `python webui.py` 即可。

## 训练

使用 **PyTorch Lightning**：AdamW、CosineAnnealing、梯度裁剪；模型带 **output_bias** 与 **LayerNorm** 减轻坍缩到黑。

Checkpoint 按模型类名存到子目录。**先激活上述虚拟环境**，然后：

```bash
python train.py --epochs 150 --save_dir checkpoints --save_every 10
```

- 保存到 `checkpoints/ColorContrastNet/ckpt_epoch_*.pt` 与 `ckpt_final.pt`。
- 日志到 `logs/color_contrast/`（TensorBoard + CSV），含 `train_loss`、`train_pred_L_mean`、`train_pred_C_mean`，便于观察是否坍缩。
- 默认使用 `cuda`（单卡）；多卡时可用 `--device cuda:1` 等。
- Windows 下 DataLoader 默认 `--num_workers 0`，避免多进程问题；Linux 可试 `--num_workers 4` 等加速。

## Web UI 测试

激活虚拟环境后：

```bash
python webui.py --checkpoint_dir checkpoints --port 7860
```

浏览器打开 http://localhost:7860。

- **预测**：选 checkpoint、背景/期望/权重，实时看预测前景与 APCA 对比度（界面与训练均使用对比度幅值 |Lc|，不区分正负极性）。
- **Loss 测试**：相同输入 + 一组「模拟前景 (L,C,H)」，查看该输出的 loss，并与当前模型预测的 loss 对比，便于调试 loss 设计。

## 推理

```python
from inference import load_model, predict

model = load_model("checkpoints/ColorContrastNet/ckpt_final.pt")
result = predict(
    model,
    bg_oklch=torch.tensor([0.9, 0.02, 180.0]),  # 背景 OKLCH
    want_l=0.3, want_c=0.15, want_h=200.0, want_contrast=60.0,
    w_contrast=0.5, w_l=0.2, w_c=0.2, w_h=0.1,
    return_contrast=True,
)
# result["l"], result["c"], result["h"], result["contrast"]
```

## 结构

- `color/`：可微 OKLCH→sRGB、APCA 对比度
- `model/`：两路网络 M1@M2 → (L,C,H)
- `data/`：无标签合成采样
- `loss.py`：加权损失
- `train.py`：训练入口
- `inference.py`：加载模型与预测 API

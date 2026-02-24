# 按对比度与期望 L/C/H 反算前景色：ML 方案设计

本文档记录在 `python/` 中实现的 ML 方案设计思路，与 `COLOR_FOR_CONTRAST_ML.md` 中的问题定义一致，并纳入「误差权重作为输入」与「两路网络 + 一次计算」的结构。

---

## 1. 问题与目标

- **输入**
  - 背景色 `bgColor`（OKLCH 或 RGB，如 3 维）
  - 目标 APCA 对比度 `wantContrast`
  - 期望亮度/饱和度/色相 `wantL`, `wantC`, `wantH`（OKLCH）
  - **误差偏好（权）**：`w_contrast`, `w_L`, `w_C`, `w_H`（用户用其控制「对比度 vs L vs C vs H」的侧重）

- **输出**
  - 前景色 OKLCH `(L, C, H)`；可选返回该前景与背景的实际 APCA 对比度（用于展示/校验）

- **要求**
  - 纯函数、与 ColorTheme 解耦；不做 rgb/hex 等格式转换。

---

## 2. 训练方式：无 Oracle，直接加权损失

- **不**使用网格搜索 + softmin 的 oracle 生成「真值前景」。
- 数据只需采样 **(bgColor, wantL, wantC, wantH, wantContrast, w_contrast, w_L, w_C, w_H)**，无需标签前景。
- 模型前向得到预测前景 `(L, C, H)`，在 Python 中用**可微的 APCA** 计算「预测前景与背景」的 APCA 对比度。
- **损失**（每条样本用该样本的权）：
  - `L_contrast = | APCA(预测前景, 背景) - wantContrast |`（或平方 / smooth L1）
  - `L_L = (pred_L - wantL)²`，`L_C = (pred_C - wantC)²`
  - `L_H` = 圆周损失（如 `1 - cos((pred_H - wantH) * π/180)`）
  - **加权总损失**：`L = w_contrast * L_contrast + w_L * L_L + w_C * L_C + w_H * L_H`

实现时需保证 APCA 在 PyTorch 中可微，以便梯度从损失回传到预测 (L,C,H) 再回传到网络参数。

---

## 3. 网络结构：两路 + 一次计算

### 3.1 整体数据流

```
(bgColor, w_contrast, w_L, w_C, w_H)  →  大网络（需训练）  →  M1
(wantL, wantC, wantH, wantContrast)   →  小网络（需训练）  →  M2
                                                                  ↘
                                                           一次计算（无额外参数）  →  (L, C, H)
                                                                  ↗
```

- **大网络**：只依赖「背景 + 误差偏好」，输出与背景和权相关的表示 **M1**。
- **小网络**：只依赖「期望 LCH + 目标对比度」，输出 **M2**（embedding）。
- **一次计算**：由 M1 和 M2 直接得到前景 (L, C, H)，中间不再增加可训练层。

### 3.2 维度约定（推荐）

| 部分     | 输入 | 输出 |
|----------|------|------|
| 大网络   | `bgColor`（如 3 维）+ `w_contrast, w_L, w_C, w_H`（4 维） | **M1**：形状 `(3, k)` 的矩阵 |
| 小网络   | `wantL, wantC, wantH, wantContrast`（4 维） | **M2**：长度为 `k` 的向量 |
| 一次计算 | M1, M2 | `(L, C, H)^T = M1 @ M2`，得到 (3,) 即 L, C, H |

- `k` 为超参（如 32、64）。
- 大网络输出可先为一维向量，再 reshape 为 `(3, k)` 以得到 M1。
- 得到 (L, C, H) 后：L、C 做 clamp 到合法范围，H 做 mod 360 或等价周期化。

### 3.3 设计意图

- 「背景 + 权」决定**如何把期望映射到颜色**（M1 相当于在该背景与偏好下的 3×k 映射矩阵）。
- 「期望 LCH + 目标对比度」被编码成 M2，与 M1 做一次矩阵乘即得到前景，计算简单、可解释。

---

## 4. 推理时的用法

用户传入：
- `bgColor`, `wantContrast`, `wantL`, `wantC`, `wantH`
- `w_contrast`, `w_L`, `w_C`, `w_H`

模型前向：大网络 → M1，小网络 → M2，`(L,C,H) = M1 @ M2` + 后处理，返回 OKLCH 前景；可选再算一次 APCA(前景, 背景) 作为实际对比度返回。

---

## 5. 小结表

| 项目       | 说明 |
|------------|------|
| 输入       | bgColor, wantL, wantC, wantH, wantContrast, **w_contrast, w_L, w_C, w_H** |
| 输出       | 前景 OKLCH (L, C, H)，可选实际对比度 |
| 训练数据   | 仅采样上述输入，无标签前景；损失由预测前景 + APCA + 目标 LCH/对比度 加权得到 |
| 损失       | `L = w_contrast*L_contrast + w_L*L_L + w_C*L_C + w_H*L_H`，权来自输入 |
| 大网络     | bgColor + 四个权 → M1 (3×k) |
| 小网络     | wantL, wantC, wantH, wantContrast → M2 (k) |
| 一次计算   | (L, C, H) = M1 @ M2，再 clamp / H 周期化 |

与 `COLOR_FOR_CONTRAST_ML.md` 中前因后果与原始 ML 设想一致；本设计在此基础上固定了「权作为输入」与「两路 + M1@M2 一次得 LCH」的结构，便于在 `python/` 中实现。

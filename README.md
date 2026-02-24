# 屏上书 ScreenScribe

全屏透明画布 + 画笔的桌面应用，基于 Tauri 2 + React。

## 功能

- 无边框、全屏、置顶、透明窗口
- 全屏画布，指针绘制（支持压感）
- Debug 栏：指针坐标、压感、画布尺寸、状态、FPS
- 清空画布（按钮或 Ctrl+Shift+C）
- 保存为 PNG（保存对话框 + 写入文件）
- 快捷键：Ctrl+Shift+Q 退出、Ctrl+Shift+H 隐藏、Ctrl+Shift+D 切换 Debug

## 环境要求

- **Node.js** 18+
- **Rust**（用于 Tauri 构建）
- **Windows**：需已安装 [WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)（Windows 11 通常已带；Windows 10 请确认或从上述链接安装）

## 从 WSL 迁到 Windows

- **清理构建产物**：在 PowerShell 中执行 `.\clean-for-windows.ps1`（在 WSL 里则用 `bash clean-for-windows.sh`）。
- **缺少 `src-tauri`**：若目录不存在，需从原 WSL 项目拷贝整个 `src-tauri` 文件夹，否则 `npm run tauri dev` 会报错。
- **打包 CI 报错**：在 PowerShell 中先执行 `$env:CI=''` 再运行 `npm run tauri build`。

## 开发

```bash
npm install
npm run tauri dev
```

## 打包

```bash
npm run tauri build
```

产物在 `src-tauri/target/release/`（或安装包在 `src-tauri/target/release/bundle/`）。

## 计划

- 阶段三（DXGI 抓屏 + MJPEG HTTP 流）未在本仓库实现，参见 `PLAN.md`。

# cpp-dwm：DWM 注入画线试验项目

在 DWM 合成管线的「最后一步」画线，使线条显示在所有窗口之上，且不创建全屏覆盖窗口（避免被环境检测导致黑屏）。

---

## 目标

- **输入**：要画的线段列表（坐标、线宽、颜色）。
- **输出**：这些线在屏幕上可见，且始终处于「最顶层」（DWM 合成后再画，相当于后处理）。
- **约束**：不创建全屏/透明覆盖窗口；通过注入 DWM 进程并 Hook 其绘制流程实现。

---

## 要做的内容

### 1. 注入器（Injector）

- 独立 exe，运行后把「我们的 DLL」注入到 **dwm.exe** 进程。
- 常见做法：`OpenProcess` → 在目标进程里分配内存、写入 DLL 路径 → `CreateRemoteThread` 调 `LoadLibraryW("我们的.dll")`（或使用 `NtCreateThreadEx` 等）。
- 需要**管理员权限**。
- 可选：注入成功后通过某种 IPC（命名管道、共享内存等）与主应用通信，或本阶段只做「注入进去能跑」的验证。

### 2. 注入用 DLL（Hook DLL）

- 被注入到 dwm.exe 的 DLL。
- **DllMain**：加载时执行一次。
  - 做 **pattern scan**：在 `d2d1.dll`（或 DWM 使用的相关模块）里找到 DWM 每帧用于合成的内部渲染函数（参考现有项目的 pattern，如与 `DrawingContext` 虚表相关）。
  - 对该函数做 **Hook**（如用 MinHook / Detours）：原函数前插我们的函数。
- **我们的 Hook 函数**：
  - 先调用 **原始函数**（让 DWM 正常画完桌面和所有窗口）。
  - 再在**同一帧的渲染目标**上画我们维护的**线段列表**（GDI 或 Direct2D，视 DWM 当前用的 API 而定）。
  - 然后返回，让 DWM 继续 Present。
- **线段数据来源**（后续与主应用对接时）：
  - 可由主应用通过 IPC 把「线段列表」传给注入器或 DLL（例如共享内存 + 命名事件）；DLL 内维护该列表，在每帧 Hook 里按列表绘制。

### 3. 本阶段验证目标（不要求一次做完）

- 能稳定把 DLL 注入进 dwm.exe，且 DWM 不崩溃。
- 能在 DLL 内完成 pattern scan，找到目标函数并成功 Hook（可先做空 Hook：只调原函数、不画线，确认不崩）。
- 在 Hook 里能拿到可用的渲染目标（DC 或 D2D 表面），并画出一条简单的线（例如固定坐标的一条线），确认屏幕上能看到且「盖在最上面」。

### 4. 后续（与 ScreenScribe 主应用对接）

- 定义 IPC 协议：主应用（Tauri）把「画线请求」发给本模块；本模块更新内部线段列表，DWM 每帧自动重画。
- 可选：提供「清空线条」等命令。

---

## 技术要点

- **语言**：C++（Win32，Visual Studio 推荐）。
- **Hook 库**：MinHook / Microsoft Detours / polyhook2 等选一。
- **Pattern**：参考 [DWM_Hook_Solution](https://github.com/0xNever/DWM_Hook_Solution) 等项目的 pattern；随 Windows 版本可能需调整。
- **风险**：Hook 错误可能导致 DWM 崩溃；系统通常会自动重启 DWM。建议先在虚拟机中测试。

---

## 目录建议（后续实现时可按此拆分）

- `injector/`：注入器工程（exe）。
- `hook_dll/`：被注入的 DLL 工程（pattern scan + Hook + 画线）。
- `README.md`：本说明。

---

## 参考

- [0xNever/DWM_Hook_Solution](https://github.com/0xNever/DWM_Hook_Solution)
- [rlybasic/DWM_Hook](https://github.com/rlybasic/DWM_Hook)
- [HadesW/dwm-overlay](https://github.com/HadesW/dwm-overlay)

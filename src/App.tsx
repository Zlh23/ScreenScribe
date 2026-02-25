import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { save } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";
import { register, unregister } from "@tauri-apps/plugin-global-shortcut";
import { useEffect, useRef, useState } from "react";
// 线条：内部 + 外描边（先画外描边再画内部）
const LINE_INNER_WIDTH = 2;
const LINE_OUTER_RING = 2; // 外描边在内部两侧各延伸的像素
const LINE_INNER_COLOR = "#fff";
const LINE_OUTER_COLOR = "#000";


type DrawState = "idle" | "drawing";

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: window.innerWidth, h: window.innerHeight });
  const [drawState, setDrawState] = useState<DrawState>("idle");
  const [pointer, setPointer] = useState({ x: 0, y: 0, pressure: 0 });
  const [debugVisible, setDebugVisible] = useState(true);
  const [fps, setFps] = useState(0);
  const [captureB64, setCaptureB64] = useState<string | null>(null); // 截屏（排除本窗）PNG base64
  const [showCaptureBackground, setShowCaptureBackground] = useState(false); // 是否用截屏当背景
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [transparentBg] = useState(true); // 常驻透明
  const [ignoreCursorEvents, setIgnoreCursorEvents] = useState(false); // 鼠标穿透（Cs+T 切换）
  const ignoreCursorEventsRef = useRef(false);
  const lastPointRef = useRef({ x: 0, y: 0 });
  const fpsRef = useRef({ frames: 0, last: performance.now() });
  const clearCanvasRef = useRef<() => void>(() => {});
  const saveCanvasRef = useRef<() => Promise<void>>(async () => {});
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  const canvasW = Math.round(size.w * dpr);
  const canvasH = Math.round(size.h * dpr);

  // 挂载时让根节点获得焦点，热键才能生效
  useEffect(() => {
    rootRef.current?.focus();
  }, []);

  // 窗口/画布尺寸：resize 时更新
  useEffect(() => {
    const onResize = () => setSize({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // 快捷键：Ctrl+Shift+Q 退出，Ctrl+Shift+H 最小化，Ctrl+Shift+D 切换 Debug，Ctrl+Shift+R 清空
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    invoke("clear_screen_lines").catch(() => {});
  };

  const saveCanvas = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    setSaveStatus("…");
    try {
      const path = await save({
        filters: [{ name: "PNG", extensions: ["png"] }],
      });
      if (!path) {
        setSaveStatus(null);
        return;
      }
      const blob = await new Promise<Blob | null>((ok) =>
        canvas.toBlob(ok, "image/png")
      );
      if (!blob) throw new Error("toBlob failed");
      const buf = await blob.arrayBuffer();
      await writeFile(path, new Uint8Array(buf));
      setSaveStatus("已保存");
      setTimeout(() => setSaveStatus(null), 2000);
    } catch (e) {
      const msg = String(e);
      setSaveStatus("失败: " + msg);
      console.error("[保存失败]", e);
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  useEffect(() => {
    clearCanvasRef.current = clearCanvas;
    saveCanvasRef.current = saveCanvas;
  });

  useEffect(() => {
    document.body.style.background = transparentBg ? "transparent" : "";
    return () => { document.body.style.background = ""; };
  }, [transparentBg]);

  useEffect(() => {
    ignoreCursorEventsRef.current = ignoreCursorEvents;
  }, [ignoreCursorEvents]);

  // 全局快捷键（Cs+H 最小/还原 Cs+T 穿透 Cs+R 重画 Cs+U 保存 Cs+O 置顶）
  useEffect(() => {
    const shortcuts = [
      { shortcut: "CommandOrControl+Shift+H", handler: async () => {
        const w = getCurrentWindow();
        const min = await w.isMinimized();
        if (min) w.unminimize(); else w.minimize();
      }},
      { shortcut: "CommandOrControl+Shift+T", handler: async () => {
        const w = getCurrentWindow();
        const next = !ignoreCursorEventsRef.current;
        await w.setIgnoreCursorEvents(next);
        setIgnoreCursorEvents(next);
      }},
      { shortcut: "CommandOrControl+Shift+R", handler: () => clearCanvasRef.current() },
      { shortcut: "CommandOrControl+Shift+U", handler: () => saveCanvasRef.current() },
      { shortcut: "CommandOrControl+Shift+O", handler: async () => {
        const w = getCurrentWindow();
        const on = await w.isAlwaysOnTop();
        await w.setAlwaysOnTop(!on);
      }},
    ];
    let cancelled = false;
    (async () => {
      for (const { shortcut, handler } of shortcuts) {
        if (cancelled) return;
        await unregister(shortcut).catch(() => {});
        if (cancelled) return;
        await register(shortcut, (event) => {
          if (event.state === "Pressed") handler();
        });
      }
    })();
    return () => {
      cancelled = true;
      shortcuts.forEach(({ shortcut }) => {
        void unregister(shortcut).catch(() => {});
      });
    };
  }, []);

  // 仅窗口有焦点时：Q 退出、D 切换 Debug
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!e.ctrlKey || !e.shiftKey) return;
      if (e.key === "Q") {
        e.preventDefault();
        getCurrentWindow().close();
      } else if (e.key === "D") {
        e.preventDefault();
        setDebugVisible((v) => !v);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // FPS 计数（requestAnimationFrame）
  useEffect(() => {
    let rafId: number;
    const tick = () => {
      fpsRef.current.frames++;
      const now = performance.now();
      const elapsed = now - fpsRef.current.last;
      if (elapsed >= 1000) {
        setFps(Math.round((fpsRef.current.frames * 1000) / elapsed));
        fpsRef.current.frames = 0;
        fpsRef.current.last = now;
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const x = e.clientX;
    const y = e.clientY;
    ctx.beginPath();
    ctx.moveTo(x * dpr, y * dpr);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    setDrawState("drawing");
    lastPointRef.current = { x: e.screenX, y: e.screenY };
    setPointer({ x: e.clientX, y: e.clientY, pressure: e.pressure ?? 0 });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (drawState === "drawing") {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.lineTo(e.clientX * dpr, e.clientY * dpr);
        }
      }
      lastPointRef.current = { x: e.screenX, y: e.screenY };
    } else {
      setPointer({ x: e.clientX, y: e.clientY, pressure: e.pressure ?? 0 });
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (drawState === "drawing") {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.lineTo(e.clientX * dpr, e.clientY * dpr);
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.lineWidth = LINE_INNER_WIDTH + 2 * LINE_OUTER_RING;
          ctx.strokeStyle = LINE_OUTER_COLOR;
          ctx.stroke();
          ctx.lineWidth = LINE_INNER_WIDTH;
          ctx.strokeStyle = LINE_INNER_COLOR;
          ctx.stroke();
        }
      }
    }
    setDrawState("idle");
  };

  const onPointerCancel = () => setDrawState("idle");

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      style={{
        width: "100vw",
        height: "100vh",
        position: "relative",
        overflow: "hidden",
        outline: "none",
        background: transparentBg ? "transparent" : "#fff",
      }}
    >
      {/* 画布层：仅用于接收笔触，线画在屏幕 DC 上（反色效果） */}
      <canvas
        ref={canvasRef}
        width={canvasW}
        height={canvasH}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          width: "100%",
          height: "100%",
          zIndex: 1,
          touchAction: "none",
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        onPointerCancel={onPointerCancel}
      />

      {/* Debug 栏：右上角，半透明，z-index 高于画布 */}
      {debugVisible && (
        <div
          style={{
            position: "fixed",
            top: 0,
            right: 0,
            padding: "8px 12px",
            background: "rgba(0,0,0,0.6)",
            color: "#eee",
            fontSize: 12,
            fontFamily: "monospace",
            zIndex: 10,
          }}
        >
          <div style={{ pointerEvents: "none" }}>
            指针: ({pointer.x}, {pointer.y}) pressure={pointer.pressure}
          </div>
          <div style={{ pointerEvents: "none" }}>画布/窗口: {size.w}×{size.h}</div>
          <div style={{ pointerEvents: "none" }}>状态: {drawState}</div>
          <div style={{ pointerEvents: "none" }}>穿透: {ignoreCursorEvents ? "开" : "关"}</div>
          <div style={{ pointerEvents: "none" }}>FPS: {fps}</div>
          <div style={{ marginTop: 6, display: "flex", gap: 8 }}>
            <button
              type="button"
              onClick={clearCanvas}
              style={{
                padding: "4px 8px",
                cursor: "pointer",
                background: "#444",
                color: "#eee",
                border: "1px solid #666",
                borderRadius: 4,
              }}
            >
              清空
            </button>
            <button
              type="button"
              onClick={saveCanvas}
              disabled={!!saveStatus && saveStatus !== "已保存"}
              style={{
                padding: "4px 8px",
                cursor: saveStatus ? "default" : "pointer",
                background: "#444",
                color: "#eee",
                border: "1px solid #666",
                borderRadius: 4,
              }}
            >
              {saveStatus ?? "保存"}
            </button>
            <button
              type="button"
              onClick={async () => {
                try {
                  const b64 = await invoke<string>("capture_screen_excluding_self");
                  setCaptureB64(b64);
                } catch (e) {
                  console.error(e);
                }
              }}
              style={{
                padding: "4px 8px",
                cursor: "pointer",
                background: "#444",
                color: "#eee",
                border: "1px solid #666",
                borderRadius: 4,
              }}
            >
              截屏(排除本窗)
            </button>
            <button
              type="button"
              onClick={() => setShowCaptureBackground((v) => !v)}
              disabled={!captureB64}
              title={captureB64 ? (showCaptureBackground ? "关闭截图背景" : "用截屏当背景") : "请先截屏"}
              style={{
                padding: "4px 8px",
                cursor: captureB64 ? "pointer" : "not-allowed",
                background: "#444",
                color: "#eee",
                border: "1px solid #666",
                borderRadius: 4,
              }}
            >
              截图背景: {showCaptureBackground ? "开" : "关"}
            </button>
          </div>
          {captureB64 && (
            <div style={{ marginTop: 6 }}>
              <img
                src={`data:image/png;base64,${captureB64}`}
                alt="截屏"
                style={{ maxWidth: 200, maxHeight: 120, border: "1px solid #666" }}
              />
              <div style={{ marginTop: 4 }}>
                <a
                  href={`data:image/png;base64,${captureB64}`}
                  download="screen-excluding-self.png"
                  style={{ color: "#8af" }}
                >
                  下载 PNG
                </a>
              </div>
            </div>
          )}
          <div style={{ marginTop: 4, opacity: 0.7, pointerEvents: "none" }}>
            Cs+H 最小/还原 · Cs+T 穿透 · Cs+R 重画 · Cs+U 保存 · Cs+O 置顶 · Q 退出 · D Debug
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

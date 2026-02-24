import { getCurrentWindow } from "@tauri-apps/api/window";
import { save } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";
import { useEffect, useRef, useState } from "react";

type DrawState = "idle" | "drawing";

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: window.innerWidth, h: window.innerHeight });
  const [drawState, setDrawState] = useState<DrawState>("idle");
  const [pointer, setPointer] = useState({ x: 0, y: 0, pressure: 0 });
  const [debugVisible, setDebugVisible] = useState(true);
  const [fps, setFps] = useState(0);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const fpsRef = useRef({ frames: 0, last: performance.now() });
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

  // 快捷键：Ctrl+Shift+Q 退出，Ctrl+Shift+H 隐藏，Ctrl+Shift+D 切换 Debug，Ctrl+Shift+C 清空
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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
      setSaveStatus("失败: " + String(e));
      setTimeout(() => setSaveStatus(null), 3000);
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!e.ctrlKey || !e.shiftKey) return;
      if (e.key === "Q") {
        e.preventDefault();
        getCurrentWindow().close();
      } else if (e.key === "H") {
        e.preventDefault();
        getCurrentWindow().hide();
      } else if (e.key === "D") {
        e.preventDefault();
        setDebugVisible((v) => !v);
      } else if (e.key === "C") {
        e.preventDefault();
        clearCanvas();
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

  const draw = (x: number, y: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const px = x * dpr;
    const py = y * dpr;
    ctx.lineTo(px, py);
    ctx.stroke();
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    setDrawState("drawing");
    const x = e.clientX;
    const y = e.clientY;
    setPointer({ x, y, pressure: e.pressure ?? 0 });
    ctx.beginPath();
    ctx.moveTo(x * dpr, y * dpr);
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  };

  const onPointerMove = (e: React.PointerEvent) => {
    setPointer({ x: e.clientX, y: e.clientY, pressure: e.pressure ?? 0 });
    if (drawState === "drawing") draw(e.clientX, e.clientY);
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (drawState === "drawing") draw(e.clientX, e.clientY);
    setDrawState("idle");
  };

  const onPointerCancel = () => setDrawState("idle");

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      style={{ width: "100vw", height: "100vh", position: "relative", overflow: "hidden", outline: "none" }}
    >
      {/* 画布层：铺满视口，逻辑尺寸用 CSS，像素尺寸用 width/height 避免拉伸模糊 */}
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
          </div>
          <div style={{ marginTop: 4, opacity: 0.7, pointerEvents: "none" }}>
            Ctrl+Shift+Q 退出 · H 隐藏 · D 切换 · C 清空
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

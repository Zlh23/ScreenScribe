//! 在屏幕 DC 上用 R2_NOT 画线：线显示为下层窗口的反色，无需截图。
//! R2_NOT 每次绘制会取反目标像素，所以每条线只画一次；清空时再画一遍取反回去。
#![cfg(target_os = "windows")]

use serde::Deserialize;
use std::sync::Mutex;
use windows::Win32::Foundation::{COLORREF, HWND};
use windows::Win32::Graphics::Gdi::{
    CreatePen, DeleteObject, GetDC, LineTo, MoveToEx, ReleaseDC, SelectObject, SetROP2, PS_SOLID,
    R2_NOT,
};

#[derive(Deserialize, Clone)]
pub struct AddLineArgs {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
    #[serde(default = "default_width")]
    pub width: i32,
}

fn default_width() -> i32 {
    4
}

#[derive(Clone)]
struct Line {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    width: i32,
}

static LINES: Mutex<Vec<Line>> = Mutex::new(Vec::new());

/// 在屏幕 DC 上用 R2_NOT 画若干线段（用于单次绘制或擦除）。
fn draw_lines_r2_not(lines: &[Line]) {
    if lines.is_empty() {
        return;
    }
    unsafe {
        let hdc = GetDC(HWND::default());
        if hdc.is_invalid() {
            return;
        }
        let _ = SetROP2(hdc, R2_NOT);
        for line in lines {
            let pen = CreatePen(PS_SOLID, line.width, COLORREF(0x00FFFFFF));
            let old = SelectObject(hdc, pen);
            let _ = MoveToEx(hdc, line.x1, line.y1, None);
            let _ = LineTo(hdc, line.x2, line.y2);
            let _ = SelectObject(hdc, old);
            let _ = DeleteObject(pen);
        }
        let _ = ReleaseDC(HWND::default(), hdc);
    }
}

/// 在屏幕 DC 上画一条线（R2_NOT 反色效果）。坐标为物理像素。每条线只画一次。
#[tauri::command]
pub fn add_line_on_screen(args: AddLineArgs) -> Result<(), String> {
    let width = if args.width <= 0 { 4 } else { args.width };
    let line = Line {
        x1: args.x1,
        y1: args.y1,
        x2: args.x2,
        y2: args.y2,
        width,
    };
    draw_lines_r2_not(&[line.clone()]);
    if let Ok(mut lines) = LINES.lock() {
        lines.push(line);
    }
    Ok(())
}

/// 清空屏幕上的反色线：用 R2_NOT 再画一遍所有线（取反回去），再清空列表。
#[tauri::command]
pub fn clear_screen_lines() -> Result<(), String> {
    let lines = if let Ok(mut lines) = LINES.lock() {
        std::mem::take(&mut *lines)
    } else {
        return Ok(());
    };
    draw_lines_r2_not(&lines);
    Ok(())
}

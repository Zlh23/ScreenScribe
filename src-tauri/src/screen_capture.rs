//! 屏幕截图（排除本窗口）：SetWindowDisplayAffinity + GDI BitBlt。
#![cfg(target_os = "windows")]

use image::{ImageBuffer, RgbImage};
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits,
    GetDeviceCaps, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, SRCCOPY,
    HORZRES, VERTRES,
};
use windows::Win32::UI::WindowsAndMessaging::{SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE};

/// 将本窗口从系统截图/录屏中排除（截图中会显示下层内容）。
pub unsafe fn set_exclude_from_capture(hwnd: HWND) -> Result<(), String> {
    SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE).map_err(|e| e.to_string())?;
    Ok(())
}

/// 截取整屏（本窗口已设为排除则截图中不含本窗口），返回 PNG 字节。
pub fn capture_screen_png() -> Result<Vec<u8>, String> {
    unsafe {
        let hdc_screen = GetDC(HWND::default());
        if hdc_screen.is_invalid() {
            return Err("GetDC(NULL) failed".into());
        }
        let width = GetDeviceCaps(hdc_screen, HORZRES) as u32;
        let height = GetDeviceCaps(hdc_screen, VERTRES) as u32;
        if width == 0 || height == 0 {
            let _ = ReleaseDC(HWND::default(), hdc_screen);
            return Err("GetDeviceCaps 获取尺寸失败".into());
        }

        let hdc_mem = CreateCompatibleDC(hdc_screen);
        if hdc_mem.is_invalid() {
            let _ = ReleaseDC(HWND::default(), hdc_screen);
            return Err("CreateCompatibleDC failed".into());
        }

        let hbm = CreateCompatibleBitmap(hdc_screen, width as i32, height as i32);
        if hbm.is_invalid() {
            let _ = DeleteDC(hdc_mem);
            let _ = ReleaseDC(HWND::default(), hdc_screen);
            return Err("CreateCompatibleBitmap failed".into());
        }

        let _old = SelectObject(hdc_mem, hbm);
        BitBlt(
            hdc_mem,
            0,
            0,
            width as i32,
            height as i32,
            hdc_screen,
            0,
            0,
            SRCCOPY,
        )
        .map_err(|e| e.to_string())?;

        let mut info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width as i32,
                biHeight: -(height as i32),
                biPlanes: 1,
                biBitCount: 24,
                biCompression: 0,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            },
            bmiColors: [std::mem::zeroed()],
        };
        let row_bytes = (width * 3 + 3) & !3;
        let size = (row_bytes * height) as usize;
        let mut bits = vec![0u8; size];
        let lines = GetDIBits(
            hdc_mem,
            hbm,
            0,
            height,
            Some(bits.as_mut_ptr() as *mut _),
            &mut info,
            DIB_RGB_COLORS,
        );
        let _ = DeleteObject(hbm);
        let _ = DeleteDC(hdc_mem);
        let _ = ReleaseDC(HWND::default(), hdc_screen);
        if lines == 0 {
            return Err("GetDIBits failed".into());
        }

        let mut img: RgbImage = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let src = (y * row_bytes + x * 3) as usize;
                let b = bits[src];
                let g = bits[src + 1];
                let r = bits[src + 2];
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }

        let mut out = Vec::new();
        let enc = image::codecs::png::PngEncoder::new(&mut out);
        image::ImageEncoder::write_image(
            enc,
            img.as_raw(),
            width,
            height,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| e.to_string())?;
        Ok(out)
    }
}

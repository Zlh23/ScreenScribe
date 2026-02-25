use tauri::Manager;

#[cfg(target_os = "windows")]
mod screen_capture;
#[cfg(target_os = "windows")]
mod screen_invert;

#[cfg(target_os = "windows")]
use screen_invert::{add_line_on_screen, clear_screen_lines};

#[cfg(target_os = "windows")]
#[tauri::command]
fn capture_screen_excluding_self() -> Result<String, String> {
  let png = screen_capture::capture_screen_png()?;
  Ok(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png))
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
fn capture_screen_excluding_self() -> Result<String, String> {
  Err("仅支持 Windows".into())
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
fn add_line_on_screen(_: serde_json::Value) -> Result<(), String> {
  Err("仅支持 Windows".into())
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
fn clear_screen_lines() -> Result<(), String> {
  Err("仅支持 Windows".into())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_fs::init())
    .plugin(tauri_plugin_global_shortcut::Builder::default().build())
    .invoke_handler(tauri::generate_handler![
      capture_screen_excluding_self,
      add_line_on_screen,
      clear_screen_lines,
    ])
    .setup(|app| {
      #[cfg(target_os = "windows")]
      if let Some(w) = app.get_webview_window("main") {
        let _ = w.with_webview(|webview| {
          #[cfg(windows)]
          unsafe {
            let controller = webview.controller();
            let mut raw_hwnd = std::mem::zeroed();
            if controller.ParentWindow(&mut raw_hwnd).is_ok() {
              let ptr = raw_hwnd.0;
              let hwnd = windows::Win32::Foundation::HWND(ptr);
              let root = windows::Win32::UI::WindowsAndMessaging::GetAncestor(
                hwnd,
                windows::Win32::UI::WindowsAndMessaging::GA_ROOT,
              );
              if !root.is_invalid() {
                let _ = screen_capture::set_exclude_from_capture(root);
              }
            }
          }
        });
      }
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
        if let Some(w) = app.get_webview_window("main") {
          let _ = w.open_devtools();
        }
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}

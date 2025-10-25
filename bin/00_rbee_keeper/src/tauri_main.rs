//! Tauri GUI entry point for rbee-keeper
//!
//! TEAM-293: Created Tauri-based GUI application
//!
//! This binary provides a graphical interface for all rbee-keeper commands.
//! It uses the same handler logic as the CLI, ensuring consistent behavior.

use rbee_keeper::tauri_commands::*;

fn main() {
    tauri::Builder::default()
        // Register all Tauri commands
        .invoke_handler(tauri::generate_handler![
            // Status
            get_status,
            // Queen commands
            queen_start,
            queen_stop,
            queen_status,
            queen_rebuild,
            queen_info,
            queen_install,
            queen_uninstall,
            // Hive commands
            hive_install,
            hive_uninstall,
            hive_start,
            hive_stop,
            hive_list,
            hive_get,
            hive_status,
            hive_refresh_capabilities,
            // Worker commands
            worker_spawn,
            worker_process_list,
            worker_process_get,
            worker_process_delete,
            // Model commands
            model_download,
            model_list,
            model_get,
            model_delete,
            // Inference
            infer,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

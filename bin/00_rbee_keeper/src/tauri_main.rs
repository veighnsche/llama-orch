//! Tauri GUI entry point for rbee-keeper
//!
//! TEAM-293: Created Tauri-based GUI application
//! TEAM-297: Updated to use tauri-specta v2.0.0-rc.21 API properly
//!
//! This binary provides a graphical interface for all rbee-keeper commands.
//! It uses the same handler logic as the CLI, ensuring consistent behavior.

use rbee_keeper::tauri_commands::*;
use specta_typescript::Typescript;
use tauri_specta::{collect_commands, Builder};

fn main() {
    // TEAM-297: Build commands with tauri-specta v2.0.0-rc.21
    let builder = Builder::<tauri::Wry>::new()
        // Register commands with #[specta::specta] annotation
        .commands(collect_commands![
            hive_list,
        ]);

    // TEAM-297: Export TypeScript bindings in debug mode
    #[cfg(debug_assertions)]
    builder
        .export(Typescript::default(), "../ui/src/generated/bindings.ts")
        .expect("Failed to export typescript bindings");

    tauri::Builder::default()
        // TEAM-297: Use builder.invoke_handler() for specta-enabled commands
        .invoke_handler(builder.invoke_handler())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

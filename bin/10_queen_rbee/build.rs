// TEAM-295: Build script to compile UI before Rust compilation
//
// This ensures the UI dist folder exists before rust-embed tries to include it.
// Pattern: Run pnpm build for the UI package before cargo build.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=ui/app/src");
    println!("cargo:rerun-if-changed=ui/app/package.json");
    println!("cargo:rerun-if-changed=ui/packages/queen-rbee-sdk/src");
    println!("cargo:rerun-if-changed=ui/packages/queen-rbee-react/src");
    println!("cargo:rerun-if-env-changed=RBEE_SKIP_UI_BUILD");

    // Get workspace root (2 levels up from bin/10_queen_rbee)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();

    // Check if UI dist already exists and is recent
    let ui_dist = Path::new(&manifest_dir).join("ui/app/dist");

    // TEAM-XXX: Allow skipping UI build to prevent Turbo dev crashes during install
    // Set RBEE_SKIP_UI_BUILD=1 when running `cargo build` while dev servers are active
    // In debug mode, the dev proxy serves from Vite (7834) anyway, so embedding dist is optional
    if std::env::var("RBEE_SKIP_UI_BUILD").is_ok() {
        println!("cargo:warning=⏭️  Skipping queen-rbee UI build (RBEE_SKIP_UI_BUILD set)");
        
        // In debug mode with dev proxy, missing dist is OK
        // In release mode, this would be a problem
        if cfg!(debug_assertions) {
            println!("cargo:warning=ℹ️  Debug mode: UI will be served from Vite dev server (7834)");
            return;
        } else if ui_dist.exists() {
            println!("cargo:warning=ℹ️  Release mode: Using existing dist folder");
            return;
        } else {
            panic!("RBEE_SKIP_UI_BUILD set in release mode but dist folder missing! Run 'cd bin/10_queen_rbee/ui/app && pnpm exec vite build' first.");
        }
    }

    // Always build UI to ensure it's up-to-date
    println!("cargo:warning=🔨 Building queen-rbee UI (vite only, skipping tsc)...");

    // Skip TypeScript check during Rust build - just run vite build
    // This prevents TypeScript errors from blocking Rust compilation
    // Developers should run `pnpm dev` or `pnpm build` separately for type checking
    let ui_app_dir = Path::new(&manifest_dir).join("ui/app");

    let status = Command::new("pnpm")
        .args(&["exec", "vite", "build"])
        .current_dir(&ui_app_dir)
        .status()
        .expect("Failed to run vite build for queen-rbee UI");

    if !status.success() {
        panic!("UI build failed! Run 'cd bin/10_queen_rbee/ui/app && pnpm exec vite build' manually to debug.");
    }

    // Verify dist exists
    if !ui_dist.exists() {
        panic!("UI dist folder not found at {:?} after build", ui_dist);
    }

    println!("cargo:warning=✅ queen-rbee UI built successfully");
}

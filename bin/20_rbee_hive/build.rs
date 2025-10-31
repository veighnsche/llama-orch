// TEAM-374: Build script to compile UI before Rust compilation
// Copied from: bin/10_queen_rbee/build.rs
//
// This ensures the UI dist folder exists before rust-embed tries to include it.
// Pattern: Run pnpm build for the UI package before cargo build.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=ui/app/src");
    println!("cargo:rerun-if-changed=ui/app/package.json");
    println!("cargo:rerun-if-changed=ui/packages/rbee-hive-sdk/src");

    // Get workspace root (2 levels up from bin/20_rbee_hive)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let _workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();

    // TEAM-374: Build packages FIRST, then app
    
    let ui_base_dir = Path::new(&manifest_dir).join("ui");
    let ui_app_dir = ui_base_dir.join("app");
    let ui_dist = ui_app_dir.join("dist");
    
    // TEAM-374: Skip ALL UI builds if Vite dev server is running (port 7836)
    // This avoids conflicts with the dev server and speeds up cargo builds during development
    let vite_dev_running = std::net::TcpStream::connect("127.0.0.1:7836").is_ok();
    
    if vite_dev_running {
        println!("cargo:warning=⚡ Vite dev server detected on port 7836 - SKIPPING ALL UI builds");
        println!("cargo:warning=   (Dev server provides fresh packages via hot reload)");
        println!("cargo:warning=   SDK and App builds skipped");
        return; // Skip all UI builds
    }
    
    println!("cargo:warning=🔨 Building rbee-hive UI packages and app...");

    // Step 1: Build the WASM SDK package (rbee-hive-sdk)
    println!("cargo:warning=  📦 Building @rbee/rbee-hive-sdk (WASM)...");
    let sdk_dir = ui_base_dir.join("packages/rbee-hive-sdk");
    let sdk_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&sdk_dir)
        .status()
        .expect("Failed to build rbee-hive-sdk");

    if !sdk_status.success() {
        panic!("SDK build failed! Run 'cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk && pnpm build' to debug.");
    }

    // Step 2: Build the app (which now has fresh packages)
    println!("cargo:warning=  🎨 Building @rbee/rbee-hive-ui app...");
    let app_status = Command::new("pnpm")
        .args(&["exec", "vite", "build"])
        .current_dir(&ui_app_dir)
        .status()
        .expect("Failed to run vite build for rbee-hive UI");

    if !app_status.success() {
        panic!("UI build failed! Run 'cd bin/20_rbee_hive/ui/app && pnpm exec vite build' to debug.");
    }

    // Verify dist exists
    if !ui_dist.exists() {
        panic!("UI dist folder not found at {:?} after build", ui_dist);
    }

    println!("cargo:warning=✅ rbee-hive UI (SDK + App) built successfully");
}

// TEAM-295: Build script to compile UI before Rust compilation
//
// This ensures the UI dist folder exists before rust-embed tries to include it.
// Pattern: Run pnpm build for the UI package before cargo build.

use std::path::Path;
use std::process::Command;

fn main() {
    // TEAM-XXX: Generate build metadata using shadow-rs
    shadow_rs::new().expect("Failed to generate shadow-rs build metadata");

    println!("cargo:rerun-if-changed=ui/app/src");
    println!("cargo:rerun-if-changed=ui/app/package.json");
    println!("cargo:rerun-if-changed=ui/packages/queen-rbee-sdk/src");
    println!("cargo:rerun-if-changed=ui/packages/queen-rbee-react/src");

    // Get workspace root (2 levels up from bin/10_queen_rbee)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let _workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap(); // TEAM-380: Reserved for future use

    // TEAM-350: REAL FIX - Build packages FIRST, then app
    // This allows cargo watch to rebuild everything without needing turbo dev server

    // TEAM_527: Allow tests and CI to skip expensive UI build when RBEE_SKIP_UI_BUILD is set
    if std::env::var("RBEE_SKIP_UI_BUILD").is_ok() {
        println!("cargo:warning=⏭️  Skipping queen-rbee UI build (RBEE_SKIP_UI_BUILD set)");
        return;
    }

    let ui_base_dir = Path::new(&manifest_dir).join("ui");
    let ui_app_dir = ui_base_dir.join("app");
    let ui_dist = ui_app_dir.join("dist");

    // TEAM-350: Skip ALL UI builds if Vite dev server is running (port 7834)
    // This avoids conflicts with the dev server and speeds up cargo builds during development
    // TEAM-381: Use HTTP check instead of TCP to avoid false positives from TIME_WAIT sockets
    let vite_dev_running = Command::new("curl")
        .args(&["-s", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:7834"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|code| code.starts_with('2') || code.starts_with('3')) // 2xx or 3xx response
        .unwrap_or(false);

    if vite_dev_running {
        println!("cargo:warning=⚡ Vite dev server detected on port 7834 - SKIPPING ALL UI builds");
        println!("cargo:warning=   (Dev server provides fresh packages via hot reload)");
        println!("cargo:warning=   SDK, React, and App builds skipped");
        return; // Skip all UI builds
    }

    println!("cargo:warning=🔨 Building queen-rbee UI packages and app...");

    // Step 1: Build the WASM SDK package (queen-rbee-sdk)
    // TEAM_528: Skip WASM SDK build during cargo build to avoid lock contention
    println!("cargo:warning=  📦 Building @rbee/queen-rbee-sdk (WASM)...");
    let sdk_dir = ui_base_dir.join("packages/queen-rbee-sdk");
    let sdk_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&sdk_dir)
        .env("RBEE_SKIP_WASM", "1") // TEAM_528: Skip wasm-pack during cargo build
        .status()
        .expect("Failed to build queen-rbee-sdk");

    if !sdk_status.success() {
        panic!("SDK build failed! Run 'cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk && pnpm build' to debug.");
    }

    // Step 2: Build the React hooks package (queen-rbee-react)
    println!("cargo:warning=  📦 Building @rbee/queen-rbee-react...");
    let react_dir = ui_base_dir.join("packages/queen-rbee-react");
    let react_status = Command::new("pnpm")
        .args(&["build"])
        .current_dir(&react_dir)
        .status()
        .expect("Failed to build queen-rbee-react");

    if !react_status.success() {
        panic!("React package build failed! Run 'cd bin/10_queen_rbee/ui/packages/queen-rbee-react && pnpm build' to debug.");
    }

    // Step 3: Build the app (which now has fresh packages)
    println!("cargo:warning=  🎨 Building @rbee/queen-rbee-ui app...");
    let app_status = Command::new("pnpm")
        .args(&["exec", "vite", "build"])
        .current_dir(&ui_app_dir)
        .status()
        .expect("Failed to run vite build for queen-rbee UI");

    if !app_status.success() {
        panic!(
            "UI build failed! Run 'cd bin/10_queen_rbee/ui/app && pnpm exec vite build' to debug."
        );
    }

    // Verify dist exists
    if !ui_dist.exists() {
        panic!("UI dist folder not found at {:?} after build", ui_dist);
    }

    println!("cargo:warning=✅ queen-rbee UI (SDK + React + App) built successfully");
}

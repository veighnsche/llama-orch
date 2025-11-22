// TEAM-374: Build script to compile UI before Rust compilation
// Copied from: bin/10_queen_rbee/build.rs
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
    println!("cargo:rerun-if-changed=ui/packages/rbee-hive-sdk/src");

    // Get workspace root (2 levels up from bin/20_rbee_hive)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let _workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();

    // TEAM-374: Build packages FIRST, then app

    // TEAM_527: Allow tests and CI to skip expensive UI build when RBEE_SKIP_UI_BUILD is set
    if std::env::var("RBEE_SKIP_UI_BUILD").is_ok() {
        println!("cargo:warning=⏭️  Skipping rbee-hive UI build (RBEE_SKIP_UI_BUILD set)");
        return;
    }

    let ui_base_dir = Path::new(&manifest_dir).join("ui");
    let ui_app_dir = ui_base_dir.join("app");
    let ui_dist = ui_app_dir.join("dist");

    // TEAM-381: Skip ALL UI builds if Vite dev server is running (port 7836)
    // TEAM-386: Also check for turbo dev process to prevent killing active dev sessions
    // This avoids conflicts with the dev server and speeds up cargo builds during development

    // Check 1: HTTP check for rbee-hive Vite dev server (port 7836)
    let vite_dev_running = Command::new("curl")
        .args(&["-s", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:7836"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|code| code.starts_with('2') || code.starts_with('3')) // 2xx or 3xx response
        .unwrap_or(false);

    // Check 2: Look for turbo dev process (prevents killing turbo dev sessions)
    let turbo_dev_running = Command::new("pgrep")
        .args(&["-f", "turbo.*dev"])
        .output()
        .ok()
        .map(|output| !output.stdout.is_empty())
        .unwrap_or(false);

    if vite_dev_running || turbo_dev_running {
        if vite_dev_running {
            println!(
                "cargo:warning=⚡ Vite dev server detected on port 7836 - SKIPPING ALL UI builds"
            );
        }
        if turbo_dev_running {
            println!("cargo:warning=⚡ Turbo dev process detected - SKIPPING ALL UI builds");
            println!("cargo:warning=   (Prevents killing active turbo dev session)");
        }
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
        panic!(
            "UI build failed! Run 'cd bin/20_rbee_hive/ui/app && pnpm exec vite build' to debug."
        );
    }

    // Verify dist exists
    if !ui_dist.exists() {
        panic!("UI dist folder not found at {:?} after build", ui_dist);
    }

    println!("cargo:warning=✅ rbee-hive UI (SDK + App) built successfully");
}

// TEAM-295: Build script to compile UI before Rust compilation
//
// This ensures the UI dist folder exists before rust-embed tries to include it.
// Pattern: Run pnpm build for the UI package before cargo build.

use std::process::Command;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=ui/app/src");
    println!("cargo:rerun-if-changed=ui/app/package.json");
    println!("cargo:rerun-if-changed=ui/packages/queen-rbee-sdk/src");
    println!("cargo:rerun-if-changed=ui/packages/queen-rbee-react/src");

    // Get workspace root (2 levels up from bin/10_queen_rbee)
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    // Check if UI dist already exists and is recent
    let ui_dist = Path::new(&manifest_dir).join("ui/app/dist");
    
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

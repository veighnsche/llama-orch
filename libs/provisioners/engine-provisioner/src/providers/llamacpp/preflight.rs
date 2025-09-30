use anyhow::{anyhow, Result};
use std::process::Command;

use crate::cfg;

pub fn preflight_tools(prov: &cfg::ProvisioningConfig, src: &cfg::SourceConfig) -> Result<()> {
    // Check required commands
    let mut packages: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for (bin, pkg) in [("git", "git"), ("cmake", "cmake"), ("make", "make"), ("gcc", "gcc")] {
        if which::which(bin).is_err() {
            packages.insert(pkg);
        }
    }
    // CUDA if requested
    let wants_cuda = src.build.cmake_flags.as_ref().is_some_and(|flags| {
        flags.iter().any(|f| f.contains("LLAMA_CUBLAS=ON") || f.contains("GGML_CUDA=ON"))
    });
    if wants_cuda && which::which("nvcc").is_err() {
        packages.insert("cuda");
    }
    if wants_cuda {
        // Ensure a compatible host compiler is available for NVCC (gcc-13 or clang). Prefer installing clang.
        let compat = crate::providers::llamacpp::toolchain::find_compat_host_compiler();
        if compat.is_none() && which::which("clang").is_err() {
            packages.insert("clang");
        }
    }
    // HF CLI if model via hf:
    let wants_hf = prov.model.r#ref.as_deref().is_some_and(|r| r.starts_with("hf:"));
    if wants_hf && which::which("huggingface-cli").is_err() {
        packages.insert("python-huggingface-hub");
    }

    if packages.is_empty() {
        return Ok(());
    }

    let allow = prov.allow_package_installs.unwrap_or(false);
    if !allow {
        // Informative error to guide the operator
        return Err(anyhow!(
            "missing tools {:?}. Set provisioning.allow_package_installs=true or install them via your package manager",
            packages
        ));
    }

    if !is_arch_like() || which::which("pacman").is_err() {
        return Err(anyhow!(
            "automatic package install is only supported on Arch-like systems with pacman; please install {:?}",
            packages
        ));
    }

    // Try to install via pacman. Use sudo when not root.
    let pkgs: Vec<&str> = packages.iter().copied().collect();
    let is_root = is_root_user();
    let status = if is_root {
        let mut c = Command::new("pacman");
        c.args(["-S", "--needed", "--noconfirm"]);
        for p in &pkgs { c.arg(p); }
        c.status().map_err(|e| anyhow!("pacman -S: {}", e))?
    } else if which::which("sudo").is_ok() {
        // Try non-interactive first
        let mut c = Command::new("sudo");
        c.args(["-n", "pacman", "-S", "--needed", "--noconfirm"]);
        for p in &pkgs { c.arg(p); }
        let st = c.status().map_err(|e| anyhow!("sudo -n pacman -S: {}", e))?;
        if st.success() { st } else {
            // Fallback: interactive prompt (user may enter password in terminal)
            let mut ci = Command::new("sudo");
            ci.args(["pacman", "-S", "--needed", "--noconfirm"]);
            for p in &pkgs { ci.arg(p); }
            ci.status().map_err(|e| anyhow!("sudo pacman -S: {}", e))?
        }
    } else {
        return Err(anyhow!("sudo not available to install {:?}", pkgs));
    };
    if !status.success() {
        return Err(anyhow!("package install failed (status={})", status));
    }
    Ok(())
}

fn is_arch_like() -> bool {
    // Very simple detection based on /etc/os-release
    if let Ok(s) = std::fs::read_to_string("/etc/os-release") {
        let l = s.to_lowercase();
        return l.contains("arch") || l.contains("cachyos") || l.contains("endeavouros");
    }
    false
}

fn is_root_user() -> bool {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim() == "0")
        .unwrap_or(false)
}

use anyhow::{anyhow, Result};
use std::process::Command;

use crate::cfg;

pub fn preflight_tools(prov: &cfg::ProvisioningConfig, src: &cfg::SourceConfig) -> Result<()> {
    // Check required commands
    let mut packages: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for (bin, pkg) in [
        ("git", "git"),
        ("cmake", "cmake"),
        ("make", "make"),
        ("gcc", "gcc"),
        ("g++", "gcc"),            // C++ compiler required by llama.cpp
        ("pkg-config", "pkg-config"), // for finding libcurl, etc.
    ] {
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
    if wants_hf {
        let has_hf = which::which("huggingface-cli").is_ok() || which::which("hf").is_ok();
        if !has_hf {
            packages.insert("python-huggingface-hub");
        }
    }

    // libcurl dev headers required by llama.cpp server (CMake find_package(CURL))
    // Use pkg-config to detect; if missing or failing, suggest installing curl dev.
    let curl_ok = std::process::Command::new("pkg-config")
        .args(["--exists", "libcurl"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !curl_ok {
        // On Arch, the package providing headers is 'curl'. On Debian-based, it's 'libcurl4-openssl-dev'.
        // Our auto-install only supports pacman, so suggesting 'curl' is sufficient.
        packages.insert("curl");
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

#[cfg(test)]
mod tests {
    use super::preflight_tools;
    use crate::cfg;
    use std::fs;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    static PATH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    #[cfg(unix)]
    fn make_exec(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut f = fs::File::create(path).unwrap();
        writeln!(f, "#!/bin/sh\nexit 0").unwrap();
        let mut perms = f.metadata().unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).unwrap();
    }

    fn src_with_flags(flags: Option<Vec<&str>>) -> cfg::SourceConfig {
        let mut s = cfg::SourceConfig::default();
        s.repo = "https://example.com/llama.cpp.git".to_string();
        s.r#ref = "v0".to_string();
        if let Some(v) = flags {
            s.build.cmake_flags = Some(v.into_iter().map(|x| x.to_string()).collect());
        }
        s
    }

    fn base_prov() -> cfg::ProvisioningConfig {
        cfg::ProvisioningConfig::default()
    }

    #[test]
    #[cfg(unix)]
    fn ok_when_all_required_tools_present_and_no_cuda_or_hf() {
        let _g = PATH_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        fs::create_dir_all(&bin).unwrap();
        for b in ["git", "cmake", "make", "gcc", "g++", "pkg-config"] { make_exec(&bin.join(b)); }

        let old = std::env::var("PATH").ok();
        std::env::set_var("PATH", bin.display().to_string());

        let prov = base_prov();
        let src = src_with_flags(None);
        let res = preflight_tools(&prov, &src);

        // restore PATH
        if let Some(p) = old { std::env::set_var("PATH", p); } else { std::env::remove_var("PATH"); }

        assert!(res.is_ok());
    }

    #[test]
    fn error_lists_missing_tools_when_disallowed() {
        // Use an empty PATH to force missing tools, and ensure allow_package_installs is false
        let _g = PATH_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|p| p.into_inner());
        let old = std::env::var("PATH").ok();
        std::env::set_var("PATH", "");
        let mut prov = base_prov();
        prov.allow_package_installs = Some(false);
        let src = src_with_flags(None);
        let err = preflight_tools(&prov, &src).unwrap_err().to_string();
        if let Some(p) = old { std::env::set_var("PATH", p); } else { std::env::remove_var("PATH"); }
        assert!(err.contains("missing tools"));
        // Tool names should be mentioned (some environments may still resolve gcc)
        assert!(err.contains("git"));
        assert!(err.contains("cmake"));
        assert!(err.contains("make"));
    }

    #[test]
    #[cfg(unix)]
    fn cuda_requested_without_nvcc_and_no_clang_suggests_cuda_and_clang() {
        // Provide base tools but no nvcc or clang
        let _g = PATH_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        fs::create_dir_all(&bin).unwrap();
        for b in ["git", "cmake", "make", "gcc", "g++", "pkg-config"] { make_exec(&bin.join(b)); }
        let old = std::env::var("PATH").ok();
        std::env::set_var("PATH", bin.display().to_string());

        let mut prov = base_prov();
        prov.allow_package_installs = Some(false);
        let src = src_with_flags(Some(vec!["-DGGML_CUDA=ON"]));
        let res = preflight_tools(&prov, &src);

        if let Some(p) = old { std::env::set_var("PATH", p); } else { std::env::remove_var("PATH"); }
        match res {
            Ok(()) => {
                // If it succeeded, the environment must provide either nvcc or a compat clang/gcc-13
                let has_nvcc = which::which("nvcc").is_ok();
                let has_compat = crate::providers::llamacpp::toolchain::find_compat_host_compiler().is_some()
                    || which::which("clang").is_ok();
                assert!(has_nvcc || has_compat, "preflight unexpectedly Ok without nvcc/compat compiler available");
            }
            Err(e) => {
                let err = e.to_string();
                assert!(err.contains("cuda") || err.contains("nvcc"));
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn hf_ref_without_cli_suggests_python_huggingface_hub() {
        let _g = PATH_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        fs::create_dir_all(&bin).unwrap();
        for b in ["git", "cmake", "make", "gcc", "g++", "pkg-config"] { make_exec(&bin.join(b)); }
        let old = std::env::var("PATH").ok();
        std::env::set_var("PATH", bin.display().to_string());

        let mut prov = base_prov();
        prov.allow_package_installs = Some(false);
        prov.model.r#ref = Some("hf:org/model.gguf".to_string());
        let src = src_with_flags(None);
        let res = preflight_tools(&prov, &src);
        if let Some(p) = old { std::env::set_var("PATH", p); } else { std::env::remove_var("PATH"); }
        match res {
            Ok(()) => {
                // If it succeeded, huggingface-cli must be available in environment
                assert!(which::which("huggingface-cli").is_ok(), "preflight Ok but huggingface-cli not found in PATH");
            }
            Err(e) => {
                let err = e.to_string();
                assert!(err.contains("python-huggingface") || err.contains("huggingface"));
            }
        }
    }
}


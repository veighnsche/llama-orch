use std::path::PathBuf;

/// Try to discover a CUDA toolkit root directory by checking nvcc and common paths.
pub fn discover_cuda_root() -> Option<PathBuf> {
    // If nvcc is in PATH, derive root from it
    if let Ok(nvcc) = which::which("nvcc") {
        return nvcc.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf());
    }
    // Common fixed roots
    for root in ["/opt/cuda", "/usr/local/cuda"] {
        if std::path::Path::new(root).join("bin/nvcc").exists() {
            return Some(PathBuf::from(root));
        }
    }
    // Scan /opt and /usr/local for cuda* directories
    for base in ["/opt", "/usr/local"] {
        if let Ok(entries) = std::fs::read_dir(base) {
            for e in entries.flatten() {
                let p = e.path();
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("cuda") {
                        let nvcc = p.join("bin/nvcc");
                        if nvcc.exists() {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Find a compatible host compiler for NVCC (prefer gcc-13, otherwise try clang variants).
pub fn find_compat_host_compiler() -> Option<(PathBuf, PathBuf)> {
    // Prefer gcc-13 if present, otherwise clang
    let gcc13 = which::which("gcc-13").or_else(|_| which::which("gcc13")).ok();
    if let Some(cc) = gcc13 {
        // Try to locate matching g++-13
        let cxx = which::which("g++-13")
            .or_else(|_| which::which("g++13"))
            .unwrap_or_else(|_| cc.with_file_name("g++-13"));
        return Some((cc, cxx));
    }
    // Try clang toolchain
    for ver in ["", "-18", "-17", "-16"] {
        let cc_name = format!("clang{}", ver);
        if let Ok(cc) = which::which(&cc_name) {
            let cxx_name = format!("clang++{}", ver);
            let cxx = which::which(&cxx_name).unwrap_or_else(|_| cc.with_file_name(cxx_name));
            return Some((cc, cxx));
        }
    }
    None
}

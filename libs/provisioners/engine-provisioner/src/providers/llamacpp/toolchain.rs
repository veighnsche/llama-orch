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

#[cfg(test)]
mod tests {
    use super::{discover_cuda_root, find_compat_host_compiler};
    use std::fs;
    use std::io::Write;

    #[cfg(unix)]
    fn make_exec(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut f = fs::File::create(path).unwrap();
        writeln!(f, "#!/bin/sh\nexit 0").unwrap();
        let mut perms = f.metadata().unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn discover_cuda_root_uses_path_nvcc_parent_parent() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("fake-cuda");
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let nvcc = bin.join("nvcc");
        make_exec(&nvcc);

        let old_path = std::env::var("PATH").ok();
        std::env::set_var("PATH", format!("{}:{}", bin.display(), old_path.as_deref().unwrap_or("")));
        let found = discover_cuda_root();
        // Restore
        if let Some(p) = old_path { std::env::set_var("PATH", p); } else { std::env::remove_var("PATH"); }

        assert_eq!(found.as_deref(), Some(root.as_path()));
    }

    #[test]
    #[cfg(unix)]
    fn find_compat_prefers_gcc13_then_falls_back_to_clang() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        fs::create_dir_all(&bin).unwrap();

        // First: only clang-17 present
        let clang = bin.join("clang-17");
        let clangxx = bin.join("clang++-17");
        make_exec(&clang);
        make_exec(&clangxx);

        let old_path = std::env::var("PATH").ok();
        std::env::set_var("PATH", format!("{}:{}", bin.display(), old_path.as_deref().unwrap_or("")));
        let (cc1, cxx1) = find_compat_host_compiler().expect("should find clang");
        assert!(cc1.file_name().unwrap().to_string_lossy().starts_with("clang"));
        assert!(cxx1.file_name().unwrap().to_string_lossy().starts_with("clang++"));

        // Now, add gcc-13 and g++-13 and expect preference switch
        let gcc = bin.join("gcc-13");
        let gxx = bin.join("g++-13");
        make_exec(&gcc);
        make_exec(&gxx);
        let (cc2, cxx2) = find_compat_host_compiler().expect("should find gcc-13");
        assert_eq!(cc2.file_name().unwrap(), "gcc-13");
        assert_eq!(cxx2.file_name().unwrap(), "g++-13");

        // Restore PATH
        if let Some(p) = old_path { std::env::set_var("PATH", p); } else { std::env::remove_var("PATH"); }
    }
}


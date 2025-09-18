use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use std::process::Command;

use crate::plan::{Plan, PlanStep};
use crate::util::{cmd, cmd_in, default_cache_dir, default_models_cache, default_run_dir, ok_status};
use crate::{cfg, EngineProvisioner};

/// Llama.cpp provider: source-from-git build and run (llama-server)
pub struct LlamaCppSourceProvisioner;

impl LlamaCppSourceProvisioner {
    pub fn new() -> Self {
        Self
    }
}

impl EngineProvisioner for LlamaCppSourceProvisioner {
    fn plan(&self, pool: &cfg::PoolConfig) -> Result<Plan> {
        let mut plan = Plan { pool_id: pool.id.clone(), steps: Vec::new() };
        let prov = &pool.provisioning;
        plan.steps.push(PlanStep { kind: "preflight-tools".into(), detail: format!("allow_package_installs={:?}", prov.allow_package_installs) });
        if let Some(src) = &prov.source {
            plan.steps.push(PlanStep { kind: "git-clone".into(), detail: format!("repo={} ref={}", src.repo, src.r#ref) });
            plan.steps.push(PlanStep { kind: "cmake-configure".into(), detail: format!("flags={:?} generator={:?} cache_dir={:?}", src.build.cmake_flags, src.build.generator, src.build.cache_dir) });
            plan.steps.push(PlanStep { kind: "cmake-build".into(), detail: "build llama-server".into() });
        }
        plan.steps.push(PlanStep { kind: "model-fetch".into(), detail: format!("ref={:?} cache_dir={:?}", prov.model.r#ref, prov.model.cache_dir) });
        plan.steps.push(PlanStep { kind: "run".into(), detail: format!("ports={:?} flags={:?}", prov.ports, prov.flags) });
        Ok(plan)
    }

    fn ensure(&self, pool: &cfg::PoolConfig) -> Result<()> {
        let prov = &pool.provisioning;
        let src = prov.source.as_ref().ok_or_else(|| anyhow!("provisioning.source required for source mode"))?;

        // Preflight: ensure required tools; optionally install via pacman if allowed and running on Arch.
        preflight_tools(prov, src)?;

        // Determine cache/build directories
        let cache_dir = src.build.cache_dir.as_deref().map(PathBuf::from).unwrap_or_else(|| default_cache_dir("llamacpp"));
        let src_dir = cache_dir.join("src");
        let build_dir = cache_dir.join("build");

        std::fs::create_dir_all(&cache_dir).context("creating cache_dir")?;

        // Clone or update repo
        if !src_dir.exists() {
            cmd("git").arg("clone").arg(&src.repo).arg(&src_dir).status().context("git clone").and_then(ok_status)?;
        }
        // Checkout ref
        cmd_in(&src_dir, "git").args(["fetch", "--all", "--tags"]).status().context("git fetch").and_then(ok_status)?;
        cmd_in(&src_dir, "git").args(["checkout", &src.r#ref]).status().context("git checkout").and_then(ok_status)?;

        // Configure CMake
        std::fs::create_dir_all(&build_dir).context("create build dir")?;
        // If prior configure cached deprecated LLAMA_CUBLAS in CMakeCache.txt, purge cache once.
        let original_flags = src.build.cmake_flags.clone().unwrap_or_default();
        let had_old_flag = original_flags.iter().any(|f| f.contains("LLAMA_CUBLAS"));
        if had_old_flag {
            let cache = build_dir.join("CMakeCache.txt");
            if cache.exists() { let _ = std::fs::remove_file(&cache); }
            let cmfiles = build_dir.join("CMakeFiles");
            if cmfiles.exists() { let _ = std::fs::remove_dir_all(&cmfiles); }
        }
        // Map deprecated flags for newer llama.cpp: remove any -DLLAMA_CUBLAS* and use -DGGML_CUDA=ON
        let orig_flags = src.build.cmake_flags.clone().unwrap_or_default();
        let has_old = orig_flags.iter().any(|f| f.contains("LLAMA_CUBLAS"));
        let has_new = orig_flags.iter().any(|f| f.contains("GGML_CUDA"));
        let mut mapped_flags: Vec<String> = orig_flags
            .into_iter()
            .filter(|f| !f.contains("LLAMA_CUBLAS"))
            .collect();
        if has_old && !has_new {
            mapped_flags.push("-DGGML_CUDA=ON".to_string());
        }

        let mut cfgcmd = Command::new("cmake");
        cfgcmd
            .current_dir(&src_dir)
            .args(["-S", ".", "-B"]).arg(build_dir.to_string_lossy().to_string())
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg("-DLLAMA_BUILD_SERVER=ON");
        // Ensure any cached deprecated flag is removed
        cfgcmd.arg("-U").arg("LLAMA_CUBLAS");
        // If CUDA is requested but nvcc is not discoverable, provide CUDAToolkit_ROOT and PATH hints
        let wants_cuda_env = mapped_flags.iter().any(|f| f.contains("GGML_CUDA=ON"));
        let mut gpu_enabled = wants_cuda_env;
        if wants_cuda_env && which::which("nvcc").is_err() {
            if let Some(root) = discover_cuda_root() {
                let root_s = root.to_string_lossy().to_string();
                let bin = root.join("bin");
                let nvcc = bin.join("nvcc");
                // Log chosen CUDA root for troubleshooting
                println!("hinting CUDA root at {}", root_s);
                cfgcmd.env("CUDAToolkit_ROOT", &root_s);
                let new_path = match std::env::var_os("PATH") {
                    Some(p) => format!("{}:{}", bin.to_string_lossy(), std::ffi::OsString::from(p).to_string_lossy()),
                    None => bin.to_string_lossy().to_string(),
                };
                cfgcmd.env("PATH", new_path);
                // Also set common aliases some Find scripts still use
                cfgcmd.env("CUDA_HOME", &root_s);
                cfgcmd.env("CUDA_PATH", &root_s);
                cfgcmd.env("CUDA_TOOLKIT_ROOT_DIR", &root_s);
                cfgcmd.env("CUDACXX", nvcc.to_string_lossy().to_string());
                // And pass as cache variables too
                cfgcmd.arg(format!("-DCUDAToolkit_ROOT={}", root_s));
                cfgcmd.arg(format!("-DCUDA_TOOLKIT_ROOT_DIR={}", root_s));
                cfgcmd.arg(format!("-DCMAKE_CUDA_COMPILER={}", nvcc.to_string_lossy()));
            }
        }
        for f in &mapped_flags { cfgcmd.arg(f); }
        let status = cfgcmd.status().context("cmake configure")?;
        if !status.success() && wants_cuda_env {
            // Try again with a compatible CUDA host compiler if available (gcc-13 or clang)
            if let Some((cc, cxx)) = find_compat_host_compiler() {
                eprintln!(
                    "warning: CUDA configure failed; retrying with host compiler CC={} CXX={}",
                    cc.display(),
                    cxx.display()
                );
                let mut cfgcmd_hc = Command::new("cmake");
                cfgcmd_hc
                    .current_dir(&src_dir)
                    .args(["-S", ".", "-B"]).arg(build_dir.to_string_lossy().to_string())
                    .arg("-DCMAKE_BUILD_TYPE=Release")
                    .arg("-DLLAMA_BUILD_SERVER=ON")
                    .arg("-U").arg("LLAMA_CUBLAS")
                    .arg(format!("-DCMAKE_C_COMPILER={}", cc.display()))
                    .arg(format!("-DCMAKE_CXX_COMPILER={}", cxx.display()))
                    .arg(format!("-DCMAKE_CUDA_HOST_COMPILER={}", cc.display()));
                for f in &mapped_flags { cfgcmd_hc.arg(f); }
                let st_hc = cfgcmd_hc.status().context("cmake configure (host-compiler)")?;
                if st_hc.success() {
                    // proceed with build
                } else {
                    // Final fallback: CPU-only
                    eprintln!("warning: CUDA configure still failing; retrying with CPU-only (-DGGML_CUDA=OFF)");
                    gpu_enabled = false;
                    let mut cfgcmd2 = Command::new("cmake");
                    cfgcmd2
                        .current_dir(&src_dir)
                        .args(["-S", ".", "-B"]).arg(build_dir.to_string_lossy().to_string())
                        .arg("-DCMAKE_BUILD_TYPE=Release")
                        .arg("-DLLAMA_BUILD_SERVER=ON")
                        .arg("-U").arg("LLAMA_CUBLAS")
                        .arg("-DGGML_CUDA=OFF");
                    // Keep other user flags except any GGML_CUDA/LLAMA_CUBLAS
                    for f in &mapped_flags {
                        if !(f.contains("GGML_CUDA") || f.contains("LLAMA_CUBLAS")) {
                            cfgcmd2.arg(f);
                        }
                    }
                    let status2 = cfgcmd2.status().context("cmake configure (cpu)")?;
                    if !status2.success() {
                        return Err(anyhow!("cmake configure failed"));
                    }
                }
            } else {
                // No compatible host compiler found: CPU-only fallback immediately
                eprintln!("warning: no gcc-13/clang found; retrying with CPU-only (-DGGML_CUDA=OFF)");
                gpu_enabled = false;
                let mut cfgcmd2 = Command::new("cmake");
                cfgcmd2
                    .current_dir(&src_dir)
                    .args(["-S", ".", "-B"]).arg(build_dir.to_string_lossy().to_string())
                    .arg("-DCMAKE_BUILD_TYPE=Release")
                    .arg("-DLLAMA_BUILD_SERVER=ON")
                    .arg("-U").arg("LLAMA_CUBLAS")
                    .arg("-DGGML_CUDA=OFF");
                for f in &mapped_flags { if !(f.contains("GGML_CUDA") || f.contains("LLAMA_CUBLAS")) { cfgcmd2.arg(f); } }
                let status2 = cfgcmd2.status().context("cmake configure (cpu)")?;
                if !status2.success() { return Err(anyhow!("cmake configure failed")); }
            }
        } else if !status.success() {
            return Err(anyhow!("cmake configure failed"));
        }

        // Build
        let jobs = std::thread::available_parallelism().map(|n| n.get().to_string()).unwrap_or_else(|_| "4".to_string());
        cmd("cmake").args(["--build", build_dir.to_string_lossy().as_ref(), "-j", &jobs]).status().context("cmake build").and_then(ok_status)?;

        let server_bin = build_dir.join("bin").join("llama-server");
        if !server_bin.exists() { return Err(anyhow!("llama-server not found at {}", server_bin.display())); }

        // Ensure model is present
        let model_ref = prov.model.r#ref.clone().ok_or_else(|| anyhow!("provisioning.model.ref required"))?;
        let model_cache_dir = prov.model.cache_dir.clone().unwrap_or_else(|| default_models_cache().to_string_lossy().to_string());
        let model_path = crate::util::resolve_model_path(&model_ref, std::path::Path::new(&model_cache_dir));
        if !model_path.exists() {
            if model_ref.starts_with("hf:") {
                if which::which("huggingface-cli").is_ok() {
                    let (repo, file) = crate::util::parse_hf_ref(&model_ref).ok_or_else(|| anyhow!("invalid hf ref: {model_ref}"))?;
                    let mut c = Command::new("huggingface-cli");
                    c.env("HF_HUB_ENABLE_HF_TRANSFER", "1");
                    c.arg("download").arg(&repo).arg(&file).arg("--local-dir").arg(&model_cache_dir).arg("--local-dir-use-symlinks").arg("False");
                    c.status().context("huggingface-cli download")?.success().then_some(()).ok_or_else(|| anyhow!("model download failed"))?;
                } else {
                    return Err(anyhow!("model file not found at {} and huggingface-cli is missing", model_path.display()));
                }
            } else {
                return Err(anyhow!("model file not found at {} (ref: {})", model_path.display(), model_ref));
            }
        }

        // Spawn server process
        let pid_dir = default_run_dir();
        std::fs::create_dir_all(&pid_dir).ok();
        let pid_file = pid_dir.join(format!("{}.pid", pool.id));

        let mut cmdline = Command::new(server_bin);
        cmdline.arg("--model").arg(&model_path).arg("--host").arg("127.0.0.1");
        let port = prov.ports.as_ref().and_then(|v| v.get(0).cloned()).unwrap_or(8080);
        cmdline.arg("--port").arg(port.to_string());
        // Normalize legacy flags and CPU/GPU expectations
        if let Some(flags) = &prov.flags {
            let norm = normalize_llamacpp_flags(flags, gpu_enabled);
            for f in norm { cmdline.arg(f); }
        }

        let child = cmdline.spawn().with_context(|| format!("spawning llama-server for pool {}", pool.id))?;
        std::fs::write(&pid_file, child.id().to_string()).ok();
        println!("spawned llama-server pid={} (pool={})", child.id(), pool.id);
        Ok(())
    }
}

fn preflight_tools(prov: &cfg::ProvisioningConfig, src: &cfg::SourceConfig) -> Result<()> {
    // Check required commands
    let mut packages: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for (bin, pkg) in [("git", "git"), ("cmake", "cmake"), ("make", "make"), ("gcc", "gcc")] {
        if which::which(bin).is_err() { packages.insert(pkg); }
    }
    // CUDA if requested
    let wants_cuda = src
        .build
        .cmake_flags
        .as_ref()
        .map_or(false, |flags| flags.iter().any(|f| f.contains("LLAMA_CUBLAS=ON") || f.contains("GGML_CUDA=ON")));
    if wants_cuda && which::which("nvcc").is_err() {
        packages.insert("cuda");
    }
    if wants_cuda {
        // Ensure a compatible host compiler is available for NVCC (gcc-13 or clang). Prefer installing clang.
        let compat = find_compat_host_compiler();
        if compat.is_none() && which::which("clang").is_err() {
            packages.insert("clang");
        }
    }
    // HF CLI if model via hf:
    let wants_hf = prov.model.r#ref.as_deref().map_or(false, |r| r.starts_with("hf:"));
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
        c.status().context("pacman -S")?
    } else if which::which("sudo").is_ok() {
        // Try non-interactive first
        let mut c = Command::new("sudo");
        c.args(["-n", "pacman", "-S", "--needed", "--noconfirm"]);
        for p in &pkgs { c.arg(p); }
        let st = c.status().context("sudo -n pacman -S")?;
        if st.success() {
            st
        } else {
            // Fallback: interactive prompt (user may enter password in terminal)
            let mut ci = Command::new("sudo");
            ci.args(["pacman", "-S", "--needed", "--noconfirm"]);
            for p in &pkgs { ci.arg(p); }
            ci.status().context("sudo pacman -S")?
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

fn find_compat_host_compiler() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    // Prefer gcc-13 if present, otherwise clang
    let gcc13 = which::which("gcc-13").or_else(|_| which::which("gcc13")).ok();
    if let Some(cc) = gcc13 {
        // Try to locate matching g++-13
        let cxx = which::which("g++-13").or_else(|_| which::which("g++13")).unwrap_or_else(|_| cc.with_file_name("g++-13"));
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

fn normalize_llamacpp_flags(flags: &[String], gpu_enabled: bool) -> Vec<String> {
    // Map legacy options to current llama.cpp conventions and enforce CPU/GPU consistency.
    // - "--ngl N" or "-ngl N" or "--gpu-layers N" -> "--n-gpu-layers N"
    // - If CPU-only, force "--n-gpu-layers 0" and drop any previous GPU layer flags.
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < flags.len() {
        let f = &flags[i];
        let next = flags.get(i + 1);
        let is_pair = |s: &str| s == "--ngl" || s == "-ngl" || s == "--gpu-layers" || s == "--n-gpu-layers";
        if is_pair(f) {
            // Consume this flag and its value
            if let Some(val) = next {
                if gpu_enabled {
                    out.push("--n-gpu-layers".to_string());
                    out.push(val.clone());
                }
                i += 2;
                continue;
            } else {
                // Malformed pair; skip it
                i += 1;
                continue;
            }
        }
        // passthrough any other flag
        out.push(f.clone());
        i += 1;
    }
    if !gpu_enabled {
        // Remove any accidental n-gpu-layers set and enforce 0
        let mut cleaned: Vec<String> = Vec::new();
        let mut j = 0usize;
        while j < out.len() {
            let s = &out[j];
            if s == "--n-gpu-layers" {
                // drop this and its value
                j += 2;
                continue;
            }
            cleaned.push(s.clone());
            j += 1;
        }
        cleaned.push("--n-gpu-layers".to_string());
        cleaned.push("0".to_string());
        return cleaned;
    }
    // If GPU enabled and no layer flag was provided, leave as-is (llama.cpp will choose default)
    out
}

fn discover_cuda_root() -> Option<std::path::PathBuf> {
    // If nvcc is in PATH, derive root from it
    if let Ok(nvcc) = which::which("nvcc") {
        return nvcc.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf());
    }
    // Common fixed roots
    for root in ["/opt/cuda", "/usr/local/cuda"] {
        if std::path::Path::new(root).join("bin/nvcc").exists() {
            return Some(std::path::PathBuf::from(root));
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

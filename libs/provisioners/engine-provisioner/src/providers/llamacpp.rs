//! This page is huge we need to break it down into smaller chunks
//! What are modular pieces that are reusable?

use anyhow::{anyhow, Context, Result};
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::plan::{Plan, PlanStep};
use crate::util::{
    cmd, cmd_in, default_cache_dir, default_models_cache, default_run_dir, ensure_flag,
    ensure_flag_pair, http_ok, ok_status, select_listen_port, wait_for_health, write_handoff_file,
};
use crate::{cfg, EngineProvisioner};

/// Llama.cpp provider: source-from-git build and run (llama-server)
pub struct LlamaCppSourceProvisioner;

impl LlamaCppSourceProvisioner {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlamaCppSourceProvisioner {
    fn default() -> Self {
        Self
    }
}

impl EngineProvisioner for LlamaCppSourceProvisioner {
    fn plan(&self, pool: &cfg::PoolConfig) -> Result<Plan> {
        let mut plan = Plan { pool_id: pool.id.clone(), steps: Vec::new() };
        let prov = &pool.provisioning;
        plan.steps.push(PlanStep {
            kind: "preflight-tools".into(),
            detail: format!("allow_package_installs={:?}", prov.allow_package_installs),
        });
        if let Some(src) = &prov.source {
            plan.steps.push(PlanStep {
                kind: "git-clone".into(),
                detail: format!("repo={} ref={}", src.repo, src.r#ref),
            });
            plan.steps.push(PlanStep {
                kind: "cmake-configure".into(),
                detail: format!(
                    "flags={:?} generator={:?} cache_dir={:?}",
                    src.build.cmake_flags, src.build.generator, src.build.cache_dir
                ),
            });
            plan.steps
                .push(PlanStep { kind: "cmake-build".into(), detail: "build llama-server".into() });
        }
        plan.steps.push(PlanStep {
            kind: "model-fetch".into(),
            detail: format!("ref={:?} cache_dir={:?}", prov.model.r#ref, prov.model.cache_dir),
        });
        plan.steps.push(PlanStep {
            kind: "run".into(),
            detail: format!("ports={:?} flags={:?}", prov.ports, prov.flags),
        });
        Ok(plan)
    }

    fn ensure(&self, pool: &cfg::PoolConfig) -> Result<()> {
        let prov = &pool.provisioning;
        let src = prov
            .source
            .as_ref()
            .ok_or_else(|| anyhow!("provisioning.source required for source mode"))?;

        // Preflight: ensure required tools; optionally install via pacman if allowed and running on Arch.
        preflight_tools(prov, src)?;

        // Determine cache/build directories
        let cache_dir = src
            .build
            .cache_dir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| default_cache_dir("llamacpp"));
        let src_dir = cache_dir.join("src");
        let build_dir = cache_dir.join("build");

        std::fs::create_dir_all(&cache_dir).context("creating cache_dir")?;

        // Clone or update repo
        if !src_dir.exists() {
            cmd("git")
                .arg("clone")
                .arg(&src.repo)
                .arg(&src_dir)
                .status()
                .context("git clone")
                .and_then(ok_status)?;
        }
        // Checkout ref
        cmd_in(&src_dir, "git")
            .args(["fetch", "--all", "--tags"])
            .status()
            .context("git fetch")
            .and_then(ok_status)?;
        cmd_in(&src_dir, "git")
            .args(["checkout", &src.r#ref])
            .status()
            .context("git checkout")
            .and_then(ok_status)?;

        // Configure CMake
        std::fs::create_dir_all(&build_dir).context("create build dir")?;
        // If prior configure cached deprecated LLAMA_CUBLAS in CMakeCache.txt, purge cache once.
        let original_flags = src.build.cmake_flags.clone().unwrap_or_default();
        let had_old_flag = original_flags.iter().any(|f| f.contains("LLAMA_CUBLAS"));
        if had_old_flag {
            let cache = build_dir.join("CMakeCache.txt");
            if cache.exists() {
                let _ = std::fs::remove_file(&cache);
            }
            let cmfiles = build_dir.join("CMakeFiles");
            if cmfiles.exists() {
                let _ = std::fs::remove_dir_all(&cmfiles);
            }
        }
        // Map deprecated flags for newer llama.cpp: remove any -DLLAMA_CUBLAS* and use -DGGML_CUDA=ON
        let orig_flags = src.build.cmake_flags.clone().unwrap_or_default();
        let has_old = orig_flags.iter().any(|f| f.contains("LLAMA_CUBLAS"));
        let has_new = orig_flags.iter().any(|f| f.contains("GGML_CUDA"));
        let mut mapped_flags: Vec<String> =
            orig_flags.into_iter().filter(|f| !f.contains("LLAMA_CUBLAS")).collect();
        if has_old && !has_new {
            mapped_flags.push("-DGGML_CUDA=ON".to_string());
        }

        let mut cfgcmd = Command::new("cmake");
        cfgcmd
            .current_dir(&src_dir)
            .args(["-S", ".", "-B"])
            .arg(build_dir.to_string_lossy().to_string())
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg("-DLLAMA_BUILD_SERVER=ON");
        // Ensure any cached deprecated flag is removed
        cfgcmd.arg("-U").arg("LLAMA_CUBLAS");
        // If CUDA is requested but nvcc is not discoverable, provide CUDAToolkit_ROOT and PATH hints
        let wants_cuda_env = mapped_flags.iter().any(|f| f.contains("GGML_CUDA=ON"));
        let gpu_enabled = wants_cuda_env;
        if wants_cuda_env && which::which("nvcc").is_err() {
            if let Some(root) = discover_cuda_root() {
                let root_s = root.to_string_lossy().to_string();
                let bin = root.join("bin");
                let nvcc = bin.join("nvcc");
                // Log chosen CUDA root for troubleshooting
                println!("hinting CUDA root at {}", root_s);
                cfgcmd.env("CUDAToolkit_ROOT", &root_s);
                let new_path = match std::env::var_os("PATH") {
                    Some(p) => format!("{}:{}", bin.to_string_lossy(), p.to_string_lossy()),
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
        for f in &mapped_flags {
            cfgcmd.arg(f);
        }
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
                    .args(["-S", ".", "-B"])
                    .arg(build_dir.to_string_lossy().to_string())
                    .arg("-DCMAKE_BUILD_TYPE=Release")
                    .arg("-DLLAMA_BUILD_SERVER=ON")
                    .arg("-U")
                    .arg("LLAMA_CUBLAS")
                    .arg(format!("-DCMAKE_C_COMPILER={}", cc.display()))
                    .arg(format!("-DCMAKE_CXX_COMPILER={}", cxx.display()))
                    .arg(format!("-DCMAKE_CUDA_HOST_COMPILER={}", cc.display()));
                for f in &mapped_flags {
                    cfgcmd_hc.arg(f);
                }
                let st_hc = cfgcmd_hc.status().context("cmake configure (host-compiler)")?;
                if !st_hc.success() {
                    return Err(anyhow!(
                        "CUDA configure failed even with host compiler hint; GPU-only enforcement"
                    ));
                }
            } else {
                return Err(anyhow!("CUDA configure failed and no compatible host compiler found; GPU-only enforcement"));
            }
        } else if !status.success() {
            return Err(anyhow!("cmake configure failed"));
        }

        // Build
        let jobs = std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "4".to_string());
        cmd("cmake")
            .args(["--build", build_dir.to_string_lossy().as_ref(), "-j", &jobs])
            .status()
            .context("cmake build")
            .and_then(ok_status)?;

        let server_bin = build_dir.join("bin").join("llama-server");
        if !server_bin.exists() {
            return Err(anyhow!("llama-server not found at {}", server_bin.display()));
        }

        // Ensure model is present via model-provisioner
        let model_ref =
            prov.model.r#ref.clone().ok_or_else(|| anyhow!("provisioning.model.ref required"))?;
        let model_cache_dir = prov
            .model
            .cache_dir
            .clone()
            .unwrap_or_else(|| default_models_cache().to_string_lossy().to_string());
        let mp = model_provisioner::ModelProvisioner::file_only(PathBuf::from(&model_cache_dir))
            .context("init model-provisioner")?;
        let resolved = mp
            .ensure_present_str(&model_ref, None)
            .with_context(|| format!("ensuring model present for ref {}", model_ref))?;
        let model_path = resolved.local_path;

        // Spawn server process with deterministic flags and port selection
        let pid_dir = default_run_dir();
        std::fs::create_dir_all(&pid_dir).ok();
        let pid_file = pid_dir.join(format!("{}.pid", pool.id));

        let preferred_port = prov.ports.as_ref().and_then(|v| v.first().copied()).unwrap_or(8080);
        let port = select_listen_port(preferred_port);

        let mut cmdline = Command::new(&server_bin);
        cmdline.arg("--model").arg(&model_path).arg("--host").arg("127.0.0.1");
        cmdline.arg("--port").arg(port.to_string());

        // Normalize legacy flags and CPU/GPU expectations
        let mut applied_flags: Vec<String> = Vec::new();
        if let Some(flags) = &prov.flags {
            let norm = normalize_llamacpp_flags(flags, gpu_enabled);
            for f in norm {
                applied_flags.push(f);
            }
        }
        // Enforce deterministic defaults for tests (OwnerC-DET-FLAGS)
        ensure_flag_pair(&mut applied_flags, "--parallel", "1");
        ensure_flag(&mut applied_flags, "--no-cont-batching");
        ensure_flag(&mut applied_flags, "--no-webui");
        ensure_flag(&mut applied_flags, "--metrics");
        for f in &applied_flags { cmdline.arg(f); }

        // Capture stdout/stderr to log file
        let run_dir = default_run_dir();
        std::fs::create_dir_all(&run_dir).ok();
        let log_path = run_dir.join(format!("llamacpp-{}.log", pool.id));
        let log = OpenOptions::new().create(true).append(true).open(&log_path)
            .with_context(|| format!("open log file {}", log_path.display()))?;
        let log_err = OpenOptions::new().create(true).append(true).open(&log_path)
            .with_context(|| format!("open log file {}", log_path.display()))?;
        cmdline.stdout(Stdio::from(log));
        cmdline.stderr(Stdio::from(log_err));

        let mut child = cmdline
            .spawn()
            .with_context(|| format!("spawning llama-server for pool {}", pool.id))?;
        std::fs::write(&pid_file, child.id().to_string()).ok();
        println!("spawned llama-server pid={} (pool={})", child.id(), pool.id);

        // Readiness/health wait (OwnerC-HEALTH)
        let url = format!("http://127.0.0.1:{}", port);
        match wait_for_health("127.0.0.1", port, Duration::from_secs(120)) {
            Ok(()) => {
                // Metrics sanity (OwnerC-METRICS)
                if !http_ok("127.0.0.1", port, "/metrics").unwrap_or(false) {
                    // Not fatal for MVP if server exposes metrics lazily; warn via stderr
                    eprintln!("warning: /metrics not reachable at {} (flag --metrics set)", url);
                }
                // Prefer server-provided version when available
                let engine_version = try_fetch_engine_version("127.0.0.1", port)
                    .unwrap_or_else(|| {
                        format!(
                            "llamacpp-source:{}{}",
                            src.r#ref,
                            if gpu_enabled { "-cuda" } else { "-cpu" }
                        )
                    });
                let handoff = serde_json::json!({
                    "engine": "llamacpp",
                    "engine_version": engine_version,
                    "provisioning_mode": "source",
                    "url": url,
                    "pool_id": pool.id,
                    "replica_id": "r0",
                    "model": { "id": model_ref, "path": model_path },
                    "flags": applied_flags,
                });
                let handoff_path = write_handoff_file("llamacpp.json", &handoff)?;
                println!("wrote handoff {}", handoff_path.display());
                Ok(())
            }
            Err(e) => {
                // Clean up child and pid on failure
                let _ = child.kill();
                let _ = std::fs::remove_file(&pid_file);
                Err(e)
            }
        }
    }
}

fn preflight_tools(prov: &cfg::ProvisioningConfig, src: &cfg::SourceConfig) -> Result<()> {
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
        let compat = find_compat_host_compiler();
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
        for p in &pkgs {
            c.arg(p);
        }
        c.status().context("pacman -S")?
    } else if which::which("sudo").is_ok() {
        // Try non-interactive first
        let mut c = Command::new("sudo");
        c.args(["-n", "pacman", "-S", "--needed", "--noconfirm"]);
        for p in &pkgs {
            c.arg(p);
        }
        let st = c.status().context("sudo -n pacman -S")?;
        if st.success() {
            st
        } else {
            // Fallback: interactive prompt (user may enter password in terminal)
            let mut ci = Command::new("sudo");
            ci.args(["pacman", "-S", "--needed", "--noconfirm"]);
            for p in &pkgs {
                ci.arg(p);
            }
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

fn try_fetch_engine_version(host: &str, port: u16) -> Option<String> {
    // Best-effort: GET /version and parse JSON body for version-like fields.
    use std::net::TcpStream;
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(addr).ok()?;
    let req = format!(
        "GET /version HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        host
    );
    let _ = std::io::Write::write_all(&mut stream, req.as_bytes());
    let mut buf = Vec::new();
    let _ = std::io::Read::read_to_end(&mut stream, &mut buf);
    let text = String::from_utf8_lossy(&buf);
    if let Some(idx) = text.find("\r\n\r\n") {
        let body = &text[idx + 4..];
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
            if let Some(s) = v.get("version").and_then(|x| x.as_str()) {
                return Some(s.to_string());
            }
            if let Some(s) = v.get("git_describe").and_then(|x| x.as_str()) {
                return Some(s.to_string());
            }
            if let Some(s) = v.get("build").and_then(|x| x.as_str()) {
                return Some(s.to_string());
            }
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
        let is_pair =
            |s: &str| s == "--ngl" || s == "-ngl" || s == "--gpu-layers" || s == "--n-gpu-layers";
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

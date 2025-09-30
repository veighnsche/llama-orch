//! Llama.cpp engine provisioner (source mode)
// Modularized into submodules: preflight, toolchain, flags, version.

mod preflight;
mod toolchain;
mod flags;
mod version;

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

use preflight::preflight_tools;
use toolchain::{discover_cuda_root, find_compat_host_compiler};
use flags::normalize_llamacpp_flags;
use version::try_fetch_engine_version;

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
            if cache.exists() { let _ = std::fs::remove_file(&cache); }
            let cmfiles = build_dir.join("CMakeFiles");
            if cmfiles.exists() { let _ = std::fs::remove_dir_all(&cmfiles); }
        }
        // Map deprecated flags for newer llama.cpp: remove any -DLLAMA_CUBLAS* and use -DGGML_CUDA=ON
        let orig_flags = src.build.cmake_flags.clone().unwrap_or_default();
        let has_old = orig_flags.iter().any(|f| f.contains("LLAMA_CUBLAS"));
        let has_new = orig_flags.iter().any(|f| f.contains("GGML_CUDA"));
        let mut mapped_flags: Vec<String> = orig_flags.into_iter().filter(|f| !f.contains("LLAMA_CUBLAS")).collect();
        if has_old && !has_new { mapped_flags.push("-DGGML_CUDA=ON".to_string()); }

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
                println!("hinting CUDA root at {}", root_s);
                cfgcmd.env("CUDAToolkit_ROOT", &root_s);
                let new_path = match std::env::var_os("PATH") {
                    Some(p) => format!("{}:{}", bin.to_string_lossy(), p.to_string_lossy()),
                    None => bin.to_string_lossy().to_string(),
                };
                cfgcmd.env("PATH", new_path);
                cfgcmd.env("CUDA_HOME", &root_s);
                cfgcmd.env("CUDA_PATH", &root_s);
                cfgcmd.env("CUDA_TOOLKIT_ROOT_DIR", &root_s);
                cfgcmd.env("CUDACXX", nvcc.to_string_lossy().to_string());
                cfgcmd.arg(format!("-DCUDAToolkit_ROOT={}", root_s));
                cfgcmd.arg(format!("-DCUDA_TOOLKIT_ROOT_DIR={}", root_s));
                cfgcmd.arg(format!("-DCMAKE_CUDA_COMPILER={}", nvcc.to_string_lossy()));
            }
        }
        for f in &mapped_flags { cfgcmd.arg(f); }
        let status = cfgcmd.status().context("cmake configure")?;
        if !status.success() && wants_cuda_env {
            if let Some((cc, cxx)) = find_compat_host_compiler() {
                eprintln!("warning: CUDA configure failed; retrying with host compiler CC={} CXX={}", cc.display(), cxx.display());
                let mut cfgcmd_hc = Command::new("cmake");
                cfgcmd_hc
                    .current_dir(&src_dir)
                    .args(["-S", ".", "-B"])
                    .arg(build_dir.to_string_lossy().to_string())
                    .arg("-DCMAKE_BUILD_TYPE=Release")
                    .arg("-DLLAMA_BUILD_SERVER=ON")
                    .arg("-U").arg("LLAMA_CUBLAS")
                    .arg(format!("-DCMAKE_C_COMPILER={}", cc.display()))
                    .arg(format!("-DCMAKE_CXX_COMPILER={}", cxx.display()))
                    .arg(format!("-DCMAKE_CUDA_HOST_COMPILER={}", cc.display()));
                for f in &mapped_flags { cfgcmd_hc.arg(f); }
                let st_hc = cfgcmd_hc.status().context("cmake configure (host-compiler)")?;
                if !st_hc.success() {
                    return Err(anyhow!("CUDA configure failed even with host compiler hint; GPU-only enforcement"));
                }
            } else {
                return Err(anyhow!("CUDA configure failed and no compatible host compiler found; GPU-only enforcement"));
            }
        } else if !status.success() {
            return Err(anyhow!("cmake configure failed"));
        }

        // Build
        let jobs = std::thread::available_parallelism().map(|n| n.get().to_string()).unwrap_or_else(|_| "4".to_string());
        cmd("cmake").args(["--build", build_dir.to_string_lossy().as_ref(), "-j", &jobs]).status().context("cmake build").and_then(ok_status)?;

        let server_bin = build_dir.join("bin").join("llama-server");
        if !server_bin.exists() { return Err(anyhow!("llama-server not found at {}", server_bin.display())); }

        // Ensure model is present via model-provisioner
        let model_ref = prov.model.r#ref.clone().ok_or_else(|| anyhow!("provisioning.model.ref required"))?;
        let model_cache_dir = prov.model.cache_dir.clone().unwrap_or_else(|| default_models_cache().to_string_lossy().to_string());
        let mp = model_provisioner::ModelProvisioner::file_only(PathBuf::from(&model_cache_dir)).context("init model-provisioner")?;
        let resolved = mp.ensure_present_str(&model_ref, None).with_context(|| format!("ensuring model present for ref {}", model_ref))?;
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
            for f in norm { applied_flags.push(f); }
        }
        // Enforce deterministic defaults for tests
        ensure_flag_pair(&mut applied_flags, "--parallel", "1");
        ensure_flag(&mut applied_flags, "--no-cont-batching");
        ensure_flag(&mut applied_flags, "--no-webui");
        ensure_flag(&mut applied_flags, "--metrics");
        for f in &applied_flags { cmdline.arg(f); }

        // Capture stdout/stderr to log file
        let run_dir = default_run_dir();
        std::fs::create_dir_all(&run_dir).ok();
        let log_path = run_dir.join(format!("llamacpp-{}.log", pool.id));
        let log = OpenOptions::new().create(true).append(true).open(&log_path).with_context(|| format!("open log file {}", log_path.display()))?;
        let log_err = OpenOptions::new().create(true).append(true).open(&log_path).with_context(|| format!("open log file {}", log_path.display()))?;
        cmdline.stdout(Stdio::from(log));
        cmdline.stderr(Stdio::from(log_err));

        let mut child = cmdline.spawn().with_context(|| format!("spawning llama-server for pool {}", pool.id))?;
        std::fs::write(&pid_file, child.id().to_string()).ok();
        println!("spawned llama-server pid={} (pool={})", child.id(), pool.id);

        // Readiness/health wait
        let url = format!("http://127.0.0.1:{}", port);
        match wait_for_health("127.0.0.1", port, std::time::Duration::from_secs(120)) {
            Ok(()) => {
                if !http_ok("127.0.0.1", port, "/metrics").unwrap_or(false) {
                    eprintln!("warning: /metrics not reachable at {} (flag --metrics set)", url);
                }
                let engine_version = try_fetch_engine_version("127.0.0.1", port)
                    .unwrap_or_else(|| format!("llamacpp-source:{}{}", src.r#ref, if gpu_enabled { "-cuda" } else { "-cpu" }));
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
                // TODO[ENGINE-PROV-POOL-NOTIFY-0003]: Notify pool-managerd registry that pool is Live+Ready
                // and publish engine metadata (engine_version, slots_total/free, device_mask). Consider calling
                // an orchestrator control endpoint (e.g., /v1/workers/register) with Bearer auth so AdapterHost
                // can bind the llama.cpp HTTP adapter automatically using the handoff URL.
                // TODO[ENGINE-PROV-CLEANUP-0004]: On failure paths, ensure process is killed and handoff/pid files
                // are removed to avoid stale Ready signals.
                Ok(())
            }
            Err(e) => {
                let _ = child.kill();
                let _ = std::fs::remove_file(&pid_file);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaCppSourceProvisioner;
    use crate::{cfg, EngineProvisioner};

    fn base_pool() -> cfg::PoolConfig {
        cfg::PoolConfig {
            id: "p1".to_string(),
            engine: cfg::Engine::Llamacpp,
            model: "m1".to_string(),
            quant: None,
            ctx: None,
            devices: vec![0],
            tensor_split: None,
            preload: None,
            require_same_engine_version: None,
            sampler_profile_version: None,
            provisioning: cfg::ProvisioningConfig::default(),
            queue: cfg::QueueConfig::default(),
            admission: cfg::AdmissionConfig::default(),
            timeouts: cfg::Timeouts::default(),
        }
    }

    #[test]
    fn plan_without_source_has_minimal_steps() {
        let pool = base_pool();
        let prov = LlamaCppSourceProvisioner::new();
        let plan = prov.plan(&pool).unwrap();
        let kinds: Vec<_> = plan.steps.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(kinds, vec!["preflight-tools", "model-fetch", "run"]);
        // Basic detail presence
        assert!(plan.steps[0].detail.contains("allow_package_installs"));
        assert!(plan.steps[1].detail.contains("ref="));
        assert!(plan.steps[2].detail.contains("ports="));
    }

    #[test]
    fn plan_with_source_includes_git_and_cmake_steps() {
        let mut pool = base_pool();
        pool.provisioning.source = Some(cfg::SourceConfig {
            repo: "https://github.com/ggml-org/llama.cpp".to_string(),
            r#ref: "v0".to_string(),
            submodules: Some(false),
            build: cfg::SourceBuildConfig {
                cmake_flags: Some(vec!["-DGGML_CUDA=ON".into()]),
                generator: Some("Ninja".into()),
                cache_dir: Some("/tmp/cache".into()),
            },
        });
        pool.provisioning.model.r#ref = Some("hf:org/model.gguf".into());
        pool.provisioning.ports = Some(vec![9999]);
        pool.provisioning.flags = Some(vec!["--ngl".into(), "35".into()]);

        let prov = LlamaCppSourceProvisioner::new();
        let plan = prov.plan(&pool).unwrap();
        let kinds: Vec<_> = plan.steps.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(kinds, vec![
            "preflight-tools", "git-clone", "cmake-configure", "cmake-build", "model-fetch", "run"
        ]);
        // Verify some details
        assert!(plan.steps[1].detail.contains("repo="));
        assert!(plan.steps[1].detail.contains("ref="));
        assert!(plan.steps[2].detail.contains("flags="));
        assert!(plan.steps[2].detail.contains("generator="));
        assert!(plan.steps[2].detail.contains("cache_dir="));
        assert!(plan.steps[5].detail.contains("ports="));
        assert!(plan.steps[5].detail.contains("flags="));
    }

    #[test]
    fn ensure_errors_without_source_config() {
        let pool = base_pool();
        let prov = LlamaCppSourceProvisioner::new();
        let err = prov.ensure(&pool).unwrap_err();
        assert!(err.to_string().contains("provisioning.source required"));
    }
}


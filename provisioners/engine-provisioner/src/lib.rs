//! Engine provisioner (source/container/package/binary) — scaffold

use anyhow::Result;
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

pub use contracts_config_schema as cfg;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub kind: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Plan {
    pub pool_id: String,
    pub steps: Vec<PlanStep>,
}

pub trait EngineProvisioner {
    fn plan(&self, pool: &cfg::PoolConfig) -> Result<Plan>;
    fn ensure(&self, pool: &cfg::PoolConfig) -> Result<()>;
}

/// Source-mode provisioner (git + cmake/make) — scaffold
pub struct SourceProvisioner;

impl SourceProvisioner {
    pub fn new() -> Self {
        Self
    }
}

impl EngineProvisioner for SourceProvisioner {
    fn plan(&self, pool: &cfg::PoolConfig) -> Result<Plan> {
        let mut plan = Plan {
            pool_id: pool.id.clone(),
            steps: Vec::new(),
        };
        if let Some(prov) = Some(&pool.provisioning) {
            // Repo/ref
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
                plan.steps.push(PlanStep {
                    kind: "cmake-build".into(),
                    detail: "build llama-server".into(),
                });
            }
            // Model
            plan.steps.push(PlanStep {
                kind: "model-fetch".into(),
                detail: format!(
                    "ref={:?} cache_dir={:?}",
                    prov.model.r#ref, prov.model.cache_dir
                ),
            });
            // Run
            plan.steps.push(PlanStep {
                kind: "run".into(),
                detail: format!(
                    "ports={:?} flags={:?}",
                    prov.ports, prov.flags
                ),
            });
        }
        Ok(plan)
    }

    fn ensure(&self, pool: &cfg::PoolConfig) -> Result<()> {
        let prov = &pool.provisioning;
        let src = prov
            .source
            .as_ref()
            .ok_or_else(|| anyhow!("provisioning.source required for source mode"))?;

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
        let mut cfg = Command::new("cmake");
        cfg.current_dir(&src_dir)
            .args(["-S", ".", "-B"]) // -B will be followed by path
            .arg(build_dir.to_string_lossy().to_string())
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg("-DLLAMA_BUILD_SERVER=ON");
        if let Some(flags) = &src.build.cmake_flags {
            for f in flags {
                cfg.arg(f);
            }
        }
        cfg.status().context("cmake configure")?.success().then_some(()).ok_or_else(|| anyhow!("cmake configure failed"))?;

        // Build
        let jobs = std::thread::available_parallelism().map(|n| n.get().to_string()).unwrap_or_else(|_| "4".to_string());
        cmd("cmake")
            .args(["--build", build_dir.to_string_lossy().as_ref(), "-j", &jobs])
            .status()
            .context("cmake build")
            .and_then(ok_status)?;

        let server_bin = build_dir.join("bin").join("llama-server");
        if !server_bin.exists() {
            return Err(anyhow!(
                "llama-server not found at {}",
                server_bin.display()
            ));
        }

        // Ensure model is present
        let model_ref = prov
            .model
            .r#ref
            .clone()
            .ok_or_else(|| anyhow!("provisioning.model.ref required"))?;
        let model_cache_dir = prov
            .model
            .cache_dir
            .clone()
            .unwrap_or_else(|| default_models_cache().to_string_lossy().to_string());
        let model_path = resolve_model_path(&model_ref, Path::new(&model_cache_dir));
        if !model_path.exists() {
            // Best-effort fetch via huggingface-cli if hf: scheme
            if model_ref.starts_with("hf:") {
                if which::which("huggingface-cli").is_ok() {
                    let (repo, file) = parse_hf_ref(&model_ref).ok_or_else(|| anyhow!("invalid hf ref: {model_ref}"))?;
                    let mut c = Command::new("huggingface-cli");
                    c.env("HF_HUB_ENABLE_HF_TRANSFER", "1");
                    c.arg("download")
                        .arg(&repo)
                        .arg(&file)
                        .arg("--local-dir")
                        .arg(&model_cache_dir)
                        .arg("--local-dir-use-symlinks")
                        .arg("False");
                    c.status().context("huggingface-cli download")?.success().then_some(()).ok_or_else(|| anyhow!("model download failed"))?;
                } else {
                    return Err(anyhow!(
                        "model file not found at {} and huggingface-cli is missing",
                        model_path.display()
                    ));
                }
            } else {
                return Err(anyhow!(
                    "model file not found at {} (ref: {})",
                    model_path.display(),
                    model_ref
                ));
            }
        }

        // Spawn server process
        let pid_dir = default_run_dir();
        std::fs::create_dir_all(&pid_dir).ok();
        let pid_file = pid_dir.join(format!("{}.pid", pool.id));

        let mut cmdline = Command::new(server_bin);
        cmdline
            .arg("--model")
            .arg(&model_path)
            .arg("--host")
            .arg("127.0.0.1");
        // Use first configured port or default 8080
        let port = prov
            .ports
            .as_ref()
            .and_then(|v| v.get(0).cloned())
            .unwrap_or(8080);
        cmdline.arg("--port").arg(port.to_string());
        if let Some(flags) = &prov.flags {
            for f in flags {
                cmdline.arg(f);
            }
        }

        let child = cmdline
            .spawn()
            .with_context(|| format!("spawning llama-server for pool {}", pool.id))?;
        std::fs::write(&pid_file, child.id().to_string()).ok();
        println!("spawned llama-server pid={} (pool={})", child.id(), pool.id);
        Ok(())
    }
}

fn default_cache_dir(engine: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("llama-orch").join(engine)
}

fn default_models_cache() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("models")
}

fn default_run_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("llama-orch").join("run")
}

fn cmd(bin: &str) -> Command {
    Command::new(bin)
}

fn cmd_in(dir: &Path, bin: &str) -> Command {
    let mut c = Command::new(bin);
    c.current_dir(dir);
    c
}

fn ok_status(status: std::process::ExitStatus) -> Result<()> {
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("command failed with status {}", status))
    }
}

fn resolve_model_path(model_ref: &str, cache_dir: &Path) -> PathBuf {
    if model_ref.starts_with("hf:") {
        if let Some((_repo, file)) = parse_hf_ref(model_ref) {
            return cache_dir.join(file);
        }
    }
    if model_ref.starts_with("/") || model_ref.starts_with("./") {
        return PathBuf::from(model_ref);
    }
    cache_dir.join(model_ref)
}

fn parse_hf_ref(input: &str) -> Option<(String, String)> {
    // hf:owner/repo/file
    let s = input.strip_prefix("hf:")?;
    let mut parts = s.splitn(3, '/');
    let owner = parts.next()?.to_string();
    let repo = parts.next()?.to_string();
    let file = parts.next()?.to_string();
    Some((format!("{}/{}", owner, repo), file))
}

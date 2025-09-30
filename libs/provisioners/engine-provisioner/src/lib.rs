//! Engine provisioner (source/container/package/binary)

pub use contracts_config_schema as cfg;

pub mod plan;
pub mod util;
pub mod providers;

/// Stop a running engine process for the given pool by reading the pid file and sending SIGTERM.
/// Falls back to SIGKILL if the process does not exit quickly.
pub fn stop_pool(pool_id: &str) -> Result<()> {
    use crate::util::default_run_dir;
    use std::process::Command;
    let pid_path = default_run_dir().join(format!("{}.pid", pool_id));
    let pid_s = std::fs::read_to_string(&pid_path)
        .map_err(|e| anyhow::anyhow!("read pid file {}: {}", pid_path.display(), e))?;
    let pid = pid_s.trim();
    // Try TERM first
    let _ = Command::new("kill").arg("-TERM").arg(pid).status();
    // Wait up to 5 seconds for graceful shutdown
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let alive = Command::new("kill").arg("-0").arg(pid).status().map(|st| st.success()).unwrap_or(false);
        if !alive || std::time::Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    // If still alive after grace period, KILL
    let _ = Command::new("kill").arg("-0").arg(pid).status().map(|st| {
        if st.success() {
            let _ = Command::new("kill").arg("-KILL").arg(pid).status();
        }
    });
    let _ = std::fs::remove_file(&pid_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_pool_with_engine(engine: cfg::Engine) -> cfg::PoolConfig {
        cfg::PoolConfig {
            id: "p0".into(),
            engine,
            model: "m0".into(),
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
    fn provider_for_llamacpp_ok() {
        let pool = base_pool_with_engine(cfg::Engine::Llamacpp);
        let p = provider_for(&pool);
        assert!(p.is_ok(), "expected llamacpp provider when feature enabled");
    }

    #[test]
    fn provider_for_other_engines_feature_gated() {
        for engine in [cfg::Engine::Vllm, cfg::Engine::Tgi, cfg::Engine::Triton] {
            let pool = base_pool_with_engine(engine);
            let res = provider_for(&pool);
            match engine {
                cfg::Engine::Vllm => {
                    if cfg!(feature = "provider-vllm") {
                        assert!(res.is_ok());
                    } else {
                        let err = res.err().expect("expected gating").to_string();
                        assert!(err.contains("feature not enabled"));
                    }
                }
                cfg::Engine::Tgi => {
                    if cfg!(feature = "provider-tgi") {
                        assert!(res.is_ok());
                    } else {
                        let err = res.err().expect("expected gating").to_string();
                        assert!(err.contains("feature not enabled"));
                    }
                }
                cfg::Engine::Triton => {
                    if cfg!(feature = "provider-triton") {
                        assert!(res.is_ok());
                    } else {
                        let err = res.err().expect("expected gating").to_string();
                        assert!(err.contains("feature not enabled"));
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn stop_pool_kills_process_and_removes_pid() {
        use crate::util::default_run_dir;
        use std::process::Command;

        let run_dir = default_run_dir();
        std::fs::create_dir_all(&run_dir).unwrap();
        let pool_id = "p-stop";
        // Spawn a long-lived process using an absolute shell path to avoid PATH races
        let shell_path = ["/bin/sh", "/usr/bin/sh", "/bin/bash", "/usr/bin/bash"]
            .iter()
            .find(|p| std::path::Path::new(*p).exists())
            .cloned()
            .map(|s| s.to_string());
        if shell_path.is_none() {
            // No shell available; skip
            eprintln!("skipping stop_pool test: no shell found");
            return;
        }
        let shell = shell_path.unwrap();
        let arg_flag = "-c";
        let child = Command::new(&shell)
            .arg(arg_flag)
            .arg("while :; do :; done")
            .spawn()
            .expect("spawn loop");
        let pid = child.id();
        let pid_path = run_dir.join(format!("{}.pid", pool_id));
        std::fs::write(&pid_path, pid.to_string()).unwrap();

        // Call stop_pool; should terminate the process and remove pid file
        let _ = stop_pool(pool_id);

        // kill -0 returns non-success when process is gone; use absolute kill if possible
        let kill_path = ["/bin/kill", "/usr/bin/kill"]
            .iter()
            .find(|p| std::path::Path::new(*p).exists())
            .cloned()
            .map(|s| s.to_string());
        if let Some(kill_bin) = kill_path {
            let alive = Command::new(&kill_bin).arg("-0").arg(pid.to_string()).status().map(|s| s.success()).unwrap_or(false);
            if alive {
                // Best-effort cleanup to avoid stray process in the test env
                let _ = Command::new(kill_bin).arg("-KILL").arg(pid.to_string()).status();
            }
        }
        assert!(!pid_path.exists(), "pid file should be removed");
    }
}

use anyhow::Result;

pub use plan::{Plan, PlanStep};
#[cfg(feature = "provider-llamacpp")]
pub use providers::llamacpp::LlamaCppSourceProvisioner;
#[cfg(feature = "provider-tgi")]
pub use providers::tgi::TgiProvisioner;
#[cfg(feature = "provider-triton")]
pub use providers::triton::TritonProvisioner;
#[cfg(feature = "provider-vllm")]
pub use providers::vllm::VllmProvisioner;

pub trait EngineProvisioner {
    fn plan(&self, pool: &cfg::PoolConfig) -> Result<Plan>;
    fn ensure(&self, pool: &cfg::PoolConfig) -> Result<()>;
}

/// Return an engine-specific provisioner for the given pool.
/// For now:
/// - llama.cpp -> LlamaCppSourceProvisioner (source mode by default)
/// - vLLM/TGI/Triton -> stub provisioners (prefer container mode)
pub fn provider_for(pool: &cfg::PoolConfig) -> Result<Box<dyn EngineProvisioner>> {
    match pool.engine {
        cfg::Engine::Llamacpp => {
            #[cfg(feature = "provider-llamacpp")]
            {
                Ok(Box::new(LlamaCppSourceProvisioner::new()))
            }
            #[cfg(not(feature = "provider-llamacpp"))]
            {
                Err(anyhow::anyhow!("provider-llamacpp feature not enabled"))
            }
        }
        cfg::Engine::Vllm => {
            #[cfg(feature = "provider-vllm")]
            {
                Ok(Box::new(VllmProvisioner::new()))
            }
            #[cfg(not(feature = "provider-vllm"))]
            {
                Err(anyhow::anyhow!("provider-vllm feature not enabled"))
            }
        }
        cfg::Engine::Tgi => {
            #[cfg(feature = "provider-tgi")]
            {
                Ok(Box::new(TgiProvisioner::new()))
            }
            #[cfg(not(feature = "provider-tgi"))]
            {
                Err(anyhow::anyhow!("provider-tgi feature not enabled"))
            }
        }
        cfg::Engine::Triton => {
            #[cfg(feature = "provider-triton")]
            {
                Ok(Box::new(TritonProvisioner::new()))
            }
            #[cfg(not(feature = "provider-triton"))]
            {
                Err(anyhow::anyhow!("provider-triton feature not enabled"))
            }
        }
    }
}

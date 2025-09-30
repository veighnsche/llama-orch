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

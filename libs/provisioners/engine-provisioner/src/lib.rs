//! Engine provisioner (source/container/package/binary)
//!
//! Responsibility: PREPARE engines (download, build, stage model)
//! NOT responsible for: spawning, health monitoring, supervision (that's pool-managerd)

pub use contracts_config_schema as cfg;

pub mod plan;
pub mod providers;
pub mod util;

use std::path::PathBuf;

/// Metadata about a prepared engine, ready to be spawned by pool-managerd.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreparedEngine {
    /// Path to the engine binary (e.g., /path/to/llama-server)
    pub binary_path: PathBuf,
    /// Command-line flags to pass to the binary
    pub flags: Vec<String>,
    /// Port the engine should listen on
    pub port: u16,
    /// Host the engine should bind to (usually 127.0.0.1)
    pub host: String,
    /// Path to the staged model file
    pub model_path: PathBuf,
    /// Engine version string (e.g., "llamacpp-source:master-cuda")
    pub engine_version: String,
    /// Pool ID this engine is for
    pub pool_id: String,
    /// Replica ID (usually "r0")
    pub replica_id: String,
    /// Device mask (e.g., "GPU0" or "0,1")
    pub device_mask: Option<String>,
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
    /// Prepare the engine: download/build binary, stage model, return metadata.
    /// Does NOT spawn the process - that's pool-managerd's job.
    fn ensure(&self, pool: &cfg::PoolConfig) -> Result<PreparedEngine>;
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

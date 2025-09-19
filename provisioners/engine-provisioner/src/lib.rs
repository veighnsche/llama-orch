//! Engine provisioner (source/container/package/binary)

pub use contracts_config_schema as cfg;

pub mod plan;
pub mod util;
pub mod providers {
    #[cfg(feature = "provider-llamacpp")] pub mod llamacpp;
    #[cfg(feature = "provider-vllm")] pub mod vllm;
    #[cfg(feature = "provider-tgi")] pub mod tgi;
    #[cfg(feature = "provider-triton")] pub mod triton;
}

use anyhow::Result;

pub use plan::{Plan, PlanStep};
#[cfg(feature = "provider-llamacpp")] pub use providers::llamacpp::LlamaCppSourceProvisioner;
#[cfg(feature = "provider-vllm")] pub use providers::vllm::VllmProvisioner;
#[cfg(feature = "provider-tgi")] pub use providers::tgi::TgiProvisioner;
#[cfg(feature = "provider-triton")] pub use providers::triton::TritonProvisioner;

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
            #[cfg(feature = "provider-llamacpp")] { return Ok(Box::new(LlamaCppSourceProvisioner::new())); }
            #[cfg(not(feature = "provider-llamacpp"))] { return Err(anyhow::anyhow!("provider-llamacpp feature not enabled")); }
        }
        cfg::Engine::Vllm => {
            #[cfg(feature = "provider-vllm")] { return Ok(Box::new(VllmProvisioner::new())); }
            #[cfg(not(feature = "provider-vllm"))] { return Err(anyhow::anyhow!("provider-vllm feature not enabled")); }
        }
        cfg::Engine::Tgi => {
            #[cfg(feature = "provider-tgi")] { return Ok(Box::new(TgiProvisioner::new())); }
            #[cfg(not(feature = "provider-tgi"))] { return Err(anyhow::anyhow!("provider-tgi feature not enabled")); }
        }
        cfg::Engine::Triton => {
            #[cfg(feature = "provider-triton")] { return Ok(Box::new(TritonProvisioner::new())); }
            #[cfg(not(feature = "provider-triton"))] { return Err(anyhow::anyhow!("provider-triton feature not enabled")); }
        }
    }
}


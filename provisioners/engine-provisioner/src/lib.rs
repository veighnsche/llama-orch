//! Engine provisioner (source/container/package/binary)

pub use contracts_config_schema as cfg;

pub mod plan;
pub mod util;
pub mod providers {
    pub mod llamacpp;
    pub mod vllm;
    pub mod tgi;
    pub mod triton;
}

use anyhow::Result;

pub use plan::{Plan, PlanStep};
pub use providers::llamacpp::LlamaCppSourceProvisioner;
pub use providers::vllm::VllmProvisioner;
pub use providers::tgi::TgiProvisioner;
pub use providers::triton::TritonProvisioner;

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
        cfg::Engine::Llamacpp => Ok(Box::new(LlamaCppSourceProvisioner::new())),
        cfg::Engine::Vllm => Ok(Box::new(VllmProvisioner::new())),
        cfg::Engine::Tgi => Ok(Box::new(TgiProvisioner::new())),
        cfg::Engine::Triton => Ok(Box::new(TritonProvisioner::new())),
    }
}

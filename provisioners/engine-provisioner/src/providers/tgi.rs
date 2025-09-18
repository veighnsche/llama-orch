use anyhow::{anyhow, Result};

use crate::plan::{Plan, PlanStep};
use crate::{cfg, EngineProvisioner};

/// TGI (HuggingFace Text Generation Inference) provider (stub): prefer container mode
pub struct TgiProvisioner;

impl TgiProvisioner {
    pub fn new() -> Self {
        Self
    }
}

impl EngineProvisioner for TgiProvisioner {
    fn plan(&self, pool: &cfg::PoolConfig) -> Result<Plan> {
        let mut plan = Plan { pool_id: pool.id.clone(), steps: Vec::new() };
        plan.steps.push(PlanStep {
            kind: "todo".into(),
            detail: "TGI provisioning plan not implemented; prefer container mode".into(),
        });
        Ok(plan)
    }

    fn ensure(&self, _pool: &cfg::PoolConfig) -> Result<()> {
        Err(anyhow!("TGI provisioner not implemented yet; prefer container mode"))
    }
}

use anyhow::{anyhow, Result};

use crate::plan::{Plan, PlanStep};
use crate::{cfg, EngineProvisioner};

/// NVIDIA Triton provider (stub): prefer container mode
pub struct TritonProvisioner;

impl TritonProvisioner {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TritonProvisioner {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineProvisioner for TritonProvisioner {
    fn plan(&self, pool: &cfg::PoolConfig) -> Result<Plan> {
        let mut plan = Plan {
            pool_id: pool.id.clone(),
            steps: Vec::new(),
        };
        plan.steps.push(PlanStep {
            kind: "todo".into(),
            detail: "Triton provisioning plan not implemented; prefer container mode".into(),
        });
        Ok(plan)
    }

    fn ensure(&self, _pool: &cfg::PoolConfig) -> Result<()> {
        Err(anyhow!(
            "Triton provisioner not implemented yet; prefer container mode"
        ))
    }
}

use anyhow::{anyhow, Result};

use crate::plan::{Plan, PlanStep};
use crate::{cfg, EngineProvisioner};

/// vLLM provider (stub): prefer container mode; source-mode TBD
pub struct VllmProvisioner;

impl VllmProvisioner {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VllmProvisioner {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineProvisioner for VllmProvisioner {
    fn plan(&self, pool: &cfg::PoolConfig) -> Result<Plan> {
        let mut plan = Plan {
            pool_id: pool.id.clone(),
            steps: Vec::new(),
        };
        plan.steps.push(PlanStep {
            kind: "todo".into(),
            detail: "vLLM provisioning plan not implemented; prefer container mode".into(),
        });
        Ok(plan)
    }

    fn ensure(&self, _pool: &cfg::PoolConfig) -> Result<()> {
        Err(anyhow!(
            "vLLM provisioner not implemented yet; prefer container mode"
        ))
    }
}

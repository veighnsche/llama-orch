#[derive(Debug, Clone)]
pub struct PoolHealth {
    pub live: bool,
    pub ready: bool,
    pub draining: bool,
}

#[derive(Debug, Clone)]
pub enum ReloadOutcome {
    Replaced,
    Conflict,
}

pub trait PoolRegistry: Send + Sync {
    fn health(&self, pool_id: &str) -> anyhow::Result<PoolHealth>;
    fn drain(&self, pool_id: &str, deadline_ms: u64) -> anyhow::Result<()>;
    fn reload(&self, pool_id: &str, new_model_ref: &str) -> anyhow::Result<ReloadOutcome>;
}

//! Replica registry (planning-only).

use std::collections::HashMap;

use crate::health::HealthStatus;

#[derive(Debug, Clone)]
pub struct Replica;

#[derive(Debug, Clone, Default)]
pub struct Registry {
    pools: HashMap<String, HealthStatus>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    pub fn set_health(&mut self, pool_id: impl Into<String>, health: HealthStatus) {
        self.pools.insert(pool_id.into(), health);
    }

    pub fn get_health(&self, pool_id: &str) -> Option<HealthStatus> {
        self.pools.get(pool_id).cloned()
    }
}

//! pool-registry — Pool state management
//!
//! Tracks pool health, capacity, and worker status.

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::missing_errors_doc)]

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolHealth {
    pub live: bool,
    pub ready: bool,
    pub slots_total: u32,
    pub slots_free: u32,
}

pub struct PoolRegistry {
    pools: HashMap<String, PoolHealth>,
}

impl PoolRegistry {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }
    
    pub fn register_pool(&mut self, pool_id: String, health: PoolHealth) {
        self.pools.insert(pool_id, health);
    }
    
    pub fn get_health(&self, pool_id: &str) -> Option<&PoolHealth> {
        self.pools.get(pool_id)
    }
}

impl Default for PoolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

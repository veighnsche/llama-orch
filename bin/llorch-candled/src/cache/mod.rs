//! KV Cache module - top-level for future growth
//!
//! This is a TOP-LEVEL module to signal that cache will need
//! significant optimization work in the future (paged attention,
//! memory pooling, etc.). For MVP, keep implementation simple.
//!
//! Checkpoint 3 validation target
//!
//! Created by: TEAM-000
//! Modified by: TEAM-005 (Using candle_nn::kv_cache)

mod kv_cache;

// Re-export Candle's KV cache types
pub use kv_cache::{Cache, KvCache};

// Future: Add more cache strategies here
// mod paged_cache;
// mod rotating_cache;
// mod memory_pool;

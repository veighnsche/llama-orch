//! KV Cache module - top-level for future growth
//!
//! This is a TOP-LEVEL module to signal that cache will need
//! significant optimization work in the future (paged attention,
//! memory pooling, etc.). For MVP, keep implementation simple.
//!
//! Checkpoint 3 validation target
//!
//! Created by: TEAM-000

mod kv_cache;

pub use kv_cache::KVCache;

// Future: Add more cache strategies here
// mod paged_cache;
// mod rotating_cache;
// mod memory_pool;

//! Cache module - Using Candle's implementation
//!
//! TEAM-008: Discovered that candle-transformers already has a complete
//! unified Cache implementation that matches what we need. Using theirs
//! instead of maintaining our own.
//!
//! This is a TOP-LEVEL module to signal that cache will need
//! significant optimization work in the future (paged attention,
//! memory pooling, etc.). For MVP, keep implementation simple.
//!
//! Checkpoint 3 validation target
//!
//! Created by: TEAM-000
//! Modified by: TEAM-005 (Using candle_nn::kv_cache)
//! Modified by: TEAM-008 (Using candle-transformers Cache)

// TEAM-008: Use Candle's unified cache from transformers library
// This provides: kvs (per-layer KV), cos/sin (RoPE), masks (causal)
pub use candle_transformers::models::llama::Cache;

// Keep KV cache module for reference/compatibility
mod kv_cache;
pub use kv_cache::KvCache;

// Future: Add more cache strategies here
// mod paged_cache;
// mod rotating_cache;
// mod memory_pool;

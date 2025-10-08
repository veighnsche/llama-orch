//! Cache module - Using Candle's implementation
//!
//! TEAM-008: Discovered that candle-transformers already has a complete
//! unified Cache implementation that matches what we need. Using theirs
//! instead of maintaining our own.
//!
//! TEAM-010: Removed deprecated kv_cache.rs. All cache logic now handled
//! by candle-transformers::models::llama::Cache.
//!
//! Created by: TEAM-000
//! Modified by: TEAM-010 (Removed deprecated KvCache implementation)

// TEAM-008: Use Candle's unified cache from transformers library
// This provides: kvs (per-layer KV), cos/sin (RoPE), masks (causal)
pub use candle_transformers::models::llama::Cache;

//! KV Cache Module (Top-Level)
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 3 (KV Cache)
//!
//! Why top-level:
//! - Used by all 24 attention layers
//! - Future optimization target (paged attention)
//! - Signals engineering investment area

mod kv_cache;

pub use kv_cache::KVCache;

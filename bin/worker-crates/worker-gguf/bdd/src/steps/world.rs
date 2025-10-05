//! World state for worker-gguf BDD tests

use cucumber::World;
use worker_gguf::GGUFMetadata;

#[derive(Debug, Default, World)]
pub struct GGUFWorld {
    pub filename: Option<String>,
    pub metadata: Option<GGUFMetadata>,
}

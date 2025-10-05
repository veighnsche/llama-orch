//! World state for worker-gguf BDD tests

use cucumber::World;
// TODO: Uncomment after worker-gguf extraction
// use worker_gguf::GGUFMetadata;

#[derive(Debug, Default, World)]
pub struct GGUFWorld {
    pub filename: Option<String>,
    // TODO: Change to GGUFMetadata after extraction
    pub metadata: Option<()>,
}

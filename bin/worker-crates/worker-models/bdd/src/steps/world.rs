//! World state for worker-models BDD tests

use cucumber::World;
use worker_models::ModelType;

#[derive(Debug, Default, World)]
pub struct ModelsWorld {
    pub model_file: Option<String>,
    pub detected_architecture: Option<String>,
    pub model_type: Option<ModelType>,
    pub vocab_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub adapter_created: bool,
}

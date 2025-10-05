//! World state for worker-models BDD tests

use cucumber::World;

#[derive(Debug, Default, World)]
pub struct ModelsWorld {
    pub model_file: Option<String>,
    pub detected_architecture: Option<String>,
    pub adapter_created: bool,
}

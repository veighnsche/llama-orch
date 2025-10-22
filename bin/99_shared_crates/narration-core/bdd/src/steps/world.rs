// World state for BDD tests

use observability_narration_core::{CaptureAdapter, NarrationFields};

// CaptureAdapter doesn't implement Debug, so we manually implement it for World
#[derive(cucumber::World, Default)]
pub struct World {
    // Capture adapter for assertions
    pub adapter: Option<CaptureAdapter>,

    // Current narration fields being built
    pub fields: NarrationFields,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("adapter_present", &self.adapter.is_some())
            .field("fields", &self.fields)
            .finish()
    }
}

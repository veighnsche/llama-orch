// World state for BDD tests
// TEAM-307: Extended for context propagation, SSE, and job lifecycle tests

use observability_narration_core::{CaptureAdapter, NarrationContext, NarrationFields};
use std::collections::HashMap;

// CaptureAdapter doesn't implement Debug, so we manually implement it for World
#[derive(cucumber::World, Default)]
pub struct World {
    // Capture adapter for assertions
    pub adapter: Option<CaptureAdapter>,

    // Current narration fields being built
    pub fields: NarrationFields,
    
    // TEAM-307: Context propagation
    pub context: Option<NarrationContext>,
    pub outer_context: Option<NarrationContext>,
    pub inner_context: Option<NarrationContext>,
    pub context_a: Option<NarrationContext>,
    pub context_b: Option<NarrationContext>,
    
    // TEAM-307: Job lifecycle
    pub job_id: Option<String>,
    pub job_ids: Vec<String>,
    pub job_state: Option<String>,
    pub job_error: Option<String>,
    
    // TEAM-307: SSE streaming
    pub sse_channels: HashMap<String, bool>,
    pub sse_events: Vec<String>,
    
    // TEAM-307: Failure scenarios
    pub last_error: Option<String>,
    pub network_timeout_ms: Option<u64>,
    
    // TEAM-308: Per-scenario event tracking (fixes BUG-003)
    pub initial_event_count: usize,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("adapter_present", &self.adapter.is_some())
            .field("fields", &self.fields)
            .field("context_present", &self.context.is_some())
            .field("job_id", &self.job_id)
            .field("job_state", &self.job_state)
            .finish()
    }
}

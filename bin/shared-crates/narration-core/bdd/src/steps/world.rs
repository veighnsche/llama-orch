// World state for BDD tests

use observability_narration_core::{CaptureAdapter, NarrationFields, RedactionPolicy};
use std::collections::HashMap;

// CaptureAdapter doesn't implement Debug, so we manually implement it for World
#[derive(cucumber::World)]
#[derive(Default)]
pub struct World {
    // Capture adapter for assertions
    pub adapter: Option<CaptureAdapter>,

    // Current narration fields being built
    pub fields: NarrationFields,

    // Redaction policy for testing
    pub redaction_policy: Option<RedactionPolicy>,

    // HTTP headers for testing
    pub headers: HashMap<String, String>,

    // Extracted context from headers
    pub extracted_correlation_id: Option<String>,
    pub extracted_trace_id: Option<String>,
    pub extracted_span_id: Option<String>,
    pub extracted_parent_span_id: Option<String>,

    // Redaction test data
    pub redaction_input: String,
    pub redaction_output: String,

    // Service identity test data
    pub service_identity: String,

    // Timestamp test data
    pub timestamp_1: u64,
    pub timestamp_2: u64,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("adapter_present", &self.adapter.is_some())
            .field("fields", &self.fields)
            .field("redaction_policy_present", &self.redaction_policy.is_some())
            .field("headers_count", &self.headers.len())
            .finish()
    }
}


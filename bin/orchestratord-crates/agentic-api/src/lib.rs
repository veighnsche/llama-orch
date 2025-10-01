//! agentic-api — Agentic workflow endpoints

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

pub struct AgenticWorkflow;

impl AgenticWorkflow {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AgenticWorkflow {
    fn default() -> Self {
        Self::new()
    }
}

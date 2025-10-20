// TEAM-152: Created by TEAM-152
// TEAM-153: Enhanced with QueenHandle for cleanup testing
// Purpose: BDD test world state

use cucumber::World as CucumberWorld;
use rbee_keeper_queen_lifecycle::QueenHandle;

#[derive(Debug, CucumberWorld)]
pub struct World {
    /// Whether queen was already running before test
    pub queen_was_running: bool,
    
    /// Whether ensure_queen_running was called
    pub ensure_called: bool,
    
    /// Result of ensure_queen_running (now returns QueenHandle)
    pub ensure_result: Option<anyhow::Result<QueenHandle>>,
    
    /// The queen handle from ensure_queen_running
    pub queen_handle: Option<QueenHandle>,
    
    /// Output messages captured
    pub output_messages: Vec<String>,
    
    /// Whether shutdown was called
    pub shutdown_called: bool,
}

impl Default for World {
    fn default() -> Self {
        Self {
            queen_was_running: false,
            ensure_called: false,
            ensure_result: None,
            queen_handle: None,
            output_messages: Vec::new(),
            shutdown_called: false,
        }
    }
}

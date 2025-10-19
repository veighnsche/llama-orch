// TEAM-135: Created by TEAM-135 (BDD scaffolding)
//\! BDD World for queen-rbee integration tests

use cucumber::World;

#[derive(Debug, Default, World)]
pub struct BddWorld {
    /// Last validation result
    pub last_result: Option<Result<(), String>>,
    
    // TODO: Add integration test state fields here
    // e.g., HTTP client, process handles, temp directories
}

impl BddWorld {
    /// Store validation result
    pub fn store_result(&mut self, result: Result<(), String>) {
        self.last_result = Some(result);
    }

    /// Check if last validation succeeded
    pub fn last_succeeded(&self) -> bool {
        matches\!(self.last_result, Some(Ok(())))
    }

    /// Check if last validation failed
    pub fn last_failed(&self) -> bool {
        matches\!(self.last_result, Some(Err(_)))
    }
}

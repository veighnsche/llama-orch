// TEAM-153: Created by TEAM-153
// Purpose: Graceful shutdown endpoint for queen-rbee

use axum::http::StatusCode;
use observability_narration_core::{narrate, Narration};

const ACTOR_QUEEN_RBEE: &str = "queen-rbee";
const ACTION_SHUTDOWN: &str = "shutdown";

/// Handle POST /shutdown
///
/// Gracefully shuts down the queen-rbee server.
/// This is called by rbee-keeper when it started the queen and is done with it.
pub async fn handle_shutdown() -> StatusCode {
    narrate!(
        Narration::new(ACTOR_QUEEN_RBEE, ACTION_SHUTDOWN, "http-server")
            .human("Received shutdown request, exiting gracefully")
    );
    
    // Exit the process gracefully
    std::process::exit(0);
}

use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

use super::git::resolve_sha8;

/// Resolve the run id, honoring LLORCH_RUN_ID or generating a timestamp(-sha8) id.
pub fn resolve_run_id() -> String {
    if let Ok(id) = env::var("LLORCH_RUN_ID") {
        return id;
    }
    let ts = epoch_seconds();
    if let Some(sha8) = resolve_sha8() {
        format!("{}-{}", ts, sha8)
    } else {
        ts
    }
}

fn epoch_seconds() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("{}", secs)
}

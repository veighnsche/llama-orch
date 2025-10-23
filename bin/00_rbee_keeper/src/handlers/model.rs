//! Model command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-186: Use typed Operation enum instead of JSON strings
//! TEAM-187: Match on &action to avoid cloning hive_id multiple times

use anyhow::Result;
use rbee_operations::Operation;

use crate::cli::ModelAction;
use crate::job_client::submit_and_stream_job;

pub async fn handle_model(hive_id: String, action: ModelAction, queen_url: &str) -> Result<()> {
    let operation = match &action {
        ModelAction::Download { model } => {
            Operation::ModelDownload { hive_id, model: model.clone() }
        }
        ModelAction::List => Operation::ModelList { hive_id },
        ModelAction::Get { id } => Operation::ModelGet { hive_id, id: id.clone() },
        ModelAction::Delete { id } => Operation::ModelDelete { hive_id, id: id.clone() },
    };
    submit_and_stream_job(queen_url, operation).await
}

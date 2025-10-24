//! Model command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-186: Use typed Operation enum instead of JSON strings
//! TEAM-187: Match on &action to avoid cloning hive_id multiple times

use anyhow::Result;
use operations_contract::{
    ModelDeleteRequest, ModelDownloadRequest, ModelGetRequest, ModelListRequest, Operation,
}; // TEAM-284: Renamed from rbee_operations

use crate::cli::ModelAction;
use crate::job_client::submit_and_stream_job;

pub async fn handle_model(hive_id: String, action: ModelAction, queen_url: &str) -> Result<()> {
    // TEAM-284: Use typed requests
    let operation = match &action {
        ModelAction::Download { model } => Operation::ModelDownload(ModelDownloadRequest {
            hive_id,
            model: model.clone(),
        }),
        ModelAction::List => Operation::ModelList(ModelListRequest { hive_id }),
        ModelAction::Get { id } => Operation::ModelGet(ModelGetRequest {
            hive_id,
            id: id.clone(),
        }),
        ModelAction::Delete { id } => Operation::ModelDelete(ModelDeleteRequest {
            hive_id,
            id: id.clone(),
        }),
    };
    submit_and_stream_job(queen_url, operation).await
}

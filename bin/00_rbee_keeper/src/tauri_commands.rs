//! Tauri commands for the rbee-keeper GUI
//!
//! TEAM-293: Created Tauri command wrappers for all CLI operations
//!
//! Each command in this module corresponds to a CLI command and delegates
//! to the same handler functions used by the CLI.

use crate::cli::{HiveAction, ModelAction, QueenAction, WorkerAction, WorkerProcessAction};
use crate::config::Config;
use crate::handlers;
use anyhow::Result;
use serde::{Deserialize, Serialize};

// ============================================================================
// RESPONSE TYPES
// ============================================================================
// All Tauri commands return a Result<String, String> where:
// - Ok(String) = success message or JSON data
// - Err(String) = error message for display in GUI

#[derive(Serialize, Deserialize)]
pub struct CommandResponse {
    pub success: bool,
    pub message: String,
    pub data: Option<String>,
}

impl CommandResponse {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: None,
        }
    }

    pub fn success_with_data(message: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: Some(data.into()),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            data: None,
        }
    }
}

// Helper for handlers that return Result<()>
fn to_response_unit(result: Result<()>) -> Result<String, String> {
    match result {
        Ok(()) => Ok(serde_json::to_string(&CommandResponse::success(
            "Operation completed successfully",
        ))
        .unwrap()),
        Err(e) => Ok(serde_json::to_string(&CommandResponse::error(e.to_string())).unwrap()),
    }
}

// ============================================================================
// STATUS COMMANDS
// ============================================================================

#[tauri::command]
pub async fn get_status() -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_status(&queen_url).await;
    to_response_unit(result)
}

// ============================================================================
// QUEEN COMMANDS
// ============================================================================

#[tauri::command]
pub async fn queen_start() -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_queen(QueenAction::Start, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn queen_stop() -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_queen(QueenAction::Stop, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn queen_status() -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_queen(QueenAction::Status, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn queen_rebuild(with_local_hive: bool) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_queen(
        QueenAction::Rebuild { with_local_hive },
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn queen_info() -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_queen(QueenAction::Info, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn queen_install(binary: Option<String>) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_queen(QueenAction::Install { binary }, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn queen_uninstall() -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_queen(QueenAction::Uninstall, &queen_url).await;
    to_response_unit(result)
}

// ============================================================================
// HIVE COMMANDS
// ============================================================================

#[tauri::command]
pub async fn hive_install(
    host: String,
    binary: Option<String>,
    install_dir: Option<String>,
) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_hive(
        HiveAction::Install {
            host,
            binary,
            install_dir,
        },
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn hive_uninstall(host: String, install_dir: Option<String>) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_hive(
        HiveAction::Uninstall { host, install_dir },
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn hive_start(
    host: String,
    install_dir: Option<String>,
    port: u16,
) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_hive(
        HiveAction::Start {
            host,
            install_dir,
            port,
        },
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn hive_stop(host: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_hive(HiveAction::Stop { host }, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn hive_list() -> Result<String, String> {
    // TEAM-294: Return JSON for UI instead of using NARRATE
    use ssh_config::parse_ssh_config;
    
    let ssh_config_path = dirs::home_dir()
        .ok_or("Failed to get home directory")?
        .join(".ssh/config");

    let targets = parse_ssh_config(&ssh_config_path)
        .map_err(|e| e.to_string())?;

    // Return JSON for UI
    let response = CommandResponse {
        success: true,
        message: format!("Found {} SSH target(s)", targets.len()),
        data: Some(serde_json::to_string(&targets).map_err(|e| e.to_string())?),
    };

    serde_json::to_string(&response).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn hive_get(alias: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_hive(HiveAction::Get { alias }, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn hive_status(alias: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_hive(HiveAction::Status { alias }, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn hive_refresh_capabilities(alias: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_hive(HiveAction::RefreshCapabilities { alias }, &queen_url).await;
    to_response_unit(result)
}

// ============================================================================
// WORKER COMMANDS
// ============================================================================

#[tauri::command]
pub async fn worker_spawn(hive_id: String, model: String, device: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_worker(
        hive_id,
        WorkerAction::Spawn { model, device },
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn worker_process_list(hive_id: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_worker(
        hive_id,
        WorkerAction::Process(WorkerProcessAction::List),
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn worker_process_get(hive_id: String, pid: u32) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_worker(
        hive_id,
        WorkerAction::Process(WorkerProcessAction::Get { pid }),
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn worker_process_delete(hive_id: String, pid: u32) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_worker(
        hive_id,
        WorkerAction::Process(WorkerProcessAction::Delete { pid }),
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

// ============================================================================
// MODEL COMMANDS
// ============================================================================

#[tauri::command]
pub async fn model_download(hive_id: String, model: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_model(hive_id, ModelAction::Download { model }, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn model_list(hive_id: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_model(hive_id, ModelAction::List, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn model_get(hive_id: String, id: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_model(hive_id, ModelAction::Get { id }, &queen_url).await;
    to_response_unit(result)
}

#[tauri::command]
pub async fn model_delete(hive_id: String, id: String) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_model(hive_id, ModelAction::Delete { id }, &queen_url).await;
    to_response_unit(result)
}

// ============================================================================
// INFERENCE COMMANDS
// ============================================================================

#[tauri::command]
pub async fn infer(
    hive_id: String,
    model: String,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    device: Option<String>,
    worker_id: Option<String>,
    stream: Option<bool>,
) -> Result<String, String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let queen_url = config.queen_url();
    
    let result = handlers::handle_infer(
        hive_id,
        model,
        prompt,
        max_tokens.unwrap_or(20),
        temperature.unwrap_or(0.7),
        top_p,
        top_k,
        device,
        worker_id,
        stream.unwrap_or(true),
        &queen_url,
    )
    .await;
    to_response_unit(result)
}

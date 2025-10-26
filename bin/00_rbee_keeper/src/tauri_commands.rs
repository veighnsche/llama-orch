//! Tauri commands for the rbee-keeper GUI
//!
//! TEAM-293: Created Tauri command wrappers for all CLI operations
//! TEAM-297: Updated to use specta v2 for proper TypeScript type generation
//!
//! Each command in this module corresponds to a CLI command and delegates
//! to the same handler functions used by the CLI.

use crate::cli::{HiveAction, ModelAction, QueenAction, WorkerAction, WorkerProcessAction};
use crate::config::Config;
use crate::handlers;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use specta::Type;

#[cfg(test)]
mod tests {
    use super::*;
    use specta_typescript::Typescript;
    use tauri_specta::{collect_commands, Builder};

    #[test]
    fn export_typescript_bindings() {
        // TEAM-297: Test that exports TypeScript bindings
        let builder = Builder::<tauri::Wry>::new()
            .commands(collect_commands![hive_list]);
        
        builder
            .export(
                Typescript::default(),
                "ui/src/generated/bindings.ts",
            )
            .expect("Failed to export typescript bindings");
    }
}

// ============================================================================
// RESPONSE TYPES
// ============================================================================
// TEAM-296: Define types with Specta for proper TypeScript generation

/// SSH target from ~/.ssh/config
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SshTarget {
    /// Host alias from SSH config
    pub host: String,
    /// Host subtitle (optional)
    pub host_subtitle: Option<String>,
    /// Hostname (IP or domain)
    pub hostname: String,
    /// SSH username
    pub user: String,
    /// SSH port
    pub port: u16,
    /// Connection status
    pub status: SshTargetStatus,
}

/// SSH target connection status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Type)]
#[serde(rename_all = "lowercase")]
pub enum SshTargetStatus {
    Online,
    Offline,
    Unknown,
}

// Convert from ssh_config types to our types
impl From<ssh_config::SshTarget> for SshTarget {
    fn from(target: ssh_config::SshTarget) -> Self {
        Self {
            host: target.host,
            host_subtitle: target.host_subtitle,
            hostname: target.hostname,
            user: target.user,
            port: target.port,
            status: match target.status {
                ssh_config::SshTargetStatus::Online => SshTargetStatus::Online,
                ssh_config::SshTargetStatus::Offline => SshTargetStatus::Offline,
                ssh_config::SshTargetStatus::Unknown => SshTargetStatus::Unknown,
            },
        }
    }
}

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
#[specta::specta]
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    // TEAM-296: Return typed data with Specta for proper TypeScript generation
    // TEAM-309: Added narration for testing
    use observability_narration_core::n;
    use ssh_config::parse_ssh_config;
    
    n!("hive_list_start", "Reading SSH config for hive list");
    
    let ssh_config_path = dirs::home_dir()
        .ok_or("Failed to get home directory")?
        .join(".ssh/config");

    n!("ssh_config_path", "SSH config path: {}", ssh_config_path.display());

    let targets = parse_ssh_config(&ssh_config_path)
        .map_err(|e| {
            n!("ssh_config_error", "Failed to parse SSH config: {}", e);
            e.to_string()
        })?;

    n!("hive_list_parsed", "Found {} SSH targets", targets.len());

    // Convert from ssh_config::SshTarget to our SshTarget type
    let converted_targets: Vec<SshTarget> = targets
        .into_iter()
        .map(|t| t.into())
        .collect();

    n!("hive_list_complete", 
        human: "Hive list complete: {} targets",
        cute: "🐝 Found {} hives ready to work!",
        story: "The keeper discovered {} hives in the network",
        converted_targets.len()
    );

    Ok(converted_targets)
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

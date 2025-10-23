// TEAM-271: Worker spawning implementation
use crate::types::{SpawnResult, WorkerSpawnConfig};
use anyhow::{anyhow, Context, Result};
use daemon_lifecycle::DaemonManager;
use observability_narration_core::NarrationFactory;
use rbee_hive_artifact_catalog::Artifact;
use rbee_hive_worker_catalog::{Platform, WorkerCatalog, WorkerType};
use std::path::PathBuf;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

/// Spawn a worker process
///
/// This is a STATELESS operation - it just spawns the process and returns.
/// The worker will send heartbeats to the queen (not the hive).
///
/// # Architecture
///
/// 1. Determine worker type from device
/// 2. Find worker binary in catalog (or fallback to target/debug)
/// 3. Spawn worker process using daemon-lifecycle
/// 4. Return spawn result (PID, port, etc.)
/// 5. Worker sends heartbeat to queen (hive doesn't track it)
///
/// # Arguments
///
/// * `config` - Worker spawn configuration
///
/// # Returns
///
/// SpawnResult with worker_id, pid, port, binary_path
pub async fn spawn_worker(config: WorkerSpawnConfig) -> Result<SpawnResult> {
    NARRATE
        .action("worker_spawn_start")
        .job_id(&config.job_id)
        .context(&config.worker_id)
        .context(&config.model_id)
        .context(&config.device)
        .context(&config.port.to_string())
        .human("🚀 Spawning worker '{}' for model '{}' on device '{}' port {}")
        .emit();

    // Step 1: Determine worker type from device
    let worker_type = determine_worker_type(&config.device)?;

    NARRATE
        .action("worker_type_determined")
        .job_id(&config.job_id)
        .context(&format!("{:?}", worker_type))
        .context(&config.device)
        .human("Worker type: {:?} (device: {})")
        .emit();

    // Step 2: Find worker binary
    let binary_path = find_worker_binary(worker_type, &config.job_id)?;

    NARRATE
        .action("worker_binary_found")
        .job_id(&config.job_id)
        .context(&binary_path.display().to_string())
        .human("Worker binary: {}")
        .emit();

    // Step 3: Build command-line arguments
    let args = vec![
        "--worker-id".to_string(),
        config.worker_id.clone(),
        "--model".to_string(),
        config.model_id.clone(),
        "--device".to_string(),
        config.device.clone(),
        "--port".to_string(),
        config.port.to_string(),
        "--queen-url".to_string(),
        config.queen_url.clone(),
    ];

    NARRATE
        .action("worker_spawn_command")
        .job_id(&config.job_id)
        .context(&binary_path.display().to_string())
        .context(&args.join(" "))
        .human("Command: {} {}")
        .emit();

    // Step 4: Spawn worker process using daemon-lifecycle
    let manager = DaemonManager::new(binary_path.clone(), args).enable_auto_update(
        worker_type.binary_name(),
        format!("bin/30_{}", worker_type.binary_name().replace("-", "_")),
    );

    let child = manager.spawn().await.context("Failed to spawn worker process")?;

    // Get PID
    let pid = child.id().ok_or_else(|| anyhow!("Failed to get worker PID"))?;

    NARRATE
        .action("worker_spawned")
        .job_id(&config.job_id)
        .context(&config.worker_id)
        .context(&pid.to_string())
        .context(&config.port.to_string())
        .human("✅ Worker '{}' spawned (PID: {}, port: {})")
        .emit();

    // Step 5: Return spawn result
    // NOTE: We do NOT track the worker in hive - it will send heartbeat to queen
    Ok(SpawnResult {
        worker_id: config.worker_id,
        pid,
        port: config.port,
        binary_path: binary_path.display().to_string(),
    })
}

/// Determine worker type from device string
///
/// # Device Format
///
/// - "cpu" or "CPU-0" → CpuLlm
/// - "cuda:0" or "GPU-0" → CudaLlm
/// - "metal" → MetalLlm
fn determine_worker_type(device: &str) -> Result<WorkerType> {
    let device_lower = device.to_lowercase();

    if device_lower.starts_with("cpu") {
        Ok(WorkerType::CpuLlm)
    } else if device_lower.starts_with("cuda") || device_lower.starts_with("gpu") {
        Ok(WorkerType::CudaLlm)
    } else if device_lower.starts_with("metal") {
        Ok(WorkerType::MetalLlm)
    } else {
        Err(anyhow!("Unknown device type: '{}'. Supported: cpu, cuda:N, metal", device))
    }
}

/// Find worker binary
///
/// # Strategy
///
/// 1. Try worker catalog (production)
/// 2. Fallback to target/debug (development)
/// 3. Fallback to target/release (development)
fn find_worker_binary(worker_type: WorkerType, job_id: &str) -> Result<PathBuf> {
    // Try catalog first
    if let Ok(catalog) = WorkerCatalog::new() {
        if let Some(worker) = catalog.find_by_type_and_platform(worker_type, Platform::current()) {
            NARRATE
                .action("worker_binary_catalog")
                .job_id(job_id)
                .context(&worker.path().display().to_string())
                .human("Found worker in catalog: {}")
                .emit();

            return Ok(worker.path().to_path_buf());
        }
    }

    // Fallback to target directory (development)
    let binary_name = worker_type.binary_name();

    // Try debug first
    let debug_path = PathBuf::from("target/debug").join(binary_name);
    if debug_path.exists() {
        NARRATE
            .action("worker_binary_debug")
            .job_id(job_id)
            .context(&debug_path.display().to_string())
            .human("Using debug binary: {}")
            .emit();

        return Ok(debug_path);
    }

    // Try release
    let release_path = PathBuf::from("target/release").join(binary_name);
    if release_path.exists() {
        NARRATE
            .action("worker_binary_release")
            .job_id(job_id)
            .context(&release_path.display().to_string())
            .human("Using release binary: {}")
            .emit();

        return Ok(release_path);
    }

    Err(anyhow!("Worker binary '{}' not found in catalog or target directory", binary_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_worker_type() {
        assert_eq!(determine_worker_type("cpu").unwrap(), WorkerType::CpuLlm);
        assert_eq!(determine_worker_type("CPU-0").unwrap(), WorkerType::CpuLlm);
        assert_eq!(determine_worker_type("cuda:0").unwrap(), WorkerType::CudaLlm);
        assert_eq!(determine_worker_type("GPU-0").unwrap(), WorkerType::CudaLlm);
        assert_eq!(determine_worker_type("metal").unwrap(), WorkerType::MetalLlm);

        assert!(determine_worker_type("unknown").is_err());
    }
}

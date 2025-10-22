// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-189: Implemented basic HTTP server with /health endpoint
// TEAM-190: Added hive heartbeat task to send status to queen every 5 seconds
// TEAM-202: Replaced println!() with narration for job-scoped SSE visibility
// Purpose: rbee-hive binary entry point (DAEMON ONLY - NO CLI!)
// Status: Basic HTTP daemon with heartbeat and narration - ready for worker management implementation

//! rbee-hive
//!
//! Daemon for managing LLM worker instances on a single machine

mod narration;
use narration::{
    NARRATE, ACTION_STARTUP, ACTION_HEARTBEAT, ACTION_LISTEN, ACTION_READY,
    ACTION_CAPS_REQUEST, ACTION_CAPS_GPU_CHECK, ACTION_CAPS_GPU_FOUND,
    ACTION_CAPS_CPU_ADD, ACTION_CAPS_RESPONSE,
};

use axum::{routing::get, Router, Json};
use serde::Serialize;
use clap::Parser;
use rbee_heartbeat::{
    start_hive_heartbeat_task, HiveHeartbeatConfig, WorkerState, WorkerStateProvider,
};
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "rbee-hive")]
#[command(about = "rbee Hive Daemon - Worker and model management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "9000")]
    port: u16,

    /// TEAM-190: Hive ID (defaults to "localhost")
    #[arg(long, default_value = "localhost")]
    hive_id: String,

    /// TEAM-190: Queen URL for heartbeat reporting
    #[arg(long, default_value = "http://localhost:8500")]
    queen_url: String,
}

/// TEAM-190: Worker state provider for heartbeat aggregation
/// Currently returns empty list - will be populated when worker registry is implemented
struct HiveWorkerProvider;

impl WorkerStateProvider for HiveWorkerProvider {
    fn get_worker_states(&self) -> Vec<WorkerState> {
        // TEAM-190: Return empty list for now
        // TODO: Query worker registry when implemented
        vec![]
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // TEAM-202: Use narration instead of println!
    // This automatically goes through job-scoped SSE (if in job context)
    // and uses centralized formatting (TEAM-201)
    NARRATE
        .action(ACTION_STARTUP)
        .context(&args.port.to_string())
        .context(&args.hive_id)
        .context(&args.queen_url)
        .human("🐝 Starting on port {}, hive_id: {}, queen: {}")
        .emit();

    // TEAM-190: Start heartbeat task (5 second interval)
    let heartbeat_config = HiveHeartbeatConfig::new(
        args.hive_id.clone(),
        args.queen_url.clone(),
        "".to_string(), // Empty auth token for now
    )
    .with_interval(5); // 5 seconds as requested

    let worker_provider = Arc::new(HiveWorkerProvider);
    let _heartbeat_handle = start_hive_heartbeat_task(heartbeat_config, worker_provider);

    // TEAM-202: Narrate heartbeat startup
    NARRATE
        .action(ACTION_HEARTBEAT)
        .context("5s")
        .human("💓 Heartbeat task started ({} interval)")
        .emit();

    // Create router with health and capabilities endpoints
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/capabilities", get(get_capabilities));

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    // TEAM-202: Narrate listen address
    NARRATE
        .action(ACTION_LISTEN)
        .context(&format!("http://{}", addr))
        .human("✅ Listening on {}")
        .emit();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    // TEAM-202: Narrate ready state
    NARRATE
        .action(ACTION_READY)
        .human("✅ Hive ready")
        .emit();
    
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint
/// TEAM-189: Returns "ok" - used by hive start/stop/status operations
async fn health_check() -> &'static str {
    "ok"
}

/// TEAM-205: Device information for capabilities response
#[derive(Debug, Serialize)]
struct HiveDevice {
    id: String,
    name: String,
    device_type: String,
    vram_gb: Option<u32>,
    compute_capability: Option<String>,
}

/// TEAM-205: Capabilities response
#[derive(Debug, Serialize)]
struct CapabilitiesResponse {
    devices: Vec<HiveDevice>,
}

/// TEAM-205: Capabilities endpoint - returns detected GPU/CPU devices
/// TEAM-206: Added comprehensive narration for device detection visibility
async fn get_capabilities() -> Json<CapabilitiesResponse> {
    // TEAM-206: Narrate incoming request
    NARRATE
        .action(ACTION_CAPS_REQUEST)
        .human("📡 Received capabilities request from queen")
        .emit();
    
    // TEAM-206: Narrate GPU detection attempt
    NARRATE
        .action(ACTION_CAPS_GPU_CHECK)
        .human("🔍 Detecting GPUs via nvidia-smi...")
        .emit();
    
    // Detect GPUs
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    
    // TEAM-206: Narrate GPU detection results
    NARRATE
        .action(ACTION_CAPS_GPU_FOUND)
        .context(gpu_info.count.to_string())
        .human(if gpu_info.count > 0 {
            "✅ Found {} GPU(s)"
        } else {
            "ℹ️  No GPUs detected, using CPU only"
        })
        .emit();
    
    let mut devices: Vec<HiveDevice> = gpu_info.devices.iter().map(|gpu| HiveDevice {
        id: format!("GPU-{}", gpu.index),
        name: gpu.name.clone(),
        device_type: "gpu".to_string(),
        vram_gb: Some(gpu.vram_total_gb() as u32),
        compute_capability: Some(format!("{}.{}", gpu.compute_capability.0, gpu.compute_capability.1)),
    }).collect();
    
    // TEAM-209: Get actual CPU system information
    let cpu_cores = rbee_hive_device_detection::get_cpu_cores();
    let system_ram_gb = rbee_hive_device_detection::get_system_ram_gb();
    
    // TEAM-206: Narrate CPU fallback
    NARRATE
        .action(ACTION_CAPS_CPU_ADD)
        .context(cpu_cores.to_string())
        .context(system_ram_gb.to_string())
        .human("🖥️  Adding CPU-0: {0} cores, {1} GB RAM")
        .emit();
    
    // Add CPU device (always available) with actual system info
    devices.push(HiveDevice {
        id: "CPU-0".to_string(),
        name: format!("CPU ({} cores)", cpu_cores),
        device_type: "cpu".to_string(),
        vram_gb: Some(system_ram_gb), // System RAM for CPU device
        compute_capability: None,
    });
    
    // TEAM-206: Narrate response being sent
    NARRATE
        .action(ACTION_CAPS_RESPONSE)
        .context(devices.len().to_string())
        .human("📤 Sending capabilities response ({} device(s))")
        .emit();
    
    Json(CapabilitiesResponse { devices })
}

// TEAM-212: Hive HTTP client for capabilities discovery
//
// COPIED FROM: bin/10_queen_rbee/src/hive_client.rs
//
// This module provides HTTP client functions to fetch device capabilities
// from running rbee-hive instances.

use anyhow::{Context, Result};
use rbee_config::{DeviceInfo, DeviceType};
use serde::Deserialize;

/// Response from hive's /capabilities endpoint
#[derive(Debug, Deserialize)]
pub struct HiveCapabilitiesResponse {
    pub devices: Vec<HiveDevice>,
}

/// Device information from hive
#[derive(Debug, Deserialize)]
pub struct HiveDevice {
    pub id: String,
    pub name: String,
    pub device_type: String, // "gpu" or "cpu"
    pub vram_gb: Option<u32>,
    pub compute_capability: Option<String>,
}

/// Fetch capabilities from a running hive
///
/// # Arguments
/// * `endpoint` - Hive endpoint (e.g., "http://localhost:8081")
///
/// # Returns
/// * `Ok(Vec<DeviceInfo>)` - List of discovered devices
/// * `Err` - If hive is unreachable or returns invalid data
pub async fn fetch_hive_capabilities(endpoint: &str) -> Result<Vec<DeviceInfo>> {
    let url = format!("{}/capabilities", endpoint);

    // TEAM-207: Timeout handled by TimeoutEnforcer at call site
    let response = reqwest::get(&url).await.context("Failed to connect to hive")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Hive returned error: {} {}",
            response.status(),
            response.text().await.unwrap_or_default()
        );
    }

    let caps: HiveCapabilitiesResponse =
        response.json().await.context("Failed to parse capabilities response")?;

    // Convert to our format
    let devices = caps
        .devices
        .into_iter()
        .map(|d| DeviceInfo {
            id: d.id,
            name: d.name,
            vram_gb: d.vram_gb.unwrap_or(0),
            compute_capability: d.compute_capability,
            device_type: match d.device_type.as_str() {
                "gpu" => DeviceType::Gpu,
                "cpu" => DeviceType::Cpu,
                _ => DeviceType::Cpu, // Default to CPU
            },
        })
        .collect();

    Ok(devices)
}

/// Health check for hive
///
/// # Arguments
/// * `endpoint` - Hive endpoint (e.g., "http://localhost:8081")
///
/// # Returns
/// * `Ok(true)` - Hive is healthy
/// * `Ok(false)` - Hive returned non-success status
/// * `Err` - Connection failed
pub async fn check_hive_health(endpoint: &str) -> Result<bool> {
    let url = format!("{}/health", endpoint);

    // TEAM-207: Timeout handled by TimeoutEnforcer at call site
    let response = reqwest::get(&url).await.context("Failed to connect to hive")?;

    Ok(response.status().is_success())
}

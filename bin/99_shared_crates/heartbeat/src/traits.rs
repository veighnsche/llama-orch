//! Trait abstractions for heartbeat handlers
//!
//! Created by: TEAM-159
//!
//! These traits allow the heartbeat logic to be generic over different
//! implementations of registries, catalogs, and device detectors.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ============================================================================
// Worker Registry Trait (for Hive)
// ============================================================================

/// Trait for worker registry operations
///
/// Implemented by rbee-hive's WorkerRegistry
#[async_trait]
pub trait WorkerRegistry: Send + Sync {
    /// Update the last heartbeat timestamp for a worker
    ///
    /// Returns true if worker was found and updated, false otherwise
    async fn update_heartbeat(&self, worker_id: &str) -> bool;
}

// ============================================================================
// Hive Catalog Trait (for Queen)
// ============================================================================

/// Hive status enum (re-exported for trait)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HiveStatus {
    /// Hive registered but not yet connected
    Unknown,
    /// Hive is online and healthy
    Online,
    /// Hive is offline (missed heartbeats)
    Offline,
}

/// Device capabilities structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// CPU information
    pub cpu: Option<CpuDevice>,
    /// GPU list
    pub gpus: Vec<GpuDevice>,
}

/// CPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuDevice {
    /// Number of CPU cores
    pub cores: u32,
    /// System RAM in GB
    pub ram_gb: u32,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// GPU index
    pub index: u32,
    /// GPU name
    pub name: String,
    /// VRAM in GB
    pub vram_gb: u32,
    /// Backend (cuda, metal, cpu)
    pub backend: DeviceBackend,
}

/// Device backend enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceBackend {
    /// CUDA backend
    Cuda,
    /// Metal backend (macOS)
    Metal,
    /// CPU backend
    Cpu,
}

/// Hive record structure
#[derive(Debug, Clone)]
pub struct HiveRecord {
    /// Hive ID
    pub id: String,
    /// Host address
    pub host: String,
    /// Port number
    pub port: u16,
    /// Current status
    pub status: HiveStatus,
    /// Last heartbeat timestamp (milliseconds)
    pub last_heartbeat_ms: Option<i64>,
    /// Device capabilities
    pub devices: Option<DeviceCapabilities>,
}

/// Errors from catalog operations
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    /// Database error
    #[error("Database error: {0}")]
    Database(String),
    /// Not found error
    #[error("Not found: {0}")]
    NotFound(String),
    /// Other error
    #[error("{0}")]
    Other(String),
}

/// Trait for hive catalog operations
///
/// Implemented by queen-rbee's HiveCatalog
#[async_trait]
pub trait HiveCatalog: Send + Sync {
    /// Update the last heartbeat timestamp for a hive
    async fn update_heartbeat(&self, hive_id: &str, timestamp_ms: i64) -> Result<(), CatalogError>;

    /// Get hive information
    async fn get_hive(&self, hive_id: &str) -> Result<Option<HiveRecord>, CatalogError>;

    /// Update device capabilities for a hive
    async fn update_devices(
        &self,
        hive_id: &str,
        devices: DeviceCapabilities,
    ) -> Result<(), CatalogError>;

    /// Update hive status
    async fn update_hive_status(
        &self,
        hive_id: &str,
        status: HiveStatus,
    ) -> Result<(), CatalogError>;
}

// ============================================================================
// Device Detector Trait (for Queen)
// ============================================================================

/// Device detection response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceResponse {
    /// CPU information
    pub cpu: CpuInfo,
    /// GPU list
    pub gpus: Vec<GpuInfo>,
    /// Number of models
    pub models: usize,
    /// Number of workers
    pub workers: usize,
}

/// CPU information from device response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Number of cores
    pub cores: u32,
    /// RAM in GB
    pub ram_gb: u32,
}

/// GPU information from device response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU ID (e.g., "gpu0")
    pub id: String,
    /// GPU name
    pub name: String,
    /// VRAM in GB
    pub vram_gb: u32,
}

/// Errors from device detection
#[derive(Debug, thiserror::Error)]
pub enum DetectionError {
    /// HTTP request failed
    #[error("HTTP request failed: {0}")]
    Http(String),
    /// JSON parsing failed
    #[error("JSON parsing failed: {0}")]
    Parse(String),
    /// Other error
    #[error("{0}")]
    Other(String),
}

/// Trait for device detection
///
/// Implemented by queen-rbee's device detector
#[async_trait]
pub trait DeviceDetector: Send + Sync {
    /// Detect devices on a hive
    ///
    /// Makes HTTP GET request to hive's /v1/devices endpoint
    async fn detect_devices(&self, hive_url: &str) -> Result<DeviceResponse, DetectionError>;
}

// ============================================================================
// Narrator Trait (for Queen)
// ============================================================================

/// Trait for emitting narration events
///
/// Implemented by narration-core's Narration
pub trait Narrator: Send + Sync {
    /// Emit a narration event
    fn emit(&self, actor: &str, action: &str, target: &str, message: &str);

    /// Emit an error narration event
    fn emit_error(&self, actor: &str, action: &str, target: &str, message: &str, error_kind: &str);
}

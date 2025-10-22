// TEAM-210: Request/Response types for all hive operations
// TEAM-220: Investigated - 9 request/response pairs documented

use serde::{Deserialize, Serialize};

// ============================================================================
// INSTALL
// ============================================================================

/// Request to install a hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveInstallRequest {
    /// Hive alias from configuration
    pub alias: String,
}

/// Response from hive installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveInstallResponse {
    /// Whether installation was successful
    pub success: bool,
    /// Status message
    pub message: String,
    /// Path to the hive binary (if found)
    pub binary_path: Option<String>,
}

// ============================================================================
// UNINSTALL
// ============================================================================

/// Request to uninstall a hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveUninstallRequest {
    /// Hive alias from configuration
    pub alias: String,
}

/// Response from hive uninstallation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveUninstallResponse {
    /// Whether uninstallation was successful
    pub success: bool,
    /// Status message
    pub message: String,
}

// ============================================================================
// START
// ============================================================================

/// Request to start a hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStartRequest {
    /// Hive alias from configuration
    pub alias: String,
    /// Job ID for SSE routing (CRITICAL)
    pub job_id: String,
}

/// Response from hive start
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStartResponse {
    /// Whether start was successful
    pub success: bool,
    /// Status message
    pub message: String,
    /// Hive endpoint (host:port) if successful
    pub endpoint: Option<String>,
}

// ============================================================================
// STOP
// ============================================================================

/// Request to stop a hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStopRequest {
    /// Hive alias from configuration
    pub alias: String,
    /// Job ID for SSE routing (CRITICAL)
    pub job_id: String,
}

/// Response from hive stop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStopResponse {
    /// Whether stop was successful
    pub success: bool,
    /// Status message
    pub message: String,
}

// ============================================================================
// LIST
// ============================================================================

/// Request to list all hives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveListRequest {
    // No parameters needed
}

/// Response from hive list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveListResponse {
    /// List of all configured hives
    pub hives: Vec<HiveInfo>,
}

/// Information about a single hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveInfo {
    /// Hive alias
    pub alias: String,
    /// Hostname or IP address
    pub hostname: String,
    /// Hive port
    pub hive_port: u16,
    /// Path to hive binary (if configured)
    pub binary_path: Option<String>,
}

// ============================================================================
// GET
// ============================================================================

/// Request to get details about a specific hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveGetRequest {
    /// Hive alias from configuration
    pub alias: String,
}

/// Response from hive get
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveGetResponse {
    /// Details about the requested hive
    pub hive: HiveInfo,
}

// ============================================================================
// STATUS
// ============================================================================

/// Request to check hive status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStatusRequest {
    /// Hive alias from configuration
    pub alias: String,
    /// Job ID for SSE routing (CRITICAL)
    pub job_id: String,
}

/// Response from hive status check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStatusResponse {
    /// Hive alias
    pub alias: String,
    /// Whether hive is running
    pub running: bool,
    /// Health check URL
    pub health_url: String,
}

// ============================================================================
// REFRESH CAPABILITIES
// ============================================================================

/// Request to refresh hive capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveRefreshCapabilitiesRequest {
    /// Hive alias from configuration
    pub alias: String,
    /// Job ID for SSE routing (CRITICAL)
    pub job_id: String,
}

/// Response from capabilities refresh
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveRefreshCapabilitiesResponse {
    /// Whether refresh was successful
    pub success: bool,
    /// Number of devices discovered
    pub device_count: usize,
    /// Status message
    pub message: String,
}

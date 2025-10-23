// TEAM-210: Request/Response types for all hive operations
// TEAM-220: Investigated - 9 request/response pairs documented
// TEAM-276: Added HiveHandle for ensure pattern

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use serde::{Deserialize, Serialize};

const NARRATE: NarrationFactory = NarrationFactory::new("hive-lc");

// TEAM-278: DELETED HiveInstallRequest, HiveInstallResponse, HiveUninstallRequest, HiveUninstallResponse

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

// ============================================================================
// HANDLE (for ensure pattern)
// ============================================================================

/// Handle to a hive process
///
/// TEAM-276: Added for ensure pattern consistency with queen-lifecycle
///
/// Tracks whether queen started the hive and provides cleanup.
/// IMPORTANT: Only shuts down hive if queen started it!
#[derive(Debug)]
pub struct HiveHandle {
    /// True if queen started the hive (must cleanup)
    /// False if hive was already running (don't touch it)
    started_by_us: bool,

    /// Hive alias
    alias: String,

    /// Hive endpoint (e.g., "http://192.168.1.100:8600")
    endpoint: String,
}

impl HiveHandle {
    /// Create handle for hive that was already running
    pub const fn already_running(alias: String, endpoint: String) -> Self {
        Self {
            started_by_us: false,
            alias,
            endpoint,
        }
    }

    /// Create handle for hive that we just started
    pub const fn started_by_us(alias: String, endpoint: String) -> Self {
        Self {
            started_by_us: true,
            alias,
            endpoint,
        }
    }

    /// Check if we started the hive (and should clean it up)
    pub const fn should_cleanup(&self) -> bool {
        self.started_by_us
    }

    /// Get the hive's alias
    pub fn alias(&self) -> &str {
        &self.alias
    }

    /// Get the hive's endpoint
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Keep the hive alive (no shutdown after task)
    ///
    /// Hive stays running for future tasks.
    ///
    /// # Returns
    /// * `Ok(())` - Always succeeds (hive stays alive)
    ///
    /// # Errors
    ///
    /// Currently never returns an error
    pub fn shutdown(self) -> Result<()> {
        NARRATE
            .action("hive_keep_alive")
            .context(&self.alias)
            .human("Task complete, keeping hive '{}' alive for future tasks")
            .emit();
        Ok(())
    }
}

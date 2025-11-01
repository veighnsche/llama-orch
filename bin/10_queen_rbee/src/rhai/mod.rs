//! RHAI Script Management Module
//!
//! Provides handlers for RHAI script operations:
//! - Save: Create or update RHAI scripts
//! - Test: Execute RHAI scripts in sandbox
//! - Get: Retrieve RHAI script by ID
//! - List: List all RHAI scripts
//! - Delete: Delete RHAI script by ID
//!
//! All operations are job-based and use narration for logging.

pub mod save;
pub mod test;
pub mod get;
pub mod list;
pub mod delete;

pub use save::execute_rhai_script_save;
pub use test::execute_rhai_script_test;
pub use get::execute_rhai_script_get;
pub use list::execute_rhai_script_list;
pub use delete::execute_rhai_script_delete;

// TEAM-350: Config structs for #[with_job_id] macro
// The macro expects a config parameter with job_id: Option<String>

/// Config for RHAI test operation
pub struct RhaiTestConfig {
    /// Job ID for SSE routing
    pub job_id: Option<String>,
    /// Script content to test
    pub content: String,
}

/// Config for RHAI save operation
pub struct RhaiSaveConfig {
    /// Job ID for SSE routing
    pub job_id: Option<String>,
    /// Script name
    pub name: String,
    /// Script content
    pub content: String,
    /// Optional script ID (for updates)
    pub id: Option<String>,
}

/// Config for RHAI get operation
pub struct RhaiGetConfig {
    /// Job ID for SSE routing
    pub job_id: Option<String>,
    /// Script ID to retrieve
    pub id: String,
}

/// Config for RHAI list operation
pub struct RhaiListConfig {
    /// Job ID for SSE routing
    pub job_id: Option<String>,
}

/// Config for RHAI delete operation
pub struct RhaiDeleteConfig {
    /// Job ID for SSE routing
    pub job_id: Option<String>,
    /// Script ID to delete
    pub id: String,
}

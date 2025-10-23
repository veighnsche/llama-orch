//! Get daemon instance by ID
//!
//! TEAM-259: Extracted common get pattern
//!
//! Provides generic daemon retrieval functionality for:
//! - hive-lifecycle (get hive by alias)
//! - worker-lifecycle (get worker by ID)

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use serde::Serialize;

const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");

/// Trait for configurations that can get daemon instances by ID
///
/// Implement this trait to enable generic get functionality.
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::GettableConfig;
/// use serde::Serialize;
///
/// #[derive(Serialize, Clone)]
/// struct HiveInfo {
///     alias: String,
///     hostname: String,
/// }
///
/// struct HiveConfig {
///     hives: Vec<HiveInfo>,
/// }
///
/// impl GettableConfig for HiveConfig {
///     type Info = HiveInfo;
///     
///     fn get_by_id(&self, id: &str) -> Option<Self::Info> {
///         self.hives.iter()
///             .find(|h| h.alias == id)
///             .cloned()
///     }
///     
///     fn daemon_type_name(&self) -> &'static str {
///         "hive"
///     }
/// }
/// ```
pub trait GettableConfig {
    /// The info type returned for the daemon instance
    type Info: Serialize;

    /// Get daemon instance by ID
    fn get_by_id(&self, id: &str) -> Option<Self::Info>;

    /// Name of the daemon type (e.g., "hive", "worker")
    fn daemon_type_name(&self) -> &'static str;
}

/// Get a daemon instance by ID
///
/// Generic function that retrieves a specific daemon instance by its ID.
/// Uses the GettableConfig trait to find the instance in configuration.
///
/// # Arguments
/// * `config` - Configuration implementing GettableConfig
/// * `id` - ID of the daemon instance (e.g., alias, worker ID)
/// * `job_id` - Optional job ID for narration routing
///
/// # Returns
/// * `Ok(T::Info)` - Daemon instance info
/// * `Err` - Instance not found
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{GettableConfig, get_daemon};
///
/// # async fn example<T: GettableConfig>(config: &T) -> anyhow::Result<()> {
/// let instance = get_daemon(config, "my-hive", Some("job_123")).await?;
/// # Ok(())
/// # }
/// ```
pub async fn get_daemon<T: GettableConfig>(
    config: &T,
    id: &str,
    job_id: Option<&str>,
) -> Result<T::Info> {
    let daemon_type = config.daemon_type_name();

    let mut narration = NARRATE.action("daemon_get").context(daemon_type).context(id);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human(&format!("🔍 Getting {} '{{}}'", daemon_type)).emit();

    match config.get_by_id(id) {
        Some(info) => {
            // Convert to JSON for display
            let info_json = serde_json::to_value(&info).unwrap_or(serde_json::Value::Null);

            let mut narration = NARRATE.action("daemon_found").context(daemon_type).context(id);
            if let Some(jid) = job_id {
                narration = narration.job_id(jid);
            }
            narration.human(&format!("✅ Found {} '{{}}':", daemon_type)).table(&info_json).emit();

            Ok(info)
        }
        None => {
            let mut narration = NARRATE
                .action("daemon_not_found")
                .context(daemon_type)
                .context(id)
                .human(&format!("❌ {} '{{}}' not found", daemon_type))
                .error_kind("not_found");
            if let Some(jid) = job_id {
                narration = narration.job_id(jid);
            }
            narration.emit_error();

            anyhow::bail!("{} '{}' not found", daemon_type, id)
        }
    }
}

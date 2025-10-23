//! List daemon instances
//!
//! TEAM-259: Extracted common list pattern
//!
//! Provides generic daemon listing functionality for:
//! - hive-lifecycle (list all hives)
//! - worker-lifecycle (list all workers)

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use serde::Serialize;

const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");

/// Trait for configurations that can list daemon instances
///
/// Implement this trait to enable generic listing functionality.
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::ListableConfig;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct HiveInfo {
///     alias: String,
///     hostname: String,
/// }
///
/// struct HiveConfig {
///     hives: Vec<HiveInfo>,
/// }
///
/// impl ListableConfig for HiveConfig {
///     type Info = HiveInfo;
///     
///     fn list_all(&self) -> Vec<Self::Info> {
///         self.hives.clone()
///     }
///     
///     fn daemon_type_name(&self) -> &'static str {
///         "hive"
///     }
/// }
/// ```
pub trait ListableConfig {
    /// The info type returned for each daemon instance
    type Info: Serialize;

    /// List all daemon instances
    fn list_all(&self) -> Vec<Self::Info>;

    /// Name of the daemon type (e.g., "hive", "worker")
    fn daemon_type_name(&self) -> &'static str;
}

/// List all daemon instances
///
/// Generic function that lists all configured daemon instances.
/// Uses the ListableConfig trait to get instances from configuration.
///
/// # Arguments
/// * `config` - Configuration implementing ListableConfig
/// * `job_id` - Optional job ID for narration routing
///
/// # Returns
/// * `Ok(Vec<T::Info>)` - List of daemon instances
/// * `Err` - Configuration error
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{ListableConfig, list_daemons};
///
/// # async fn example<T: ListableConfig>(config: &T) -> anyhow::Result<()> {
/// let instances = list_daemons(config, Some("job_123")).await?;
/// println!("Found {} instances", instances.len());
/// # Ok(())
/// # }
/// ```
pub async fn list_daemons<T: ListableConfig>(
    config: &T,
    job_id: Option<&str>,
) -> Result<Vec<T::Info>> {
    let daemon_type = config.daemon_type_name();

    let mut narration = NARRATE.action("daemon_list").context(daemon_type);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human(&format!("📊 Listing all {}s", daemon_type)).emit();

    let instances = config.list_all();

    if instances.is_empty() {
        let mut narration = NARRATE.action("daemon_empty").context(daemon_type);
        if let Some(jid) = job_id {
            narration = narration.job_id(jid);
        }
        narration.human(&format!("No {}s registered", daemon_type)).emit();
    } else {
        // Convert to JSON for table display
        let instances_json: Vec<serde_json::Value> = instances
            .iter()
            .map(|i| serde_json::to_value(i).unwrap_or(serde_json::Value::Null))
            .collect();

        let mut narration = NARRATE
            .action("daemon_result")
            .context(instances.len().to_string())
            .context(daemon_type);
        if let Some(jid) = job_id {
            narration = narration.job_id(jid);
        }
        narration
            .human(&format!("Found {{}} {}(s):", daemon_type))
            .table(&serde_json::Value::Array(instances_json))
            .emit();
    }

    Ok(instances)
}

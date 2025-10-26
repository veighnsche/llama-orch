//! List daemon instances
//!
//! TEAM-259: Extracted common list pattern
//!
//! Provides generic daemon listing functionality for:
//! - hive-lifecycle (list all hives)
//! - worker-lifecycle (list all workers)

use anyhow::Result;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use serde::Serialize;

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
    // TEAM-311: Migrated to n!() macro
    let ctx = job_id.map(|jid| NarrationContext::new().with_job_id(jid));
    let daemon_type = config.daemon_type_name();

    let list_impl = async {
        n!("daemon_list", "📊 Listing all {}s", daemon_type);

        let instances = config.list_all();

        if instances.is_empty() {
            n!("daemon_empty", "No {}s registered", daemon_type);
        } else {
            // Convert to JSON for table display
            let instances_json: Vec<serde_json::Value> = instances
                .iter()
                .map(|i| serde_json::to_value(i).unwrap_or(serde_json::Value::Null))
                .collect();
            let table_str = serde_json::to_string_pretty(&serde_json::Value::Array(instances_json)).unwrap_or_default();

            n!("daemon_result", "Found {} {}(s):\n{}", instances.len(), daemon_type, table_str);
        }

        Ok(instances)
    };
    
    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, list_impl).await
    } else {
        list_impl.await
    }
}

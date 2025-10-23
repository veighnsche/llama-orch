//! State comparison (desired vs actual)
//!
//! Created by: TEAM-280
//!
//! Compares desired state (from config file) with actual state (installed components)
//! to determine what needs to be installed, removed, or is already correct.

use rbee_config::declarative::{HiveConfig, WorkerConfig};
use serde::{Deserialize, Serialize};

/// Difference between desired and actual state
///
/// TEAM-280: Used by sync and status operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDiff {
    /// Hives that need to be installed
    pub hives_to_install: Vec<HiveConfig>,

    /// Hives that are already installed correctly
    pub hives_already_installed: Vec<HiveConfig>,

    /// Workers that need to be installed (hive_alias, workers)
    pub workers_to_install: Vec<(String, Vec<WorkerConfig>)>,

    /// Workers that are already installed correctly (hive_alias, workers)
    pub workers_already_installed: Vec<(String, Vec<WorkerConfig>)>,

    /// Hives that should be removed (not in config)
    /// Only populated if `remove_extra` is true
    pub hives_to_remove: Vec<String>,

    /// Workers that should be removed (not in config)
    /// Only populated if `remove_extra` is true
    pub workers_to_remove: Vec<(String, Vec<String>)>,
}

impl StateDiff {
    /// Create a new empty diff
    pub fn new() -> Self {
        Self {
            hives_to_install: Vec::new(),
            hives_already_installed: Vec::new(),
            workers_to_install: Vec::new(),
            workers_already_installed: Vec::new(),
            hives_to_remove: Vec::new(),
            workers_to_remove: Vec::new(),
        }
    }

    /// Check if there are any changes needed
    pub fn has_changes(&self) -> bool {
        !self.hives_to_install.is_empty()
            || !self.workers_to_install.is_empty()
            || !self.hives_to_remove.is_empty()
            || !self.workers_to_remove.is_empty()
    }

    /// Get total number of changes
    pub fn change_count(&self) -> usize {
        self.hives_to_install.len()
            + self.workers_to_install.iter().map(|(_, w)| w.len()).sum::<usize>()
            + self.hives_to_remove.len()
            + self.workers_to_remove.iter().map(|(_, w)| w.len()).sum::<usize>()
    }
}

impl Default for StateDiff {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute difference between desired and actual state
///
/// TEAM-280: Core diff logic
///
/// # Arguments
/// * `desired` - Desired state from config file
/// * `actual_hives` - List of currently installed hive aliases
/// * `actual_workers` - Map of hive alias to list of installed worker types
/// * `remove_extra` - Whether to include extra components in diff
///
/// # Returns
/// * `StateDiff` - Difference between desired and actual state
pub fn compute_diff(
    desired: &[HiveConfig],
    actual_hives: &[String],
    actual_workers: &[(String, Vec<String>)],
    remove_extra: bool,
) -> StateDiff {
    let mut diff = StateDiff::new();

    // Convert actual state to sets for easier lookup
    let actual_hive_set: std::collections::HashSet<_> = actual_hives.iter().collect();
    let actual_worker_map: std::collections::HashMap<_, _> = actual_workers
        .iter()
        .map(|(alias, workers)| (alias.as_str(), workers))
        .collect();

    // Check each desired hive
    for hive in desired {
        if actual_hive_set.contains(&hive.alias) {
            // Hive exists - check workers
            diff.hives_already_installed.push(hive.clone());

            if let Some(installed_workers) = actual_worker_map.get(hive.alias.as_str()) {
                let installed_set: std::collections::HashSet<_> =
                    installed_workers.iter().map(|s| s.as_str()).collect();

                let mut workers_to_add = Vec::new();
                let mut workers_ok = Vec::new();

                for worker in &hive.workers {
                    if installed_set.contains(worker.worker_type.as_str()) {
                        workers_ok.push(worker.clone());
                    } else {
                        workers_to_add.push(worker.clone());
                    }
                }

                if !workers_to_add.is_empty() {
                    diff.workers_to_install.push((hive.alias.clone(), workers_to_add));
                }
                if !workers_ok.is_empty() {
                    diff.workers_already_installed.push((hive.alias.clone(), workers_ok));
                }

                // Check for extra workers (if remove_extra)
                if remove_extra {
                    let desired_set: std::collections::HashSet<_> =
                        hive.workers.iter().map(|w| w.worker_type.as_str()).collect();

                    let extra_workers: Vec<String> = installed_workers
                        .iter()
                        .filter(|w| !desired_set.contains(w.as_str()))
                        .cloned()
                        .collect();

                    if !extra_workers.is_empty() {
                        diff.workers_to_remove.push((hive.alias.clone(), extra_workers));
                    }
                }
            } else {
                // Hive exists but no workers installed
                if !hive.workers.is_empty() {
                    diff.workers_to_install.push((hive.alias.clone(), hive.workers.clone()));
                }
            }
        } else {
            // Hive doesn't exist - needs installation
            diff.hives_to_install.push(hive.clone());
        }
    }

    // Check for extra hives (if remove_extra)
    if remove_extra {
        let desired_set: std::collections::HashSet<_> =
            desired.iter().map(|h| h.alias.as_str()).collect();

        for actual_alias in actual_hives {
            if !desired_set.contains(actual_alias.as_str()) {
                diff.hives_to_remove.push(actual_alias.clone());
            }
        }
    }

    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_diff_empty() {
        let diff = compute_diff(&[], &[], &[], false);
        assert!(!diff.has_changes());
        assert_eq!(diff.change_count(), 0);
    }

    #[test]
    fn test_compute_diff_new_hive() {
        let desired = vec![HiveConfig {
            alias: "test-hive".to_string(),
            hostname: "localhost".to_string(),
            ssh_user: "test".to_string(),
            ssh_port: 22,
            hive_port: 8600,
            binary_path: None,
            workers: vec![],
            auto_start: true,
        }];

        let diff = compute_diff(&desired, &[], &[], false);
        assert!(diff.has_changes());
        assert_eq!(diff.hives_to_install.len(), 1);
        assert_eq!(diff.hives_already_installed.len(), 0);
    }

    #[test]
    fn test_compute_diff_existing_hive() {
        let desired = vec![HiveConfig {
            alias: "test-hive".to_string(),
            hostname: "localhost".to_string(),
            ssh_user: "test".to_string(),
            ssh_port: 22,
            hive_port: 8600,
            binary_path: None,
            workers: vec![],
            auto_start: true,
        }];

        let actual_hives = vec!["test-hive".to_string()];

        let diff = compute_diff(&desired, &actual_hives, &[], false);
        assert!(!diff.has_changes());
        assert_eq!(diff.hives_to_install.len(), 0);
        assert_eq!(diff.hives_already_installed.len(), 1);
    }
}

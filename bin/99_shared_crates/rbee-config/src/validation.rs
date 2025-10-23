//! Configuration validation
//!
//! Created by: TEAM-193

use crate::capabilities::CapabilitiesCache;
use crate::error::Result;
// TEAM-278: Use new declarative HivesConfig
use crate::declarative::HivesConfig;

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the validation passed
    pub valid: bool,
    /// List of validation errors
    pub errors: Vec<String>,
    /// List of validation warnings
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create new validation result
    pub const fn new() -> Self {
        Self { valid: true, errors: Vec::new(), warnings: Vec::new() }
    }

    /// Add error
    pub fn add_error(&mut self, error: String) {
        self.valid = false;
        self.errors.push(error);
    }

    /// Add warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if valid
    pub const fn is_valid(&self) -> bool {
        self.valid
    }

    /// Check if has warnings
    pub const fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate hives configuration
pub fn validate_hives_config(hives: &HivesConfig) -> ValidationResult {
    let mut result = ValidationResult::new();

    // Check if empty
    if hives.is_empty() {
        result.add_warning("No hives configured in hives.conf".to_string());
        return result;
    }

    // TEAM-278: Validate each hive entry (using new Vec-based API)
    for entry in &hives.hives {
        // Validate hostname
        if entry.hostname.is_empty() {
            result.add_error(format!("Hive '{}': hostname is empty", entry.alias));
        }

        // Validate SSH port (u16 max is 65535, so no need to check upper bound)
        if entry.ssh_port == 0 {
            result.add_error(format!("Hive '{}': SSH port cannot be 0", entry.alias));
        }

        // Validate hive port (u16 max is 65535, so no need to check upper bound)
        if entry.hive_port == 0 {
            result.add_error(format!("Hive '{}': hive port cannot be 0", entry.alias));
        }

        // Validate SSH user
        if entry.ssh_user.is_empty() {
            result.add_error(format!("Hive '{}': SSH user is empty", entry.alias));
        }

        // Warn if SSH and hive ports are the same
        if entry.ssh_port == entry.hive_port {
            result.add_warning(format!(
                "Hive '{}': SSH port and hive port are the same ({})",
                entry.alias, entry.ssh_port
            ));
        }
    }

    result
}

/// Validate capabilities cache against hives config
pub fn validate_capabilities_sync(
    hives: &HivesConfig,
    capabilities: &CapabilitiesCache,
) -> ValidationResult {
    let mut result = ValidationResult::new();

    let hive_aliases: std::collections::HashSet<_> = hives.aliases().into_iter().collect();
    let cap_aliases: std::collections::HashSet<_> = capabilities.aliases().into_iter().collect();

    // Find hives without capabilities
    for alias in hive_aliases.difference(&cap_aliases) {
        result
            .add_warning(format!("Hive '{}' is configured but has no cached capabilities", alias));
    }

    // Find capabilities for non-existent hives
    for alias in cap_aliases.difference(&hive_aliases) {
        result
            .add_warning(format!("Capabilities cached for '{}' but hive is not configured", alias));
    }

    result
}

/// Preflight validation - run before queen-rbee starts
///
/// # Errors
///
/// Returns an error if validation fails critically
pub fn preflight_validation(
    hives: &HivesConfig,
    capabilities: &CapabilitiesCache,
) -> Result<ValidationResult> {
    let mut result = ValidationResult::new();

    // Validate hives config
    let hives_result = validate_hives_config(hives);
    result.errors.extend(hives_result.errors);
    result.warnings.extend(hives_result.warnings);
    result.valid = result.valid && hives_result.valid;

    // Validate capabilities sync
    let sync_result = validate_capabilities_sync(hives, capabilities);
    result.warnings.extend(sync_result.warnings);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::HiveCapabilities;
    // TEAM-278: Use new declarative types
    use crate::declarative::HiveConfig;
    use std::collections::HashMap;

    // TEAM-278: Helper to create test hive using new declarative API
    fn create_test_hive(alias: &str) -> HiveConfig {
        HiveConfig {
            alias: alias.to_string(),
            hostname: "192.168.1.100".to_string(),
            ssh_port: 22,
            ssh_user: "user".to_string(),
            hive_port: 8081,
            binary_path: None,
            workers: vec![],
            auto_start: true,
        }
    }

    // TEAM-278: All tests updated to use new declarative API

    #[test]
    fn test_validate_empty_hives() {
        let hives = HivesConfig { hives: vec![] };
        let result = validate_hives_config(&hives);
        assert!(result.is_valid());
        assert!(result.has_warnings());
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_validate_valid_hive() {
        let hives = HivesConfig {
            hives: vec![create_test_hive("test")],
        };

        let result = validate_hives_config(&hives);
        assert!(result.is_valid());
        assert!(!result.has_warnings());
    }

    #[test]
    fn test_validate_invalid_hive() {
        let mut invalid_hive = create_test_hive("invalid");
        invalid_hive.hostname = "".to_string();
        invalid_hive.ssh_port = 0;
        let hives = HivesConfig {
            hives: vec![invalid_hive],
        };

        let result = validate_hives_config(&hives);
        assert!(!result.is_valid());
        assert!(result.errors.len() >= 2);
    }

    #[test]
    fn test_validate_port_conflict() {
        let mut hive = create_test_hive("test");
        hive.ssh_port = 8081;
        hive.hive_port = 8081;
        let hives = HivesConfig {
            hives: vec![hive],
        };

        let result = validate_hives_config(&hives);
        assert!(result.is_valid()); // Valid but has warning
        assert!(result.has_warnings());
    }

    #[test]
    fn test_validate_capabilities_sync() {
        let hives = HivesConfig {
            hives: vec![
                create_test_hive("hive1"),
                create_test_hive("hive2"),
            ],
        };

        let mut cap_map = HashMap::new();
        cap_map.insert(
            "hive1".to_string(),
            HiveCapabilities::new("hive1".to_string(), vec![], "http://localhost:8081".to_string()),
        );
        cap_map.insert(
            "hive3".to_string(),
            HiveCapabilities::new("hive3".to_string(), vec![], "http://localhost:8081".to_string()),
        );
        let capabilities = CapabilitiesCache::from_map(
            std::path::PathBuf::new(),
            "2025-10-21T20:00:00Z".to_string(),
            cap_map,
        );

        let result = validate_capabilities_sync(&hives, &capabilities);
        assert!(result.is_valid());
        assert_eq!(result.warnings.len(), 2); // hive2 missing caps, hive3 not configured
    }

    #[test]
    fn test_preflight_validation() {
        let hives = HivesConfig {
            hives: vec![create_test_hive("test")],
        };

        let capabilities = CapabilitiesCache::from_map(
            std::path::PathBuf::new(),
            "2025-10-21T20:00:00Z".to_string(),
            HashMap::new(),
        );

        let result = preflight_validation(&hives, &capabilities).unwrap();
        assert!(result.is_valid());
        assert!(result.has_warnings()); // test hive has no capabilities
    }

    // TEAM-195: Port validation tests
    #[test]
    fn test_validate_zero_ssh_port() {
        let mut hive = create_test_hive("test");
        hive.ssh_port = 0; // Invalid port
        let hives = HivesConfig {
            hives: vec![hive],
        };

        let result = validate_hives_config(&hives);
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| e.contains("SSH port cannot be 0")));
    }

    #[test]
    fn test_validate_zero_hive_port() {
        let mut hive = create_test_hive("test");
        hive.hive_port = 0; // Invalid port
        let hives = HivesConfig {
            hives: vec![hive],
        };

        let result = validate_hives_config(&hives);
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| e.contains("hive port cannot be 0")));
    }
}

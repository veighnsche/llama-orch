// TEAM-210: Validation helpers for hive operations
// TEAM-220: Investigated - Localhost special case + auto-generation documented

use anyhow::Result;
use once_cell::sync::Lazy;
use rbee_config::{HiveConfig, RbeeConfig};
use std::path::PathBuf;

/// Validate that a hive alias exists in config
///
/// Returns helpful error message listing available hives if alias not found.
/// Special case: "localhost" always returns a default entry.
///
/// COPIED FROM: job_router.rs lines 98-160
pub fn validate_hive_exists<'a>(config: &'a RbeeConfig, alias: &str) -> Result<&'a HiveConfig> {
    if alias == "localhost" {
        // Localhost operations do not require configuration
        // TEAM-280: Updated to include new fields from declarative config
        static LOCALHOST_ENTRY: Lazy<HiveConfig> = Lazy::new(|| HiveConfig {
            alias: "localhost".to_string(),
            hostname: "127.0.0.1".to_string(),
            ssh_port: 22,
            ssh_user: "user".to_string(),
            hive_port: 9000,
            binary_path: Some("target/debug/rbee-hive".to_string()),
            install_method: Default::default(), // Use default git installation
            workers: Vec::new(), // TEAM-280: No workers for localhost default
            auto_start: false,   // TEAM-280: Don't auto-start localhost
        });
        return Ok(&LOCALHOST_ENTRY);
    }

    config.get_hive(alias).ok_or_else(|| {
        let available_hives = config.all_hives();
        let hive_list = if available_hives.is_empty() {
            "  (none configured)".to_string()
        } else {
            available_hives
                .iter()
                .map(|h| format!("  - {}", h.alias))
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Check if hives.conf exists
        let config_dir = RbeeConfig::config_dir().unwrap_or_else(|_| PathBuf::from("~/.config/rbee"));
        let hives_conf_path = config_dir.join("hives.conf");
        if !hives_conf_path.exists() && alias != "localhost" {
            // Auto-generate a template for hives.conf
            let template_content = format!(
                "# hives.conf - rbee hive configuration\n\nHost {}\n  HostName <hostname or IP>\n  Port 22\n  User <username>\n  HivePort 8600\n  BinaryPath /path/to/rbee-hive\n",
                alias
            );
            let _ = std::fs::write(&hives_conf_path, &template_content);

            anyhow::anyhow!(
                "Hive alias '{}' not found in hives.conf.\n\nAvailable hives:\n{}\n\nA template hives.conf has been auto-generated at {}.\nPlease edit this file to configure access to '{}'.\n\nExample configuration:\n{}\n",
                alias,
                hive_list,
                hives_conf_path.display(),
                alias,
                template_content
            )
        } else {
            anyhow::anyhow!(
                "Hive alias '{}' not found in hives.conf.\n\nAvailable hives:\n{}\n\nAdd '{}' to ~/.config/rbee/hives.conf to use it.",
                alias,
                hive_list,
                alias
            )
        }
    })
}

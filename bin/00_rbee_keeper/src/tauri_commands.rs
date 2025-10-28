//! Tauri commands for the rbee-keeper GUI
//!
//! TEAM-293: Created Tauri command wrappers for all CLI operations
//! TEAM-297: Updated to use specta v2 for proper TypeScript type generation
//! TEAM-334: Cleaned up - only keeping ssh_list, rest to be re-implemented later
//!
//! Currently only exposes SSH config parsing. Other commands will be added
//! as the architecture stabilizes.

use crate::ssh_resolver;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;
    use specta_typescript::Typescript;
    use tauri_specta::{collect_commands, Builder};

    #[test]
    fn export_typescript_bindings() {
        // TEAM-297: Test that exports TypeScript bindings
        // TEAM-333: Updated to use ssh_list command
        let builder = Builder::<tauri::Wry>::new()
            .commands(collect_commands![ssh_list]);
        
        builder
            .export(
                Typescript::default(),
                "ui/src/generated/bindings.ts",
            )
            .expect("Failed to export typescript bindings");
    }
}

// ============================================================================
// SSH TYPES
// ============================================================================

/// SSH target from ~/.ssh/config
/// 
/// TEAM-333: Type for SSH config entries with specta support for TypeScript bindings
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct SshTarget {
    /// Host alias from SSH config
    pub host: String,
    /// Host subtitle (optional)
    pub host_subtitle: Option<String>,
    /// Hostname (IP or domain)
    pub hostname: String,
    /// SSH username
    pub user: String,
    /// SSH port
    pub port: u16,
    /// Connection status
    pub status: SshTargetStatus,
}

/// SSH target connection status
/// 
/// TEAM-333: Status enum for SSH targets
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum SshTargetStatus {
    Online,
    Offline,
    Unknown,
}

// ============================================================================
// SSH COMMANDS
// ============================================================================

#[tauri::command]
#[specta::specta]
pub async fn ssh_list() -> Result<Vec<SshTarget>, String> {
    // TEAM-333: Parse ~/.ssh/config and return list of SSH targets
    // TEAM-333: Deduplicate by hostname, keep shortest host alias
    use observability_narration_core::n;
    use std::collections::HashMap;
    
    n!("ssh_list", "Reading SSH config");
    
    // Get SSH config path
    let home = std::env::var("HOME")
        .map_err(|_| "HOME environment variable not set".to_string())?;
    let ssh_config_path = std::path::PathBuf::from(home).join(".ssh/config");
    
    // Parse SSH config
    let hosts = ssh_resolver::parse_ssh_config(&ssh_config_path)
        .map_err(|e| format!("Failed to parse SSH config: {}", e))?;
    
    // TEAM-333: Deduplicate by hostname - collect all aliases first, then pick shortest
    let mut by_hostname: HashMap<String, Vec<String>> = HashMap::new();
    let mut configs_map: HashMap<String, daemon_lifecycle::SshConfig> = HashMap::new();
    
    // First pass: collect all aliases for each unique hostname
    for (host, config) in hosts {
        let key = format!("{}:{}@{}", config.hostname, config.port, config.user);
        by_hostname.entry(key.clone()).or_insert_with(Vec::new).push(host);
        configs_map.insert(key, config);
    }
    
    // Second pass: for each hostname, pick shortest alias and use others as subtitle
    let mut targets: Vec<SshTarget> = Vec::new();
    for (key, mut aliases) in by_hostname {
        // Sort aliases by length (shortest first)
        aliases.sort_by_key(|a| a.len());
        
        let config = configs_map.get(&key).unwrap();
        let primary = aliases[0].clone();
        let subtitle = if aliases.len() > 1 {
            Some(aliases[1..].join(", "))
        } else {
            None
        };
        
        targets.push(SshTarget {
            host: primary,
            host_subtitle: subtitle,
            hostname: config.hostname.clone(),
            user: config.user.clone(),
            port: config.port,
            status: SshTargetStatus::Unknown,
        });
    }
    
    // Sort by host name
    targets.sort_by(|a, b| a.host.cmp(&b.host));
    
    // TEAM-333: NO localhost injection - only show what's in SSH config
    // If user wants localhost, they can add it to ~/.ssh/config
    
    n!("ssh_list", "Found {} unique SSH targets", targets.len());
    
    Ok(targets)
}

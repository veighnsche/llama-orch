//! SSH Config Parser
//!
//! TEAM-365: Created by TEAM-365
//! Extracted from bin/00_rbee_keeper/src/ssh_resolver.rs for reusability
//!
//! This crate provides SSH config parsing for Queen hive discovery and
//! rbee-keeper SSH target listing.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// SSH target from parsed SSH config
///
/// TEAM-365: Represents a single SSH host entry
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SshTarget {
    /// Host alias from SSH config (e.g., "workstation")
    pub host: String,
    
    /// Actual hostname or IP address (e.g., "192.168.1.100")
    pub hostname: String,
    
    /// SSH username
    pub user: String,
    
    /// SSH port (default: 22)
    pub port: u16,
}

/// Get default SSH config path (~/.ssh/config)
///
/// TEAM-365: Helper to get standard SSH config location
pub fn get_default_ssh_config_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join(".ssh/config")
}

/// Parse SSH config file into list of SSH targets
///
/// TEAM-365: Extracted from bin/00_rbee_keeper/src/ssh_resolver.rs:93-170
///
/// # Arguments
/// * `path` - Path to SSH config file (typically ~/.ssh/config)
///
/// # Returns
/// * `Ok(Vec<SshTarget>)` - List of parsed SSH targets
/// * `Err` - Failed to read or parse SSH config
///
/// # Behavior
/// - Returns empty vec if file doesn't exist
/// - Supports multiple aliases per host (e.g., `Host workstation workstation.local`)
/// - Defaults to current user if `User` not specified
/// - Defaults to port 22 if `Port` not specified
///
/// # SSH Config Format
/// ```text
/// Host workstation
///     HostName 192.168.1.100
///     User vince
///     Port 22
/// ```
pub fn parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>> {
    // TEAM-365: If file doesn't exist, return empty vec (only localhost will work)
    if !path.exists() {
        return Ok(Vec::new());
    }

    let content = fs::read_to_string(path).context("Failed to read SSH config")?;

    let mut hosts_map = HashMap::new();
    let mut current_host: Option<String> = None;
    let mut current_hostname: Option<String> = None;
    let mut current_user: Option<String> = None;
    let mut current_port: u16 = 22;

    for line in content.lines() {
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse key-value pairs
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        let key = parts[0].to_lowercase();
        let value = parts[1..].join(" ");

        match key.as_str() {
            "host" => {
                // TEAM-365: Save previous host entry for ALL aliases
                if let (Some(host_aliases), Some(hostname)) =
                    (current_host.take(), current_hostname.take())
                {
                    let user = current_user.take().unwrap_or_else(whoami::username);
                    
                    // Add entry for each alias (e.g., "workstation" and "workstation.home.arpa")
                    for alias in host_aliases.split_whitespace() {
                        hosts_map.insert(
                            alias.to_string(),
                            SshTarget {
                                host: alias.to_string(),
                                hostname: hostname.clone(),
                                user: user.clone(),
                                port: current_port,
                            },
                        );
                    }
                }

                // Start new host entry (store all aliases as a single string)
                current_host = Some(value);
                current_hostname = None;
                current_user = None;
                current_port = 22;
            }
            "hostname" => {
                current_hostname = Some(value);
            }
            "user" => {
                current_user = Some(value);
            }
            "port" => {
                current_port = value.parse().unwrap_or(22);
            }
            _ => {} // Ignore other directives
        }
    }

    // TEAM-365: Save last host entry for ALL aliases
    if let (Some(host_aliases), Some(hostname)) = (current_host, current_hostname) {
        let user = current_user.unwrap_or_else(whoami::username);
        
        for alias in host_aliases.split_whitespace() {
            hosts_map.insert(
                alias.to_string(),
                SshTarget {
                    host: alias.to_string(),
                    hostname: hostname.clone(),
                    user: user.clone(),
                    port: current_port,
                },
            );
        }
    }

    // TEAM-365: Convert HashMap to Vec and sort by host alias
    let mut targets: Vec<SshTarget> = hosts_map.into_values().collect();
    targets.sort_by(|a, b| a.host.cmp(&b.host));

    Ok(targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_ssh_config() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Host workstation").unwrap();
        writeln!(file, "    HostName 192.168.1.100").unwrap();
        writeln!(file, "    User vince").unwrap();
        writeln!(file, "    Port 2222").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "Host server").unwrap();
        writeln!(file, "    HostName example.com").unwrap();
        writeln!(file, "    User admin").unwrap();
        file.flush().unwrap();

        let targets = parse_ssh_config(file.path()).unwrap();

        assert_eq!(targets.len(), 2);

        // Find workstation (order may vary)
        let workstation = targets.iter().find(|t| t.host == "workstation").unwrap();
        assert_eq!(workstation.hostname, "192.168.1.100");
        assert_eq!(workstation.user, "vince");
        assert_eq!(workstation.port, 2222);

        // Find server
        let server = targets.iter().find(|t| t.host == "server").unwrap();
        assert_eq!(server.hostname, "example.com");
        assert_eq!(server.user, "admin");
        assert_eq!(server.port, 22); // default
    }

    #[test]
    fn test_parse_ssh_config_multiple_aliases() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Host workstation workstation.local").unwrap();
        writeln!(file, "    HostName 192.168.1.100").unwrap();
        writeln!(file, "    User vince").unwrap();
        file.flush().unwrap();

        let targets = parse_ssh_config(file.path()).unwrap();

        assert_eq!(targets.len(), 2);
        
        let ws1 = targets.iter().find(|t| t.host == "workstation").unwrap();
        let ws2 = targets.iter().find(|t| t.host == "workstation.local").unwrap();
        
        assert_eq!(ws1.hostname, "192.168.1.100");
        assert_eq!(ws2.hostname, "192.168.1.100");
    }

    #[test]
    fn test_parse_ssh_config_missing_file() {
        let path = PathBuf::from("/nonexistent/ssh/config");
        let targets = parse_ssh_config(&path).unwrap();
        assert_eq!(targets.len(), 0);
    }

    #[test]
    fn test_get_default_ssh_config_path() {
        let path = get_default_ssh_config_path();
        assert!(path.to_string_lossy().ends_with(".ssh/config"));
    }
}

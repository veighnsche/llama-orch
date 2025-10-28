//! SSH Config Resolver
//!
//! TEAM-332: Middleware to resolve host aliases to SshConfig
//!
//! This module provides automatic translation of host aliases (e.g., "workstation")
//! to SSH configuration by parsing ~/.ssh/config. Defaults to localhost.
//!
//! # Usage
//! ```rust,ignore
//! use crate::ssh_resolver::resolve_ssh_config;
//!
//! // Resolves to localhost (no SSH)
//! let ssh = resolve_ssh_config("localhost")?;
//!
//! // Parses ~/.ssh/config for "workstation" host
//! let ssh = resolve_ssh_config("workstation")?;
//! ```

use anyhow::{Context, Result};
use daemon_lifecycle::SshConfig;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Resolve host alias to SSH configuration
///
/// TEAM-332: Middleware that eliminates repeated SshConfig::localhost() calls
///
/// # Arguments
/// * `host_alias` - Host alias (e.g., "localhost", "workstation")
///
/// # Returns
/// * `Ok(SshConfig)` - Resolved SSH configuration
/// * `Err` - Failed to parse SSH config or host not found
///
/// # Behavior
/// - "localhost" → `SshConfig::localhost()` (no SSH)
/// - Other aliases → Parse `~/.ssh/config` for host entry
///
/// # Example
/// ```rust,ignore
/// // Localhost (no SSH)
/// let ssh = resolve_ssh_config("localhost")?;
///
/// // Remote host (parses ~/.ssh/config)
/// let ssh = resolve_ssh_config("workstation")?;
/// ```
pub fn resolve_ssh_config(host_alias: &str) -> Result<SshConfig> {
    // TEAM-332: Localhost bypass - no SSH config needed
    if host_alias == "localhost" {
        return Ok(SshConfig::localhost());
    }
    
    // Parse ~/.ssh/config for remote host
    let ssh_config_path = get_ssh_config_path()?;
    let hosts = parse_ssh_config(&ssh_config_path)?;
    
    // Look up host alias
    hosts.get(host_alias)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!(
            "Host '{}' not found in ~/.ssh/config\n\
             \n\
             Add an entry like:\n\
             \n\
             Host {}\n\
                 HostName 192.168.1.100\n\
                 User vince\n\
                 Port 22",
            host_alias, host_alias
        ))
}

/// Get path to SSH config file
fn get_ssh_config_path() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .context("HOME environment variable not set")?;
    Ok(PathBuf::from(home).join(".ssh/config"))
}

/// Parse SSH config file into host entries
///
/// TEAM-332: Simple SSH config parser (supports Host, HostName, User, Port)
/// TEAM-333: Made public for use in tauri_commands
///
/// # Format
/// ```text
/// Host workstation
///     HostName 192.168.1.100
///     User vince
///     Port 22
/// ```
pub fn parse_ssh_config(path: &PathBuf) -> Result<HashMap<String, SshConfig>> {
    // If file doesn't exist, return empty map (only localhost will work)
    if !path.exists() {
        return Ok(HashMap::new());
    }
    
    let content = fs::read_to_string(path)
        .context("Failed to read ~/.ssh/config")?;
    
    let mut hosts = HashMap::new();
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
                // Save previous host entry for ALL aliases
                if let (Some(host_aliases), Some(hostname)) = (current_host.take(), current_hostname.take()) {
                    let user = current_user.take().unwrap_or_else(whoami::username);
                    let config = SshConfig::new(hostname, user, current_port);
                    
                    // Add entry for each alias (e.g., "workstation" and "workstation.home.arpa")
                    for alias in host_aliases.split_whitespace() {
                        hosts.insert(alias.to_string(), config.clone());
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
    
    // Save last host entry for ALL aliases
    if let (Some(host_aliases), Some(hostname)) = (current_host, current_hostname) {
        let user = current_user.unwrap_or_else(whoami::username);
        let config = SshConfig::new(hostname, user, current_port);
        
        // Add entry for each alias
        for alias in host_aliases.split_whitespace() {
            hosts.insert(alias.to_string(), config.clone());
        }
    }
    
    Ok(hosts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_localhost_resolution() {
        let ssh = resolve_ssh_config("localhost").unwrap();
        assert!(ssh.is_localhost());
    }

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

        let hosts = parse_ssh_config(&file.path().to_path_buf()).unwrap();
        
        assert_eq!(hosts.len(), 2);
        
        let workstation = hosts.get("workstation").unwrap();
        assert_eq!(workstation.hostname, "192.168.1.100");
        assert_eq!(workstation.user, "vince");
        assert_eq!(workstation.port, 2222);
        
        let server = hosts.get("server").unwrap();
        assert_eq!(server.hostname, "example.com");
        assert_eq!(server.user, "admin");
        assert_eq!(server.port, 22); // default
    }

    #[test]
    fn test_missing_host() {
        let result = resolve_ssh_config("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }
}

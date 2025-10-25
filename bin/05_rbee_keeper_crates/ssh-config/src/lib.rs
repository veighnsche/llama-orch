//! SSH Config Parser for rbee-keeper
//!
//! TEAM-294: Parse ~/.ssh/config to discover available hives
//! Returns structured data for both CLI (NARRATE.table) and UI (JSON)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// SSH target from ~/.ssh/config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshTarget {
    /// Host alias from SSH config (first word)
    pub host: String,
    /// Host subtitle (second word, optional)
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SshTargetStatus {
    Online,
    Offline,
    Unknown,
}

/// Parse SSH config file and extract hosts
///
/// # Example
///
/// ```no_run
/// use ssh_config::parse_ssh_config;
/// use std::path::Path;
///
/// let targets = parse_ssh_config(Path::new("/home/user/.ssh/config")).unwrap();
/// for target in targets {
///     println!("{}: {}@{}:{}", target.host, target.user, target.hostname, target.port);
/// }
/// ```
pub fn parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let content = fs::read_to_string(path)
        .context("Failed to read SSH config file")?;

    let mut targets = Vec::new();
    let mut current_host: Option<String> = None;
    let mut current_hostname: Option<String> = None;
    let mut current_user: Option<String> = None;
    let mut current_port: Option<u16> = None;

    for line in content.lines() {
        let line = line.trim();
        
        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        let key = parts[0].to_lowercase();
        let value = parts[1..].join(" ");

        match key.as_str() {
            "host" => {
                // Save previous host if complete
                if let (Some(host), Some(hostname)) = (current_host.take(), current_hostname.take()) {
                    // Skip wildcard hosts
                    if !host.contains('*') && !host.contains('?') {
                        // Split host into main and subtitle
                        let host_parts: Vec<&str> = host.split_whitespace().collect();
                        let main_host = host_parts.first().unwrap_or(&"").to_string();
                        let subtitle = host_parts.get(1).map(|s| s.to_string());
                        
                        targets.push(SshTarget {
                            host: main_host,
                            host_subtitle: subtitle,
                            hostname,
                            user: current_user.take().unwrap_or_else(|| "root".to_string()),
                            port: current_port.take().unwrap_or(22),
                            status: SshTargetStatus::Unknown,
                        });
                    }
                }
                
                // Start new host
                current_host = Some(value);
                current_hostname = None;
                current_user = None;
                current_port = None;
            }
            "hostname" => {
                current_hostname = Some(value);
            }
            "user" => {
                current_user = Some(value);
            }
            "port" => {
                if let Ok(port) = value.parse::<u16>() {
                    current_port = Some(port);
                }
            }
            _ => {}
        }
    }

    // Save last host if complete
    if let (Some(host), Some(hostname)) = (current_host, current_hostname) {
        if !host.contains('*') && !host.contains('?') {
            // Split host into main and subtitle
            let host_parts: Vec<&str> = host.split_whitespace().collect();
            let main_host = host_parts.first().unwrap_or(&"").to_string();
            let subtitle = host_parts.get(1).map(|s| s.to_string());
            
            targets.push(SshTarget {
                host: main_host,
                host_subtitle: subtitle,
                hostname,
                user: current_user.unwrap_or_else(|| "root".to_string()),
                port: current_port.unwrap_or(22),
                status: SshTargetStatus::Unknown,
            });
        }
    }

    Ok(targets)
}

/// Check if a host is reachable (optional, can be slow)
///
/// This is a simple TCP connection test. For production, you might want
/// to use a proper SSH library or ping.
pub fn check_host_status(_target: &SshTarget) -> Result<SshTargetStatus> {
    // TEAM-294: Stub implementation - always return Unknown
    // Real implementation would do:
    // - TCP connection test to hostname:port
    // - Or SSH connection attempt
    // - Or ping test
    Ok(SshTargetStatus::Unknown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_empty_config() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "").unwrap();
        
        let targets = parse_ssh_config(file.path()).unwrap();
        assert_eq!(targets.len(), 0);
    }

    #[test]
    fn test_parse_single_host() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Host production").unwrap();
        writeln!(file, "  HostName 192.168.1.100").unwrap();
        writeln!(file, "  User deploy").unwrap();
        writeln!(file, "  Port 22").unwrap();
        
        let targets = parse_ssh_config(file.path()).unwrap();
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].host, "production");
        assert_eq!(targets[0].host_subtitle, None);
        assert_eq!(targets[0].hostname, "192.168.1.100");
        assert_eq!(targets[0].user, "deploy");
        assert_eq!(targets[0].port, 22);
    }

    #[test]
    fn test_parse_multiple_hosts() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Host server1").unwrap();
        writeln!(file, "  HostName 10.0.0.1").unwrap();
        writeln!(file, "  User admin").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "Host server2").unwrap();
        writeln!(file, "  HostName 10.0.0.2").unwrap();
        writeln!(file, "  User root").unwrap();
        writeln!(file, "  Port 2222").unwrap();
        
        let targets = parse_ssh_config(file.path()).unwrap();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].host, "server1");
        assert_eq!(targets[1].host, "server2");
        assert_eq!(targets[1].port, 2222);
    }

    #[test]
    fn test_skip_wildcard_hosts() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Host *").unwrap();
        writeln!(file, "  User default").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "Host server1").unwrap();
        writeln!(file, "  HostName 10.0.0.1").unwrap();
        
        let targets = parse_ssh_config(file.path()).unwrap();
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].host, "server1");
    }

    #[test]
    fn test_default_values() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Host minimal").unwrap();
        writeln!(file, "  HostName 10.0.0.1").unwrap();
        
        let targets = parse_ssh_config(file.path()).unwrap();
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].user, "root"); // default user
        assert_eq!(targets[0].port, 22); // default port
    }

    #[test]
    fn test_multi_word_host_aliases() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Host workstation workstation.home.arpa").unwrap();
        writeln!(file, "  HostName 192.168.178.29").unwrap();
        writeln!(file, "  User vince").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "Host mac mac.home.arpa").unwrap();
        writeln!(file, "  HostName 192.168.178.15").unwrap();
        writeln!(file, "  User vinceliem").unwrap();
        
        let targets = parse_ssh_config(file.path()).unwrap();
        assert_eq!(targets.len(), 2);
        
        // First host
        assert_eq!(targets[0].host, "workstation");
        assert_eq!(targets[0].host_subtitle, Some("workstation.home.arpa".to_string()));
        assert_eq!(targets[0].hostname, "192.168.178.29");
        assert_eq!(targets[0].user, "vince");
        
        // Second host
        assert_eq!(targets[1].host, "mac");
        assert_eq!(targets[1].host_subtitle, Some("mac.home.arpa".to_string()));
        assert_eq!(targets[1].hostname, "192.168.178.15");
        assert_eq!(targets[1].user, "vinceliem");
    }
}

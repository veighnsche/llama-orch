//! SSH Config Resolver
//!
//! TEAM-332: Middleware to resolve host aliases to SshConfig
//! TEAM-365: Updated to use shared ssh-config-parser crate (RULE ZERO)
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
use lifecycle_ssh::SshConfig;
use ssh_config_parser; // TEAM-365: Use shared crate

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
    // TEAM-365: Localhost bypass - no SSH config needed
    // TEAM-358 removed SshConfig::localhost(), so we create it manually
    if host_alias == "localhost" {
        return Ok(SshConfig::new(
            "localhost".to_string(),
            whoami::username(),
            22,
        ));
    }

    // TEAM-365: Use shared ssh-config-parser crate
    let ssh_config_path = ssh_config_parser::get_default_ssh_config_path();
    let targets = ssh_config_parser::parse_ssh_config(&ssh_config_path)?;

    // TEAM-365: Look up host alias and convert to SshConfig
    targets
        .iter()
        .find(|t| t.host == host_alias)
        .map(|t| SshConfig::new(t.hostname.clone(), t.user.clone(), t.port))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Host '{}' not found in ~/.ssh/config\n\
                 \n\
                 Add an entry like:\n\
                 \n\
                 Host {}\n\
                     HostName 192.168.1.100\n\
                     User vince\n\
                     Port 22",
                host_alias,
                host_alias
            )
        })
}

// TEAM-365: RULE ZERO - Deleted deprecated parse_ssh_config() and get_ssh_config_path()
// These functions are now in bin/99_shared_crates/ssh-config-parser/
// Use ssh_config_parser::parse_ssh_config() and ssh_config_parser::get_default_ssh_config_path() instead

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_localhost_resolution() {
        let ssh = resolve_ssh_config("localhost").unwrap();
        // TEAM-365: TEAM-358 removed is_localhost(), check hostname instead
        assert_eq!(ssh.hostname, "localhost");
    }

    // TEAM-365: RULE ZERO - Deleted test_parse_ssh_config()
    // This test is now in bin/99_shared_crates/ssh-config-parser/src/lib.rs
    // Run: cargo test -p ssh-config-parser

    #[test]
    fn test_missing_host() {
        let result = resolve_ssh_config("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }
}

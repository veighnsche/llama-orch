//! Hives configuration (hives.conf) - SSH config style
//!
//! Created by: TEAM-193

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Hives configuration loaded from hives.conf
#[derive(Debug, Clone)]
pub struct HivesConfig {
    hives: HashMap<String, HiveEntry>,
}

/// Single hive entry from hives.conf
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HiveEntry {
    /// Alias (from "Host" directive)
    pub alias: String,

    /// Hostname or IP address
    pub hostname: String,

    /// SSH port
    pub ssh_port: u16,

    /// SSH username
    pub ssh_user: String,

    /// Hive HTTP API port
    pub hive_port: u16,

    /// Optional path to rbee-hive binary
    pub binary_path: Option<String>,
}

impl HivesConfig {
    /// Create new empty hives config (for testing)
    #[cfg(test)]
    pub(crate) fn new_empty() -> Self {
        Self { hives: HashMap::new() }
    }

    /// Create from hashmap (for testing)
    #[cfg(test)]
    pub(crate) fn from_map(hives: HashMap<String, HiveEntry>) -> Self {
        Self { hives }
    }

    /// Load from hives.conf file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            // Return an empty config if file doesn't exist, as localhost doesn't require a hive entry
            return Ok(Self { hives: HashMap::new() });
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::ReadError { path: path.to_path_buf(), source: e })?;

        let hives = parse_hives_conf(&content)?;

        let config = Self { hives };
        config.validate_unique_aliases()?;

        Ok(config)
    }

    /// Get hive by alias
    pub fn get(&self, alias: &str) -> Option<&HiveEntry> {
        self.hives.get(alias)
    }

    /// List all hives
    pub fn all(&self) -> Vec<&HiveEntry> {
        self.hives.values().collect()
    }

    /// Get all aliases
    pub fn aliases(&self) -> Vec<String> {
        self.hives.keys().cloned().collect()
    }

    /// Check if alias exists
    pub fn contains(&self, alias: &str) -> bool {
        self.hives.contains_key(alias)
    }

    /// Validate unique aliases (should already be enforced by HashMap)
    ///
    /// # Errors
    ///
    /// Currently always returns Ok as HashMap enforces uniqueness
    pub const fn validate_unique_aliases(&self) -> Result<()> {
        // HashMap already enforces uniqueness, but we check during parsing
        Ok(())
    }

    /// Number of hives
    pub fn len(&self) -> usize {
        self.hives.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.hives.is_empty()
    }

    /// Import from SSH config file, injecting default HivePort
    ///
    /// Reads ~/.ssh/config format and converts to hives.conf format by:
    /// 1. Parsing SSH config (Host, HostName, Port, User)
    /// 2. Injecting HivePort for each host
    /// 3. Parsing with existing hives.conf parser
    ///
    /// # Errors
    ///
    /// Returns an error if the SSH config file cannot be read or parsed
    pub fn import_from_ssh_config(ssh_config_path: &Path, default_hive_port: u16) -> Result<Self> {
        let ssh_content = std::fs::read_to_string(ssh_config_path).map_err(|e| {
            ConfigError::ReadError { path: ssh_config_path.to_path_buf(), source: e }
        })?;

        // Inject HivePort into SSH config content
        let hives_content = inject_hive_port(&ssh_content, default_hive_port);

        // Parse with existing parser
        let hives = parse_hives_conf(&hives_content)?;

        Ok(Self { hives })
    }

    /// Merge another HivesConfig into this one
    ///
    /// Existing entries are NOT overwritten (existing wins)
    pub fn merge(&mut self, other: HivesConfig) {
        for (alias, entry) in other.hives {
            self.hives.entry(alias).or_insert(entry);
        }
    }

    /// Write to hives.conf file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut content = String::new();

        for entry in self.all() {
            content.push_str(&format!("Host {}\n", entry.alias));
            content.push_str(&format!("    HostName {}\n", entry.hostname));
            content.push_str(&format!("    Port {}\n", entry.ssh_port));
            content.push_str(&format!("    User {}\n", entry.ssh_user));
            content.push_str(&format!("    HivePort {}\n", entry.hive_port));
            if let Some(binary_path) = &entry.binary_path {
                content.push_str(&format!("    BinaryPath {}\n", binary_path));
            }
            content.push('\n');
        }

        std::fs::write(path, content)
            .map_err(|e| ConfigError::ReadError { path: path.to_path_buf(), source: e })?;

        Ok(())
    }
}

/// Inject HivePort directive after each User directive in SSH config
///
/// Converts:
/// ```
/// Host workstation
///     HostName 192.168.1.100
///     User vince
/// ```
///
/// To:
/// ```
/// Host workstation
///     HostName 192.168.1.100
///     User vince
///     HivePort 8081
/// ```
fn inject_hive_port(ssh_content: &str, default_hive_port: u16) -> String {
    let mut result = String::new();
    let mut in_host_block = false;
    let mut hive_port_injected = false;

    for line in ssh_content.lines() {
        let trimmed = line.trim();

        // Track if we're in a Host block
        if trimmed.starts_with("Host ") {
            in_host_block = true;
            hive_port_injected = false;
            result.push_str(line);
            result.push('\n');
            continue;
        }

        // If we see another Host or empty line after User, inject HivePort
        if in_host_block && !hive_port_injected {
            if trimmed.starts_with("User ") {
                result.push_str(line);
                result.push('\n');
                // Inject HivePort with same indentation
                let indent = line.len() - line.trim_start().len();
                result.push_str(&format!("{}HivePort {}\n", " ".repeat(indent), default_hive_port));
                hive_port_injected = true;
                continue;
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    result
}

/// Parse hives.conf content (SSH config style)
fn parse_hives_conf(content: &str) -> Result<HashMap<String, HiveEntry>> {
    let mut hives = HashMap::new();
    let mut current_host: Option<String> = None;
    let mut current_entry = PartialHiveEntry::default();
    let mut line_num = 0;

    for line in content.lines() {
        line_num += 1;
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse key-value pairs
        if line.starts_with("Host ") {
            // Save previous host if exists
            if let Some(alias) = current_host.take() {
                // Try to finalize, but skip if incomplete (for SSH config import tolerance)
                if let Some(entry) = current_entry.finalize_optional(&alias) {
                    // Check for duplicate alias
                    if hives.contains_key(&alias) {
                        return Err(ConfigError::DuplicateAlias { alias });
                    }
                    hives.insert(alias, entry);
                }
                current_entry = PartialHiveEntry::default();
            }

            // Start new host
            // SSH config allows multiple aliases: "Host alias1 alias2 alias3"
            // We only take the first one
            let alias_line = line[5..].trim();
            let alias = alias_line.split_whitespace().next().unwrap_or("").to_string();

            if alias.is_empty() {
                return Err(ConfigError::InvalidSyntax {
                    line: line_num,
                    message: "Empty host alias".to_string(),
                });
            }

            // Skip wildcards (*, ?)
            if alias.contains('*') || alias.contains('?') {
                continue;
            }

            current_host = Some(alias);
        } else if current_host.is_some() {
            // Parse host properties
            let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
            if parts.len() != 2 {
                return Err(ConfigError::InvalidSyntax {
                    line: line_num,
                    message: format!("Invalid line format: {}", line),
                });
            }

            let key = parts[0];
            let value = parts[1].trim();

            match key {
                "HostName" => current_entry.hostname = Some(value.to_string()),
                "Port" => {
                    current_entry.ssh_port =
                        Some(value.parse().map_err(|_| ConfigError::InvalidPort {
                            host: current_host.clone().unwrap_or_default(),
                            value: value.to_string(),
                        })?);
                }
                "User" => current_entry.ssh_user = Some(value.to_string()),
                "HivePort" => {
                    current_entry.hive_port =
                        Some(value.parse().map_err(|_| ConfigError::InvalidPort {
                            host: current_host.clone().unwrap_or_default(),
                            value: value.to_string(),
                        })?);
                }
                "BinaryPath" => current_entry.binary_path = Some(value.to_string()),
                _ => {
                    // Ignore unknown fields (SSH config compatibility)
                }
            }
        } else {
            return Err(ConfigError::InvalidSyntax {
                line: line_num,
                message: "Property outside of Host block".to_string(),
            });
        }
    }

    // Save last host
    if let Some(alias) = current_host {
        // Try to finalize, but skip if incomplete (for SSH config import tolerance)
        if let Some(entry) = current_entry.finalize_optional(&alias) {
            if hives.contains_key(&alias) {
                return Err(ConfigError::DuplicateAlias { alias });
            }
            hives.insert(alias, entry);
        }
    }

    Ok(hives)
}

/// Partial hive entry during parsing
#[derive(Default)]
struct PartialHiveEntry {
    hostname: Option<String>,
    ssh_port: Option<u16>,
    ssh_user: Option<String>,
    hive_port: Option<u16>,
    binary_path: Option<String>,
}

impl PartialHiveEntry {
    /// Finalize into HiveEntry, checking required fields
    fn finalize(self, alias: &str) -> Result<HiveEntry> {
        let hostname = self.hostname.ok_or_else(|| ConfigError::MissingField {
            host: alias.to_string(),
            field: "HostName".to_string(),
        })?;

        let ssh_port = self.ssh_port.unwrap_or(22); // Default SSH port

        let ssh_user = self.ssh_user.ok_or_else(|| ConfigError::MissingField {
            host: alias.to_string(),
            field: "User".to_string(),
        })?;

        let hive_port = self.hive_port.ok_or_else(|| ConfigError::MissingField {
            host: alias.to_string(),
            field: "HivePort".to_string(),
        })?;

        Ok(HiveEntry {
            alias: alias.to_string(),
            hostname,
            ssh_port,
            ssh_user,
            hive_port,
            binary_path: self.binary_path,
        })
    }

    /// Finalize into HiveEntry, returning None if required fields are missing
    /// Used for SSH config import to skip incomplete entries
    fn finalize_optional(self, alias: &str) -> Option<HiveEntry> {
        let hostname = self.hostname?;
        let ssh_port = self.ssh_port.unwrap_or(22);
        let ssh_user = self.ssh_user?;
        let hive_port = self.hive_port?;

        Some(HiveEntry {
            alias: alias.to_string(),
            hostname,
            ssh_port,
            ssh_user,
            hive_port,
            binary_path: self.binary_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_valid_hives_conf() {
        let content = r#"
# Localhost hive
Host localhost
    HostName 127.0.0.1
    Port 22
    User vince
    HivePort 8081

# Remote workstation
Host workstation
    HostName 192.168.1.100
    User admin
    HivePort 8081
    BinaryPath /usr/local/bin/rbee-hive
"#;

        let hives = parse_hives_conf(content).unwrap();
        assert_eq!(hives.len(), 2);

        let localhost = hives.get("localhost").unwrap();
        assert_eq!(localhost.alias, "localhost");
        assert_eq!(localhost.hostname, "127.0.0.1");
        assert_eq!(localhost.ssh_port, 22);
        assert_eq!(localhost.ssh_user, "vince");
        assert_eq!(localhost.hive_port, 8081);
        assert_eq!(localhost.binary_path, None);

        let workstation = hives.get("workstation").unwrap();
        assert_eq!(workstation.hostname, "192.168.1.100");
        assert_eq!(workstation.ssh_port, 22); // Default
        assert_eq!(workstation.binary_path, Some("/usr/local/bin/rbee-hive".to_string()));
    }

    #[test]
    fn test_parse_duplicate_alias() {
        let content = r#"
Host duplicate
    HostName 192.168.1.1
    User user1
    HivePort 8081

Host duplicate
    HostName 192.168.1.2
    User user2
    HivePort 8082
"#;

        let result = parse_hives_conf(content);
        assert!(result.is_err());
        match result.unwrap_err() {
            ConfigError::DuplicateAlias { alias } => assert_eq!(alias, "duplicate"),
            _ => panic!("Expected DuplicateAlias error"),
        }
    }

    #[test]
    fn test_parse_missing_required_field() {
        let content = r#"
Host incomplete
    HostName 192.168.1.1
    HivePort 8081
"#;

        let result = parse_hives_conf(content);
        assert!(result.is_err());
        match result.unwrap_err() {
            ConfigError::MissingField { host, field } => {
                assert_eq!(host, "incomplete");
                assert_eq!(field, "User");
            }
            _ => panic!("Expected MissingField error"),
        }
    }

    #[test]
    fn test_load_nonexistent_file() {
        let path = Path::new("/nonexistent/hives.conf");
        let config = HivesConfig::load(path).unwrap();
        assert_eq!(config.len(), 0);
    }

    #[test]
    fn test_load_from_file() {
        let mut file = NamedTempFile::new().unwrap();
        let content = r#"
Host test
    HostName 192.168.1.100
    Port 2222
    User testuser
    HivePort 8081
"#;
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();

        let config = HivesConfig::load(file.path()).unwrap();
        assert_eq!(config.len(), 1);

        let entry = config.get("test").unwrap();
        assert_eq!(entry.hostname, "192.168.1.100");
        assert_eq!(entry.ssh_port, 2222);
        assert_eq!(entry.ssh_user, "testuser");
        assert_eq!(entry.hive_port, 8081);
    }

    #[test]
    fn test_hives_config_api() {
        let content = r#"
Host hive1
    HostName 192.168.1.1
    User user1
    HivePort 8081

Host hive2
    HostName 192.168.1.2
    User user2
    HivePort 8082
"#;

        let hives = parse_hives_conf(content).unwrap();
        let config = HivesConfig { hives };

        assert_eq!(config.len(), 2);
        assert!(!config.is_empty());
        assert!(config.contains("hive1"));
        assert!(config.contains("hive2"));
        assert!(!config.contains("hive3"));

        let aliases = config.aliases();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&"hive1".to_string()));
        assert!(aliases.contains(&"hive2".to_string()));

        let all = config.all();
        assert_eq!(all.len(), 2);
    }
}

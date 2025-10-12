// Created by: TEAM-DX-001
// Configuration management for workspace-aware defaults

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the DX tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default base URL for dev servers
    pub base_url: String,
    
    /// Timeout in seconds
    pub timeout: u64,
    
    /// Workspace-specific settings
    pub workspace: WorkspaceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    /// Commercial frontend settings
    pub commercial: ProjectConfig,
    
    /// Storybook settings
    pub storybook: ProjectConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Project directory relative to frontend/
    pub dir: String,
    
    /// Default dev server URL
    pub url: String,
    
    /// Default dev server port
    pub port: u16,
    
    /// Paths to scan for components
    pub component_paths: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:3000".to_string(),
            timeout: 2,
            workspace: WorkspaceConfig::default(),
        }
    }
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            commercial: ProjectConfig {
                dir: "bin/commercial".to_string(),
                url: "http://localhost:3000".to_string(),
                port: 3000,
                component_paths: vec![
                    "app/**/*.vue".to_string(),
                    "components/**/*.vue".to_string(),
                    "layouts/**/*.vue".to_string(),
                    "pages/**/*.vue".to_string(),
                ],
            },
            storybook: ProjectConfig {
                dir: "libs/storybook".to_string(),
                url: "http://localhost:6006".to_string(),
                port: 6006,
                component_paths: vec![
                    "stories/**/*.vue".to_string(),
                ],
            },
        }
    }
}

impl Config {
    /// Load config from file or use defaults
    pub fn load() -> Self {
        // Try to load from .dxrc.json in frontend directory
        let config_path = Self::find_config_file();
        
        if let Some(path) = config_path {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(config) = serde_json::from_str(&contents) {
                    return config;
                }
            }
        }
        
        Self::default()
    }
    
    /// Find config file in current or parent directories
    fn find_config_file() -> Option<PathBuf> {
        let mut current = std::env::current_dir().ok()?;
        
        loop {
            let config_path = current.join(".dxrc.json");
            if config_path.exists() {
                return Some(config_path);
            }
            
            if !current.pop() {
                break;
            }
        }
        
        None
    }
    
    /// Get project config by name
    pub fn get_project(&self, name: &str) -> Option<&ProjectConfig> {
        match name {
            "commercial" => Some(&self.workspace.commercial),
            "storybook" => Some(&self.workspace.storybook),
            _ => {
                // Try to load from custom config
                if let Ok(contents) = std::fs::read_to_string("frontend/.dxrc.json")
                    .or_else(|_| std::fs::read_to_string(".dxrc.json"))
                {
                    if let Ok(config) = serde_json::from_str::<serde_json::Value>(&contents) {
                        if let Some(workspace) = config.get("workspace") {
                            if let Some(project) = workspace.get(name) {
                                if let Ok(_proj_config) = serde_json::from_value::<ProjectConfig>(project.clone()) {
                                    // This is a workaround - ideally we'd cache this
                                    // For now, return None and let the full config load handle it
                                }
                            }
                        }
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // TEAM-DX-002: Additional config tests
    
    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.base_url, "http://localhost:3000");
        assert_eq!(config.workspace.commercial.url, "http://localhost:3000");
        assert_eq!(config.workspace.storybook.url, "http://localhost:6006");
    }
    
    #[test]
    fn test_commercial_defaults() {
        let config = Config::default();
        let commercial = config.get_project("commercial").unwrap();
        assert_eq!(commercial.url, "http://localhost:3000");
        assert_eq!(commercial.port, 3000);
    }
    
    #[test]
    fn test_storybook_defaults() {
        let config = Config::default();
        let storybook = config.get_project("storybook").unwrap();
        assert_eq!(storybook.url, "http://localhost:6006");
        assert_eq!(storybook.port, 6006);
    }
    
    #[test]
    fn test_get_project_unknown() {
        let config = Config::default();
        let result = config.get_project("unknown");
        assert!(result.is_none());
    }
    
    #[test]
    fn test_project_config_structure() {
        let project = ProjectConfig {
            dir: "test".to_string(),
            url: "http://test:8080".to_string(),
            port: 8080,
            component_paths: vec!["src".to_string()],
        };
        assert_eq!(project.url, "http://test:8080");
        assert_eq!(project.port, 8080);
    }
    
    #[test]
    fn test_workspace_structure() {
        let workspace = WorkspaceConfig {
            commercial: ProjectConfig {
                dir: "bin/commercial".to_string(),
                url: "http://localhost:3000".to_string(),
                port: 3000,
                component_paths: vec!["src".to_string()],
            },
            storybook: ProjectConfig {
                dir: "libs/storybook".to_string(),
                url: "http://localhost:6006".to_string(),
                port: 6006,
                component_paths: vec!["stories".to_string()],
            },
        };
        assert_eq!(workspace.commercial.port, 3000);
        assert_eq!(workspace.storybook.port, 6006);
    }
    
    #[test]
    fn test_config_load_creates_default() {
        let config = Config::load();
        assert_eq!(config.base_url, "http://localhost:3000");
    }
}

// Created by: TEAM-DX-003
// BDD World for dx CLI integration tests

use cucumber::World;
use std::collections::HashMap;
use std::path::PathBuf;

/// Storybook server URL (port 6006)
pub const STORYBOOK_URL: &str = "http://localhost:6006";

#[derive(Debug, Default, World)]
pub struct DxWorld {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Command Results
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Last command result
    pub last_result: Option<Result<String, String>>,
    
    /// Error message if command failed
    pub error_message: Option<String>,
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // CSS Command State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Last CSS class check result
    pub class_exists: Option<bool>,
    
    /// Last extracted styles
    pub styles: HashMap<String, String>,
    
    /// Last extracted classes
    pub classes: Vec<String>,
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // HTML Command State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Last HTML element info
    pub element_tag: Option<String>,
    pub element_count: Option<usize>,
    
    /// Last attributes
    pub attributes: HashMap<String, String>,
    
    /// Last DOM tree
    pub dom_tree: Option<String>,
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Story File Locator State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Story file path
    pub story_file: Option<PathBuf>,
    
    /// Component file path
    pub component_file: Option<PathBuf>,
    
    /// Story path (relative)
    pub story_path: Option<String>,
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Timing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pub start_time: Option<std::time::Instant>,
}

impl DxWorld {
    /// Store success result
    pub fn store_success(&mut self, output: String) {
        self.last_result = Some(Ok(output));
        self.error_message = None;
    }
    
    /// Store error result
    pub fn store_error(&mut self, error: String) {
        self.last_result = Some(Err(error.clone()));
        self.error_message = Some(error);
    }
    
    /// Clear state for new scenario
    pub fn clear(&mut self) {
        self.last_result = None;
        self.class_exists = None;
        self.styles.clear();
        self.classes.clear();
        self.element_tag = None;
        self.element_count = None;
        self.attributes.clear();
        self.dom_tree = None;
        self.story_file = None;
        self.component_file = None;
        self.story_path = None;
        self.error_message = None;
    }
}

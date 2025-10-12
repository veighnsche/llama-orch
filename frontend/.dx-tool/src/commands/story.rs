// Created by: TEAM-DX-003
// Story file locator command

use crate::error::{DxError, Result};
use colored::*;
use std::path::{Path, PathBuf};

/// Story command handler
pub struct StoryCommand {
    storybook_base: PathBuf,
}

impl StoryCommand {
    pub fn new() -> Self {
        // Default to frontend/libs/storybook from repo root
        let storybook_base = PathBuf::from("frontend/libs/storybook");
        Self { storybook_base }
    }
    
    /// Parse Storybook URL and locate the story file
    pub fn locate_story_file(&self, url: &str) -> Result<StoryFileInfo> {
        // Parse URL: http://localhost:6006/story/stories-atoms-button-button-story-vue
        let story_path = self.parse_story_path(url)?;
        
        // Convert to filesystem path
        let story_file = self.resolve_story_file(&story_path)?;
        let component_file = self.find_component_file(&story_file);
        
        Ok(StoryFileInfo {
            url: url.to_string(),
            story_path,
            story_file,
            component_file,
        })
    }
    
    fn parse_story_path(&self, url: &str) -> Result<String> {
        // Extract story path from URL
        // stories-atoms-button-button-story-vue -> stories/atoms/Button/Button.story.vue
        // The URL format is: stories-<category>-<component>-<component>-story-vue
        // Where component name is repeated (once for dir, once for file)
        let url = url.split('?').next().unwrap_or(url); // Remove query params
        
        if let Some(story_part) = url.strip_prefix("http://localhost:6006/story/") {
            // Convert: stories-atoms-button-button-story-vue
            // To: stories/atoms/Button/Button.story.vue
            let parts: Vec<&str> = story_part.split('-').collect();
            
            if parts.len() < 4 {
                return Err(DxError::Parse("Could not parse story path from URL".to_string()));
            }
            
            // Find where "story" and "vue" are (should be last 2 parts)
            if parts.len() >= 2 && parts[parts.len() - 2] == "story" && parts[parts.len() - 1] == "vue" {
                // Remove "story" and "vue" from the end
                // stories-atoms-button-button-story-vue -> [stories, atoms, button, button]
                let mut path_parts: Vec<&str> = parts[..parts.len() - 2].to_vec();
                
                if path_parts.len() < 2 {
                    return Err(DxError::Parse("Could not extract component name".to_string()));
                }
                
                // Last part is the component name (repeated in URL)
                let component = path_parts.last().unwrap();
                let component_cap = capitalize(component);
                
                // Remove BOTH occurrences of the component name
                // [stories, atoms, button, button] -> [stories, atoms]
                path_parts.pop(); // Remove last "button"
                path_parts.pop(); // Remove second-to-last "button"
                
                // Build directory path: stories/atoms
                let dir_path = path_parts.join("/");
                
                // Final path: stories/atoms/Button/Button.story.vue
                Ok(format!("{}/{}/{}.story.vue", dir_path, component_cap, component_cap))
            } else {
                Err(DxError::Parse("Could not parse story path from URL".to_string()))
            }
        } else {
            Err(DxError::Parse("Could not parse story path from URL".to_string()))
        }
    }
    
    fn resolve_story_file(&self, story_path: &str) -> Result<PathBuf> {
        let full_path = self.storybook_base.join(story_path);
        
        if full_path.exists() {
            Ok(full_path)
        } else {
            Err(DxError::Parse(format!("Story file not found: {}", full_path.display())))
        }
    }
    
    fn find_component_file(&self, story_file: &Path) -> Option<PathBuf> {
        // Button.story.vue -> Button.vue
        let parent = story_file.parent()?;
        let stem = story_file.file_stem()?.to_str()?;
        let component_name = stem.strip_suffix(".story")?;
        
        let component_file = parent.join(format!("{}.vue", component_name));
        if component_file.exists() {
            Some(component_file)
        } else {
            None
        }
    }
    
    /// Print story file info
    pub fn print_story_info(&self, info: &StoryFileInfo) {
        println!("{} Story file located", "✓".green());
        println!("  URL: {}", info.url.cyan());
        println!("  File: {}", info.story_file.display().to_string().yellow());
        
        if let Some(component) = &info.component_file {
            println!("  Component: {}", component.display().to_string().yellow());
        }
        
        println!();
        println!("  To add more stories, edit:");
        println!("    - Story file: {}", info.story_path.cyan());
        if let Some(component) = &info.component_file {
            if let Some(name) = component.file_name() {
                println!("    - Component: {} (if needed)", name.to_string_lossy().cyan());
            }
        }
    }
}

impl Default for StoryCommand {
    fn default() -> Self {
        Self::new()
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[derive(Debug)]
pub struct StoryFileInfo {
    pub url: String,
    pub story_path: String,
    pub story_file: PathBuf,
    pub component_file: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("button"), "Button");
        assert_eq!(capitalize("navbar"), "Navbar");
        assert_eq!(capitalize(""), "");
    }

    #[test]
    fn test_parse_story_path() {
        let cmd = StoryCommand::new();
        
        let result = cmd.parse_story_path("http://localhost:6006/story/stories-atoms-button-button-story-vue");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "stories/atoms/Button/Button.story.vue");
    }

    #[test]
    fn test_parse_story_path_with_query() {
        let cmd = StoryCommand::new();
        
        let result = cmd.parse_story_path("http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=0");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "stories/atoms/Button/Button.story.vue");
    }

    #[test]
    fn test_parse_invalid_url() {
        let cmd = StoryCommand::new();
        
        let result = cmd.parse_story_path("http://localhost:6006/invalid");
        assert!(result.is_err());
    }
}

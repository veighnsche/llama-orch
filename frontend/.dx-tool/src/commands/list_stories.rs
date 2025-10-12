// Created by: TEAM-DX-004
// List available stories (components) from Histoire/Storybook sidebar

use crate::error::{DxError, Result};
use crate::fetcher::Fetcher;
use colored::*;
use scraper::{Html, Selector};
use std::collections::HashMap;

/// List stories command handler
pub struct ListStoriesCommand {
    fetcher: Fetcher,
}

impl ListStoriesCommand {
    pub fn new() -> Self {
        Self {
            fetcher: Fetcher::new(),
        }
    }
    
    /// List all available stories from Histoire
    pub async fn list_stories(&self, base_url: &str) -> Result<Vec<StoryInfo>> {
        let html = self.fetcher.fetch_page(base_url).await?;
        self.parse_stories(&html, base_url)
    }
    
    /// Parse stories from Histoire HTML
    fn parse_stories(&self, html: &str, base_url: &str) -> Result<Vec<StoryInfo>> {
        let document = Html::parse_document(html);
        
        // Histoire renders stories in the sidebar
        // Look for links with story URLs
        let link_selector = Selector::parse("a[href*='/story/']")
            .map_err(|e| DxError::Parse(format!("Invalid selector: {}", e)))?;
        
        let mut stories: HashMap<String, StoryInfo> = HashMap::new();
        
        for element in document.select(&link_selector) {
            if let Some(href) = element.value().attr("href") {
                // Extract story path from href
                if let Some(story_path) = href.strip_prefix("/story/") {
                    // Parse story info
                    if let Ok(info) = self.parse_story_info(story_path, base_url) {
                        // Use story_id as key to deduplicate
                        stories.insert(info.story_id.clone(), info);
                    }
                }
            }
        }
        
        let mut story_list: Vec<StoryInfo> = stories.into_values().collect();
        story_list.sort_by(|a, b| a.full_path.cmp(&b.full_path));
        
        Ok(story_list)
    }
    
    /// Parse story information from story path
    fn parse_story_info(&self, story_path: &str, base_url: &str) -> Result<StoryInfo> {
        // Remove query params
        let story_path = story_path.split('?').next().unwrap_or(story_path);
        
        // Parse: stories-atoms-button-button-story-vue
        let parts: Vec<&str> = story_path.split('-').collect();
        
        if parts.len() < 4 {
            return Err(DxError::Parse("Invalid story path".to_string()));
        }
        
        // Check if it ends with "story" and "vue"
        if parts.len() >= 2 && parts[parts.len() - 2] == "story" && parts[parts.len() - 1] == "vue" {
            // Remove "story" and "vue"
            let mut path_parts: Vec<&str> = parts[..parts.len() - 2].to_vec();
            
            if path_parts.len() < 2 {
                return Err(DxError::Parse("Invalid story path".to_string()));
            }
            
            // Last part is component name
            let component = path_parts.last().unwrap();
            let component_cap = capitalize(component);
            
            // Remove duplicate component names
            path_parts.pop();
            path_parts.pop();
            
            // First part should be "stories"
            if path_parts.first() == Some(&"stories") {
                path_parts.remove(0);
            }
            
            // Category is what's left (atoms, molecules, etc.)
            let category = if path_parts.is_empty() {
                "".to_string()
            } else {
                path_parts.join("/")
            };
            
            let url = format!("{}/story/{}", base_url.trim_end_matches('/'), story_path);
            let full_path = if category.is_empty() {
                component_cap.clone()
            } else {
                format!("{}/{}", category, component_cap)
            };
            
            Ok(StoryInfo {
                story_id: story_path.to_string(),
                component_name: component_cap,
                category,
                full_path,
                url,
                variant_count: 0, // Will be populated by list_variants
            })
        } else {
            Err(DxError::Parse("Invalid story path format".to_string()))
        }
    }
    
    /// Print stories in a tree format
    pub fn print_stories(&self, stories: &[StoryInfo]) {
        if stories.is_empty() {
            println!("{} No stories found", "✗".red());
            return;
        }
        
        println!("{} Found {} stories in Histoire\n", "✓".green(), stories.len());
        
        // Group by category
        let mut by_category: HashMap<String, Vec<&StoryInfo>> = HashMap::new();
        for story in stories {
            by_category
                .entry(story.category.clone())
                .or_insert_with(Vec::new)
                .push(story);
        }
        
        let mut categories: Vec<String> = by_category.keys().cloned().collect();
        categories.sort();
        
        for category in categories {
            if let Some(stories_in_cat) = by_category.get(&category) {
                let category_display = if category.is_empty() {
                    "(root)".to_string()
                } else {
                    category.clone()
                };
                
                println!("{}/", category_display.cyan().bold());
                
                for story in stories_in_cat {
                    println!("  {} {}", "•".green(), story.component_name.yellow());
                    println!("    {}", story.url.dimmed());
                }
                println!();
            }
        }
    }
}

impl Default for ListStoriesCommand {
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

#[derive(Debug, Clone)]
pub struct StoryInfo {
    pub story_id: String,
    pub component_name: String,
    pub category: String,
    pub full_path: String,
    pub url: String,
    pub variant_count: usize,
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
    fn test_parse_story_info() {
        let cmd = ListStoriesCommand::new();
        let result = cmd.parse_story_info(
            "stories-atoms-button-button-story-vue",
            "http://localhost:6006"
        );
        
        assert!(result.is_ok());
        let info = result.unwrap();
        assert_eq!(info.component_name, "Button");
        assert_eq!(info.category, "atoms");
        assert_eq!(info.full_path, "atoms/Button");
    }
}

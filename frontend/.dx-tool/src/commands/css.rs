// Created by: TEAM-DX-001
// TEAM-DX-002: Added selector and list-classes commands
// CSS verification commands

use crate::error::{DxError, Result};
use crate::fetcher::Fetcher;
use crate::parser::{HtmlParser, CssParser};
use colored::*;
use std::collections::HashMap;

/// CSS command handler
pub struct CssCommand {
    fetcher: Fetcher,
}

impl CssCommand {
    pub fn new() -> Self {
        Self {
            fetcher: Fetcher::new(),
        }
    }
    
    /// Check if a CSS class exists in the page's stylesheets
    pub async fn check_class_exists(&self, url: &str, class_name: &str) -> Result<bool> {
        // Fetch the HTML page
        let html = self.fetcher.fetch_page(url).await?;
        let parser = HtmlParser::parse(&html);
        
        // Extract inline styles
        let inline_styles = parser.extract_inline_styles();
        for css in &inline_styles {
            if CssParser::class_exists(css, class_name) {
                return Ok(true);
            }
        }
        
        // Extract and fetch external stylesheets
        let base_url = Self::extract_base_url(url)?;
        let stylesheet_urls = parser.extract_stylesheet_urls(&base_url);
        
        for stylesheet_url in stylesheet_urls {
            match self.fetcher.fetch_stylesheet(&stylesheet_url).await {
                Ok(css) => {
                    if CssParser::class_exists(&css, class_name) {
                        return Ok(true);
                    }
                }
                Err(e) => {
                    eprintln!("{} Failed to fetch stylesheet {}: {}", 
                        "⚠".yellow(), stylesheet_url, e);
                    // Continue checking other stylesheets
                }
            }
        }
        
        Ok(false)
    }
    
    /// Print result of class existence check
    pub fn print_class_exists_result(&self, class_name: &str, exists: bool) {
        if exists {
            println!("{} Class '{}' found in stylesheet", 
                "✓".green(), class_name.cyan());
        } else {
            eprintln!("{} Error: Class '{}' not found in stylesheet", 
                "✗".red(), class_name.cyan());
            eprintln!("  Possible causes:");
            eprintln!("    - Class not used in any component");
            eprintln!("    - Tailwind not scanning source files");
            eprintln!("    - Class tree-shaken by build tool");
        }
    }
    
    // TEAM-DX-002: Get computed styles for a selector
    /// Get computed styles for a CSS selector
    pub async fn get_selector_styles(&self, url: &str, selector: &str) -> Result<HashMap<String, String>> {
        // Fetch the HTML page
        let html = self.fetcher.fetch_page(url).await?;
        let parser = HtmlParser::parse(&html);
        
        // Find the element
        let elements = parser.select(selector)?;
        let element = &elements[0];
        
        // Extract classes from the element
        let classes = HtmlParser::extract_classes(element);
        
        // Fetch all stylesheets and extract styles for each class
        let mut all_styles = HashMap::new();
        
        // Extract inline styles
        let inline_styles = parser.extract_inline_styles();
        for css in &inline_styles {
            for class_name in &classes {
                let styles = CssParser::extract_styles_for_class(css, class_name);
                all_styles.extend(styles);
            }
        }
        
        // Extract and fetch external stylesheets
        let base_url = Self::extract_base_url(url)?;
        let stylesheet_urls = parser.extract_stylesheet_urls(&base_url);
        
        for stylesheet_url in stylesheet_urls {
            if let Ok(css) = self.fetcher.fetch_stylesheet(&stylesheet_url).await {
                for class_name in &classes {
                    let styles = CssParser::extract_styles_for_class(&css, class_name);
                    all_styles.extend(styles);
                }
            }
        }
        
        Ok(all_styles)
    }
    
    /// Print computed styles result
    pub fn print_selector_styles(&self, selector: &str, styles: &HashMap<String, String>) {
        println!("{} Selector: {}", "✓".green(), selector.cyan());
        println!("  Computed Styles:");
        
        if styles.is_empty() {
            println!("    {}", "(no styles found)".dimmed());
        } else {
            let mut sorted_styles: Vec<_> = styles.iter().collect();
            sorted_styles.sort_by_key(|(k, _)| *k);
            
            for (property, value) in sorted_styles {
                println!("    {}: {}", property.yellow(), value);
            }
        }
    }
    
    // TEAM-DX-002: List classes on an element
    /// List all classes on a selector
    pub async fn list_classes(&self, url: &str, selector: &str) -> Result<Vec<String>> {
        let html = self.fetcher.fetch_page(url).await?;
        let parser = HtmlParser::parse(&html);
        
        let elements = parser.select(selector)?;
        let element = &elements[0];
        
        Ok(HtmlParser::extract_classes(element))
    }
    
    /// Print classes list result
    pub fn print_classes_list(&self, selector: &str, classes: &[String]) {
        println!("{} Classes on {}", "✓".green(), selector.cyan());
        
        if classes.is_empty() {
            println!("  {}", "(no classes)".dimmed());
        } else {
            for class in classes {
                println!("  - {}", class.yellow());
            }
        }
    }
    
    /// Extract base URL from full URL
    fn extract_base_url(url: &str) -> Result<String> {
        let parsed = url::Url::parse(url)
            .map_err(|e| DxError::InvalidUrl(format!("{}: {}", url, e)))?;
        
        let host = parsed.host_str().ok_or_else(|| DxError::InvalidUrl(url.to_string()))?;
        
        if let Some(port) = parsed.port() {
            Ok(format!("{}://{}:{}", parsed.scheme(), host, port))
        } else {
            Ok(format!("{}://{}", parsed.scheme(), host))
        }
    }
}

impl Default for CssCommand {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_base_url() {
        let url = "http://localhost:3000/page";
        let base = CssCommand::extract_base_url(url).unwrap();
        assert_eq!(base, "http://localhost:3000");
    }
    
    #[test]
    fn test_extract_base_url_with_port() {
        let url = "http://localhost:3000";
        let base = CssCommand::extract_base_url(url).unwrap();
        assert_eq!(base, "http://localhost:3000");
    }
    
    // TEAM-DX-002: Unit tests for CSS commands
    
    #[test]
    fn test_css_command_creation() {
        let _cmd = CssCommand::new();
        assert!(true); // Command created successfully
    }
    
    #[test]
    fn test_css_command_default() {
        let _cmd = CssCommand::default();
        assert!(true); // Default trait works
    }
    
    #[test]
    fn test_print_class_exists_true() {
        let cmd = CssCommand::new();
        // Should not panic
        cmd.print_class_exists_result("cursor-pointer", true);
    }
    
    #[test]
    fn test_print_class_exists_false() {
        let cmd = CssCommand::new();
        // Should not panic
        cmd.print_class_exists_result("nonexistent-class", false);
    }
    
    #[test]
    fn test_print_selector_styles_empty() {
        use std::collections::HashMap;
        
        let cmd = CssCommand::new();
        let styles = HashMap::new();
        
        // Should not panic
        cmd.print_selector_styles(".test", &styles);
    }
    
    #[test]
    fn test_print_selector_styles_with_data() {
        use std::collections::HashMap;
        
        let cmd = CssCommand::new();
        let mut styles = HashMap::new();
        styles.insert("color".to_string(), "red".to_string());
        styles.insert("display".to_string(), "block".to_string());
        
        // Should not panic
        cmd.print_selector_styles("button", &styles);
    }
    
    #[test]
    fn test_print_classes_list_empty() {
        let cmd = CssCommand::new();
        let classes: Vec<String> = vec![];
        
        // Should not panic
        cmd.print_classes_list(".test", &classes);
    }
    
    #[test]
    fn test_print_classes_list_with_data() {
        let cmd = CssCommand::new();
        let classes = vec![
            "btn".to_string(),
            "btn-primary".to_string(),
            "hover:bg-blue-500".to_string(),
        ];
        
        // Should not panic
        cmd.print_classes_list("button", &classes);
    }
    
    #[test]
    fn test_extract_base_url_simple() {
        let url = "http://example.com/page";
        let base = CssCommand::extract_base_url(url).unwrap();
        assert_eq!(base, "http://example.com");
    }
    
    #[test]
    fn test_extract_base_url_https() {
        let url = "https://example.com:8080/path/to/page";
        let base = CssCommand::extract_base_url(url).unwrap();
        assert_eq!(base, "https://example.com:8080");
    }
    
    #[test]
    fn test_extract_base_url_invalid() {
        let url = "not-a-url";
        let result = CssCommand::extract_base_url(url);
        assert!(result.is_err());
    }
    
    // Integration tests require a running server
    #[tokio::test]
    #[ignore]
    async fn test_get_selector_styles() {
        let cmd = CssCommand::new();
        let styles = cmd.get_selector_styles("http://localhost:3000", "button").await;
        assert!(styles.is_ok());
    }
    
    #[tokio::test]
    #[ignore]
    async fn test_check_class_exists() {
        let cmd = CssCommand::new();
        let exists = cmd.check_class_exists("http://localhost:3000", "cursor-pointer").await;
        assert!(exists.is_ok());
    }
    
    #[tokio::test]
    #[ignore]
    async fn test_list_classes() {
        let cmd = CssCommand::new();
        let classes = cmd.list_classes("http://localhost:3000", "button").await;
        assert!(classes.is_ok());
    }
}

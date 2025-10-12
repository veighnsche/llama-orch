// Created by: TEAM-DX-001
// CSS verification commands

use crate::error::{DxError, Result};
use crate::fetcher::Fetcher;
use crate::parser::{HtmlParser, CssParser};
use colored::*;

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
}

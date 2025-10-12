// Created by: TEAM-DX-003
// Comprehensive inspect command - get HTML + all related CSS in one shot

use crate::error::Result;
use crate::fetcher::Fetcher;
use crate::parser::{HtmlParser, CssParser};
use colored::*;
use std::collections::HashMap;

/// Inspect command handler - combines HTML and CSS analysis
pub struct InspectCommand {
    fetcher: Fetcher,
}

impl InspectCommand {
    pub fn new() -> Self {
        Self {
            fetcher: Fetcher::new(),
        }
    }
    
    /// Inspect an element: get HTML structure + all related Tailwind CSS
    pub async fn inspect_element(&self, url: &str, selector: &str) -> Result<InspectResult> {
        // Fetch the HTML page
        let html = self.fetcher.fetch_page(url).await?;
        let parser = HtmlParser::parse(&html);
        
        // Find the element
        let elements = parser.select(selector)?;
        if elements.is_empty() {
            return Err(crate::error::DxError::Parse(format!("No elements found for selector '{}'", selector)));
        }
        
        let element = &elements[0];
        
        // Extract element info
        let tag = element.value().name().to_string();
        let classes = HtmlParser::extract_classes(element);
        let attributes = HtmlParser::extract_attributes(element);
        let text = HtmlParser::extract_text(element);
        let html_snippet = element.html();
        
        // Get all CSS for the classes on this element
        let mut class_styles: HashMap<String, HashMap<String, String>> = HashMap::new();
        
        if !classes.is_empty() {
            // Extract inline styles
            let inline_styles = parser.extract_inline_styles();
            
            // Extract and fetch external stylesheets
            let base_url = Self::extract_base_url(url)?;
            let stylesheet_urls = parser.extract_stylesheet_urls(&base_url);
            
            // Collect all CSS
            let mut all_css = String::new();
            for css in &inline_styles {
                all_css.push_str(css);
                all_css.push('\n');
            }
            
            for stylesheet_url in stylesheet_urls {
                match self.fetcher.fetch_stylesheet(&stylesheet_url).await {
                    Ok(css) => {
                        all_css.push_str(&css);
                        all_css.push('\n');
                    }
                    Err(_) => {
                        // Silently skip failed stylesheets
                    }
                }
            }
            
            // Extract styles for each class
            for class_name in &classes {
                let styles = CssParser::extract_styles_for_class(&all_css, class_name);
                if !styles.is_empty() {
                    class_styles.insert(class_name.clone(), styles);
                }
            }
        }
        
        Ok(InspectResult {
            selector: selector.to_string(),
            tag,
            classes,
            attributes,
            text,
            html_snippet,
            class_styles,
            element_count: elements.len(),
        })
    }
    
    fn extract_base_url(url: &str) -> Result<String> {
        let parsed = url::Url::parse(url)
            .map_err(|e| crate::error::DxError::Parse(format!("Invalid URL: {}", e)))?;
        
        let scheme = parsed.scheme();
        let host = parsed.host_str()
            .ok_or_else(|| crate::error::DxError::Parse("No host in URL".to_string()))?;
        
        if let Some(port) = parsed.port() {
            Ok(format!("{}://{}:{}", scheme, host, port))
        } else {
            Ok(format!("{}://{}", scheme, host))
        }
    }
    
    /// Print comprehensive inspection result
    pub fn print_inspect_result(&self, result: &InspectResult) {
        println!("{} Inspected: {}", "✓".green(), result.selector.cyan());
        println!();
        
        // Element info
        println!("{}", "Element:".bold());
        println!("  Tag: {}", result.tag.yellow());
        println!("  Count: {} element{}", 
            result.element_count,
            if result.element_count == 1 { "" } else { "s" });
        
        if !result.text.is_empty() {
            let display_text = if result.text.len() > 60 {
                format!("{}...", &result.text[..60])
            } else {
                result.text.clone()
            };
            println!("  Text: {}", display_text.dimmed());
        }
        println!();
        
        // Classes
        if !result.classes.is_empty() {
            println!("{}", "Classes:".bold());
            for class in &result.classes {
                println!("  • {}", class.cyan());
            }
            println!();
        }
        
        // Attributes
        if !result.attributes.is_empty() {
            println!("{}", "Attributes:".bold());
            for (key, value) in &result.attributes {
                let display_value = if value.len() > 50 {
                    format!("{}...", &value[..50])
                } else {
                    value.clone()
                };
                println!("  {}={}", key.yellow(), display_value.dimmed());
            }
            println!();
        }
        
        // CSS Styles for each class
        if !result.class_styles.is_empty() {
            println!("{}", "Tailwind CSS:".bold());
            for (class_name, styles) in &result.class_styles {
                println!("  .{} {{", class_name.cyan());
                for (prop, value) in styles {
                    println!("    {}: {};", prop.green(), value.dimmed());
                }
                println!("  }}");
                println!();
            }
        } else if !result.classes.is_empty() {
            println!("{} No CSS rules found for classes (may be utility-only)", "ℹ".blue());
            println!();
        }
        
        // HTML snippet
        println!("{}", "HTML:".bold());
        let display_html = if result.html_snippet.len() > 200 {
            format!("{}...", &result.html_snippet[..200])
        } else {
            result.html_snippet.clone()
        };
        println!("{}", display_html.dimmed());
    }
}

impl Default for InspectCommand {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct InspectResult {
    pub selector: String,
    pub tag: String,
    pub classes: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub text: String,
    pub html_snippet: String,
    pub class_styles: HashMap<String, HashMap<String, String>>,
    pub element_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspect_command_creation() {
        let cmd = InspectCommand::new();
        assert!(true); // Just verify it compiles
    }

    #[test]
    fn test_inspect_command_default() {
        let cmd = InspectCommand::default();
        assert!(true);
    }

    #[test]
    fn test_extract_base_url() {
        let result = InspectCommand::extract_base_url("http://localhost:6006/story/button");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "http://localhost:6006");
    }

    #[test]
    fn test_extract_base_url_https() {
        let result = InspectCommand::extract_base_url("https://example.com:8080/path");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "https://example.com:8080");
    }

    #[tokio::test]
    #[ignore]
    async fn test_inspect_element() {
        let cmd = InspectCommand::new();
        let result = cmd.inspect_element("http://localhost:6006", "button").await;
        assert!(result.is_ok());
    }
}

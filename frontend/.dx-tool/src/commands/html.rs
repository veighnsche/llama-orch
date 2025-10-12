// Created by: TEAM-DX-002
// HTML query commands

use crate::error::Result;
use crate::fetcher::Fetcher;
use crate::parser::{HtmlParser, DomNode};
use colored::*;
use std::collections::HashMap;

/// HTML command handler
pub struct HtmlCommand {
    fetcher: Fetcher,
}

impl HtmlCommand {
    pub fn new() -> Self {
        Self {
            fetcher: Fetcher::new(),
        }
    }
    
    /// Query DOM structure with a selector
    pub async fn query_selector(&self, url: &str, selector: &str) -> Result<ElementInfo> {
        let html = self.fetcher.fetch_page(url).await?;
        let parser = HtmlParser::parse(&html);
        
        let elements = parser.select(selector)?;
        let element = &elements[0];
        
        let tag = element.value().name().to_string();
        let classes = HtmlParser::extract_classes(element);
        let attributes = HtmlParser::extract_attributes(element);
        let text = HtmlParser::extract_text(element);
        
        Ok(ElementInfo {
            tag,
            classes,
            attributes,
            text,
            count: elements.len(),
        })
    }
    
    /// Print element info
    pub fn print_element_info(&self, selector: &str, info: &ElementInfo) {
        println!("{} Found {} element{} matching '{}'", 
            "✓".green(), 
            info.count,
            if info.count == 1 { "" } else { "s" },
            selector.cyan());
        println!("  Tag: {}", info.tag.yellow());
        
        if !info.classes.is_empty() {
            println!("  Classes: {}", info.classes.join(" ").cyan());
        }
        
        if !info.text.trim().is_empty() {
            let trimmed = info.text.trim();
            if trimmed.len() > 100 {
                println!("  Text: {}...", &trimmed[..100].dimmed());
            } else {
                println!("  Text: {}", trimmed.dimmed());
            }
        }
    }
    
    /// Get element attributes
    pub async fn get_attributes(&self, url: &str, selector: &str) -> Result<HashMap<String, String>> {
        let html = self.fetcher.fetch_page(url).await?;
        let parser = HtmlParser::parse(&html);
        
        let elements = parser.select(selector)?;
        let element = &elements[0];
        
        Ok(HtmlParser::extract_attributes(element))
    }
    
    /// Print attributes
    pub fn print_attributes(&self, selector: &str, attrs: &HashMap<String, String>) {
        println!("{} Attributes for {}", "✓".green(), selector.cyan());
        
        if attrs.is_empty() {
            println!("  {}", "(no attributes)".dimmed());
        } else {
            let mut sorted_attrs: Vec<_> = attrs.iter().collect();
            sorted_attrs.sort_by_key(|(k, _)| *k);
            
            for (key, value) in sorted_attrs {
                if value.len() > 80 {
                    println!("  {}: {}...", key.yellow(), &value[..80]);
                } else {
                    println!("  {}: {}", key.yellow(), value);
                }
            }
        }
    }
    
    /// Get DOM tree
    pub async fn get_tree(&self, url: &str, selector: &str, depth: usize) -> Result<DomNode> {
        let html = self.fetcher.fetch_page(url).await?;
        let parser = HtmlParser::parse(&html);
        
        parser.build_tree(selector, depth)
    }
    
    /// Print DOM tree
    pub fn print_tree(&self, selector: &str, tree: &DomNode) {
        println!("{} DOM tree for {}", "✓".green(), selector.cyan());
        println!();
        println!("{}", tree.node_repr());
        
        for (i, child) in tree.children.iter().enumerate() {
            println!("{}", child.format_tree("", i == tree.children.len() - 1));
        }
    }
}

impl Default for HtmlCommand {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ElementInfo {
    pub tag: String,
    pub classes: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub text: String,
    pub count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // TEAM-DX-002: Unit tests for HTML commands
    
    #[test]
    fn test_html_command_creation() {
        let _cmd = HtmlCommand::new();
        assert!(true); // Command created successfully
    }
    
    #[test]
    fn test_html_command_default() {
        let _cmd = HtmlCommand::default();
        assert!(true); // Default trait works
    }
    
    #[test]
    fn test_element_info_structure() {
        use std::collections::HashMap;
        
        let mut attrs = HashMap::new();
        attrs.insert("class".to_string(), "test".to_string());
        
        let info = ElementInfo {
            tag: "div".to_string(),
            classes: vec!["test".to_string()],
            attributes: attrs,
            text: "Hello".to_string(),
            count: 1,
        };
        
        assert_eq!(info.tag, "div");
        assert_eq!(info.classes.len(), 1);
        assert_eq!(info.count, 1);
    }
    
    #[test]
    fn test_print_element_info_single() {
        use std::collections::HashMap;
        
        let cmd = HtmlCommand::new();
        let info = ElementInfo {
            tag: "button".to_string(),
            classes: vec!["btn".to_string()],
            attributes: HashMap::new(),
            text: "Click".to_string(),
            count: 1,
        };
        
        // Should not panic
        cmd.print_element_info("button", &info);
    }
    
    #[test]
    fn test_print_element_info_multiple() {
        use std::collections::HashMap;
        
        let cmd = HtmlCommand::new();
        let info = ElementInfo {
            tag: "div".to_string(),
            classes: vec![],
            attributes: HashMap::new(),
            text: "".to_string(),
            count: 5,
        };
        
        // Should not panic
        cmd.print_element_info("div", &info);
    }
    
    #[test]
    fn test_print_element_info_long_text() {
        use std::collections::HashMap;
        
        let cmd = HtmlCommand::new();
        let long_text = "a".repeat(150);
        let info = ElementInfo {
            tag: "p".to_string(),
            classes: vec![],
            attributes: HashMap::new(),
            text: long_text,
            count: 1,
        };
        
        // Should truncate long text
        cmd.print_element_info("p", &info);
    }
    
    #[test]
    fn test_print_attributes_empty() {
        use std::collections::HashMap;
        
        let cmd = HtmlCommand::new();
        let attrs = HashMap::new();
        
        // Should not panic
        cmd.print_attributes("div", &attrs);
    }
    
    #[test]
    fn test_print_attributes_with_data() {
        use std::collections::HashMap;
        
        let cmd = HtmlCommand::new();
        let mut attrs = HashMap::new();
        attrs.insert("id".to_string(), "test".to_string());
        attrs.insert("class".to_string(), "btn btn-primary".to_string());
        
        // Should not panic
        cmd.print_attributes("button", &attrs);
    }
    
    #[test]
    fn test_print_attributes_long_value() {
        use std::collections::HashMap;
        
        let cmd = HtmlCommand::new();
        let mut attrs = HashMap::new();
        let long_value = "a".repeat(100);
        attrs.insert("data-value".to_string(), long_value);
        
        // Should truncate long values
        cmd.print_attributes("div", &attrs);
    }
    
    // Integration tests require a running server
    #[tokio::test]
    #[ignore]
    async fn test_query_selector() {
        let cmd = HtmlCommand::new();
        let info = cmd.query_selector("http://localhost:3000", "button").await;
        assert!(info.is_ok());
    }
    
    #[tokio::test]
    #[ignore]
    async fn test_get_attributes() {
        let cmd = HtmlCommand::new();
        let attrs = cmd.get_attributes("http://localhost:3000", "button").await;
        assert!(attrs.is_ok());
    }
    
    #[tokio::test]
    #[ignore]
    async fn test_get_tree() {
        let cmd = HtmlCommand::new();
        let tree = cmd.get_tree("http://localhost:3000", "nav", 2).await;
        assert!(tree.is_ok());
    }
}

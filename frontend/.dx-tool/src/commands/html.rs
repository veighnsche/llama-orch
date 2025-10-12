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
    
    // Integration tests require a running server
    #[tokio::test]
    #[ignore]
    async fn test_query_selector() {
        let cmd = HtmlCommand::new();
        let info = cmd.query_selector("http://localhost:3000", "button").await;
        assert!(info.is_ok());
    }
}

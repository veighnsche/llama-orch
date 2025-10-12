// Created by: TEAM-DX-001
// HTML parsing and querying

use crate::error::{DxError, Result};
use scraper::{Html, Selector, ElementRef};

/// HTML parser for DOM queries
pub struct HtmlParser {
    document: Html,
}

impl HtmlParser {
    /// Parse HTML document
    pub fn parse(html: &str) -> Self {
        let document = Html::parse_document(html);
        Self { document }
    }
    
    /// Select elements by CSS selector
    pub fn select(&self, selector_str: &str) -> Result<Vec<ElementRef<'_>>> {
        let selector = Selector::parse(selector_str)
            .map_err(|e| DxError::Parse(format!("Invalid selector '{}': {:?}", selector_str, e)))?;
        
        let elements: Vec<_> = self.document.select(&selector).collect();
        
        if elements.is_empty() {
            return Err(DxError::SelectorNotFound {
                selector: selector_str.to_string(),
            });
        }
        
        Ok(elements)
    }
    
    /// Check if selector exists
    pub fn selector_exists(&self, selector: &str) -> bool {
        self.select(selector).is_ok()
    }
    
    /// Extract all stylesheet links from HTML
    pub fn extract_stylesheet_urls(&self, base_url: &str) -> Vec<String> {
        let link_selector = Selector::parse("link[rel='stylesheet']")
            .expect("Valid selector");
        
        self.document
            .select(&link_selector)
            .filter_map(|element| {
                element.value().attr("href").map(|href| {
                    if href.starts_with("http") {
                        href.to_string()
                    } else if href.starts_with('/') {
                        format!("{}{}", base_url.trim_end_matches('/'), href)
                    } else {
                        format!("{}/{}", base_url.trim_end_matches('/'), href)
                    }
                })
            })
            .collect()
    }
    
    /// Extract inline styles from HTML
    pub fn extract_inline_styles(&self) -> Vec<String> {
        let style_selector = Selector::parse("style")
            .expect("Valid selector");
        
        self.document
            .select(&style_selector)
            .filter_map(|element| {
                element.text().collect::<String>().into()
            })
            .collect()
    }
    
    // TEAM-DX-002: Added element attribute and text extraction
    
    /// Extract attributes from an element
    pub fn extract_attributes(element: &ElementRef) -> std::collections::HashMap<String, String> {
        element.value()
            .attrs()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }
    
    /// Extract text content from an element
    pub fn extract_text(element: &ElementRef) -> String {
        element.text().collect::<Vec<_>>().join("")
    }
    
    /// Extract classes from an element
    pub fn extract_classes(element: &ElementRef) -> Vec<String> {
        element.value()
            .attr("class")
            .map(|classes| {
                classes.split_whitespace()
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Build a simple DOM tree representation
    pub fn build_tree(&self, selector_str: &str, max_depth: usize) -> Result<DomNode> {
        let elements = self.select(selector_str)?;
        if elements.is_empty() {
            return Err(DxError::SelectorNotFound {
                selector: selector_str.to_string(),
            });
        }
        
        let element = elements[0];
        Ok(Self::build_tree_recursive(element, 0, max_depth))
    }
    
    fn build_tree_recursive(element: ElementRef, depth: usize, max_depth: usize) -> DomNode {
        let tag = element.value().name().to_string();
        let classes = Self::extract_classes(&element);
        let id = element.value().attr("id").map(|s| s.to_string());
        
        let mut children = Vec::new();
        if depth < max_depth {
            for child in element.children() {
                if let Some(child_element) = ElementRef::wrap(child) {
                    children.push(Self::build_tree_recursive(child_element, depth + 1, max_depth));
                }
            }
        }
        
        DomNode {
            tag,
            classes,
            id,
            children,
        }
    }
}

// TEAM-DX-002: DOM tree structure
#[derive(Debug, Clone)]
pub struct DomNode {
    pub tag: String,
    pub classes: Vec<String>,
    pub id: Option<String>,
    pub children: Vec<DomNode>,
}

impl DomNode {
    /// Format tree with box-drawing characters
    pub fn format_tree(&self, prefix: &str, is_last: bool) -> String {
        let mut result = String::new();
        
        // Current node
        let connector = if is_last { "└── " } else { "├── " };
        let node_repr = self.node_repr();
        result.push_str(&format!("{}{}{}", prefix, connector, node_repr));
        
        // Children
        let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
        for (i, child) in self.children.iter().enumerate() {
            result.push('\n');
            result.push_str(&child.format_tree(&child_prefix, i == self.children.len() - 1));
        }
        
        result
    }
    
    /// Get string representation of node
    pub fn node_repr(&self) -> String {
        let mut repr = self.tag.clone();
        if let Some(id) = &self.id {
            repr.push_str(&format!("#{}", id));
        }
        if !self.classes.is_empty() {
            repr.push('.');
            repr.push_str(&self.classes.join("."));
        }
        repr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_html() {
        let html = r#"<html><body><div class="test">Hello</div></body></html>"#;
        let parser = HtmlParser::parse(html);
        assert!(parser.selector_exists(".test"));
    }
    
    #[test]
    fn test_select_elements() {
        let html = r#"<html><body><div class="test">Hello</div></body></html>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select(".test").unwrap();
        assert_eq!(elements.len(), 1);
    }
    
    #[test]
    fn test_selector_not_found() {
        let html = r#"<html><body><div class="test">Hello</div></body></html>"#;
        let parser = HtmlParser::parse(html);
        assert!(parser.select(".nonexistent").is_err());
    }
    
    #[test]
    fn test_extract_stylesheet_urls() {
        let html = r#"
            <html>
            <head>
                <link rel="stylesheet" href="/styles.css">
                <link rel="stylesheet" href="https://cdn.example.com/styles.css">
            </head>
            </html>
        "#;
        let parser = HtmlParser::parse(html);
        let urls = parser.extract_stylesheet_urls("http://localhost:3000");
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "http://localhost:3000/styles.css");
        assert_eq!(urls[1], "https://cdn.example.com/styles.css");
    }
    
    #[test]
    fn test_extract_inline_styles() {
        let html = r#"
            <html>
            <head>
                <style>.test { color: red; }</style>
            </head>
            </html>
        "#;
        let parser = HtmlParser::parse(html);
        let styles = parser.extract_inline_styles();
        assert_eq!(styles.len(), 1);
        assert!(styles[0].contains(".test"));
    }
    
    // TEAM-DX-002: Tests for new functionality
    #[test]
    fn test_extract_attributes() {
        let html = r#"<button aria-label="Toggle" type="button">Click</button>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("button").unwrap();
        let attrs = HtmlParser::extract_attributes(&elements[0]);
        assert_eq!(attrs.get("aria-label"), Some(&"Toggle".to_string()));
        assert_eq!(attrs.get("type"), Some(&"button".to_string()));
    }
    
    #[test]
    fn test_extract_attributes_empty() {
        let html = r#"<div>No attributes</div>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("div").unwrap();
        let attrs = HtmlParser::extract_attributes(&elements[0]);
        assert!(attrs.is_empty());
    }
    
    #[test]
    fn test_extract_attributes_multiple() {
        let html = r#"<input id="email" type="email" name="email" required placeholder="Enter email">"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("input").unwrap();
        let attrs = HtmlParser::extract_attributes(&elements[0]);
        assert_eq!(attrs.len(), 5);
        assert_eq!(attrs.get("type"), Some(&"email".to_string()));
    }
    
    #[test]
    fn test_extract_text() {
        let html = r#"<div>Hello World</div>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("div").unwrap();
        let text = HtmlParser::extract_text(&elements[0]);
        assert_eq!(text, "Hello World");
    }
    
    #[test]
    fn test_extract_text_empty() {
        let html = r#"<div></div>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("div").unwrap();
        let text = HtmlParser::extract_text(&elements[0]);
        assert_eq!(text, "");
    }
    
    #[test]
    fn test_extract_text_nested() {
        let html = r#"<div>Hello <span>World</span>!</div>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("div").unwrap();
        let text = HtmlParser::extract_text(&elements[0]);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }
    
    #[test]
    fn test_extract_classes() {
        let html = r#"<div class="foo bar baz">Test</div>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("div").unwrap();
        let classes = HtmlParser::extract_classes(&elements[0]);
        assert_eq!(classes, vec!["foo", "bar", "baz"]);
    }
    
    #[test]
    fn test_extract_classes_empty() {
        let html = r#"<div>No classes</div>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("div").unwrap();
        let classes = HtmlParser::extract_classes(&elements[0]);
        assert!(classes.is_empty());
    }
    
    #[test]
    fn test_extract_classes_single() {
        let html = r#"<div class="single">Test</div>"#;
        let parser = HtmlParser::parse(html);
        let elements = parser.select("div").unwrap();
        let classes = HtmlParser::extract_classes(&elements[0]);
        assert_eq!(classes, vec!["single"]);
    }
    
    #[test]
    fn test_build_tree() {
        let html = r#"
            <nav id="main-nav" class="navbar">
                <div class="container">
                    <a href="/">Home</a>
                    <button class="toggle">Menu</button>
                </div>
            </nav>
        "#;
        let parser = HtmlParser::parse(html);
        let tree = parser.build_tree("nav", 2).unwrap();
        assert_eq!(tree.tag, "nav");
        assert_eq!(tree.id, Some("main-nav".to_string()));
        assert!(tree.classes.contains(&"navbar".to_string()));
        assert_eq!(tree.children.len(), 1);
    }
    
    #[test]
    fn test_build_tree_depth_zero() {
        let html = r#"<div><span>Test</span></div>"#;
        let parser = HtmlParser::parse(html);
        let tree = parser.build_tree("div", 0).unwrap();
        assert_eq!(tree.tag, "div");
        assert_eq!(tree.children.len(), 0); // No children at depth 0
    }
    
    #[test]
    fn test_build_tree_not_found() {
        let html = r#"<div>Test</div>"#;
        let parser = HtmlParser::parse(html);
        let result = parser.build_tree("nav", 2);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_dom_node_repr_simple() {
        let node = DomNode {
            tag: "div".to_string(),
            classes: vec![],
            id: None,
            children: vec![],
        };
        assert_eq!(node.node_repr(), "div");
    }
    
    #[test]
    fn test_dom_node_repr_with_id() {
        let node = DomNode {
            tag: "div".to_string(),
            classes: vec![],
            id: Some("main".to_string()),
            children: vec![],
        };
        assert_eq!(node.node_repr(), "div#main");
    }
    
    #[test]
    fn test_dom_node_repr_with_classes() {
        let node = DomNode {
            tag: "div".to_string(),
            classes: vec!["foo".to_string(), "bar".to_string()],
            id: None,
            children: vec![],
        };
        assert_eq!(node.node_repr(), "div.foo.bar");
    }
    
    #[test]
    fn test_dom_node_repr_full() {
        let node = DomNode {
            tag: "nav".to_string(),
            classes: vec!["navbar".to_string(), "fixed".to_string()],
            id: Some("main-nav".to_string()),
            children: vec![],
        };
        assert_eq!(node.node_repr(), "nav#main-nav.navbar.fixed");
    }
    
    #[test]
    fn test_dom_node_format_tree_single() {
        let node = DomNode {
            tag: "div".to_string(),
            classes: vec![],
            id: None,
            children: vec![],
        };
        let formatted = node.format_tree("", true);
        assert!(formatted.contains("div"));
    }
    
    #[test]
    fn test_dom_node_format_tree_with_children() {
        let child = DomNode {
            tag: "span".to_string(),
            classes: vec![],
            id: None,
            children: vec![],
        };
        let parent = DomNode {
            tag: "div".to_string(),
            classes: vec![],
            id: None,
            children: vec![child],
        };
        let formatted = parent.format_tree("", true);
        assert!(formatted.contains("div"));
        assert!(formatted.contains("span"));
    }
}

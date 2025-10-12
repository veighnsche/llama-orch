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
}

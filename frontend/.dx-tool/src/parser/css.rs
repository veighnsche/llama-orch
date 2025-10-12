// Created by: TEAM-DX-001
// TEAM-DX-002: Added computed styles extraction
// CSS parsing and analysis

use std::collections::HashMap;

/// CSS parser for stylesheet analysis
pub struct CssParser;

impl CssParser {
    /// Check if a class exists in CSS content
    pub fn class_exists(css: &str, class_name: &str) -> bool {
        // Simple string search for now - can be enhanced with proper CSS parsing
        let patterns = [
            format!(".{} ", class_name),
            format!(".{}:", class_name),
            format!(".{}{}", class_name, "{"),
            format!(".{},", class_name),
            format!(".{}.", class_name),
        ];
        
        patterns.iter().any(|pattern| css.contains(pattern))
    }
    
    /// Extract all class names from CSS
    pub fn extract_classes(css: &str) -> Vec<String> {
        let mut classes = Vec::new();
        
        // Simple regex-like extraction
        for line in css.lines() {
            if let Some(start) = line.find('.') {
                let rest = &line[start + 1..];
                if let Some(end) = rest.find(|c: char| !c.is_alphanumeric() && c != '-' && c != '_') {
                    let class = &rest[..end];
                    if !class.is_empty() {
                        classes.push(class.to_string());
                    }
                }
            }
        }
        
        classes.sort();
        classes.dedup();
        classes
    }
    
    // TEAM-DX-002: Extract computed styles for a selector
    /// Extract CSS properties for a given class
    pub fn extract_styles_for_class(css: &str, class_name: &str) -> HashMap<String, String> {
        let mut styles = HashMap::new();
        let class_pattern = format!(".{}", class_name);
        
        // Simple extraction - find class definition and extract properties
        let mut in_class = false;
        let mut brace_count = 0;
        
        for line in css.lines() {
            let trimmed = line.trim();
            
            // Check if we're entering the class definition
            if trimmed.contains(&class_pattern) && trimmed.contains('{') {
                in_class = true;
                brace_count = 1;
                // Handle inline style definitions
                if let Some(content) = trimmed.split('{').nth(1) {
                    Self::parse_properties(content, &mut styles);
                }
                continue;
            }
            
            if in_class {
                // Track braces
                brace_count += trimmed.chars().filter(|&c| c == '{').count() as i32;
                brace_count -= trimmed.chars().filter(|&c| c == '}').count() as i32;
                
                if brace_count <= 0 {
                    in_class = false;
                    continue;
                }
                
                // Parse property: value pairs
                Self::parse_properties(trimmed, &mut styles);
            }
        }
        
        styles
    }
    
    fn parse_properties(line: &str, styles: &mut HashMap<String, String>) {
        if let Some(colon_pos) = line.find(':') {
            let property = line[..colon_pos].trim();
            let rest = &line[colon_pos + 1..];
            
            // Remove semicolon and closing brace, then trim again
            let value = rest
                .trim()
                .trim_end_matches('}')
                .trim()
                .trim_end_matches(';')
                .trim();
            
            if !property.is_empty() && !value.is_empty() {
                styles.insert(property.to_string(), value.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_class_exists() {
        let css = ".cursor-pointer { cursor: pointer; }";
        assert!(CssParser::class_exists(css, "cursor-pointer"));
    }
    
    #[test]
    fn test_class_not_exists() {
        let css = ".other-class { }";
        assert!(!CssParser::class_exists(css, "cursor-pointer"));
    }
    
    #[test]
    fn test_class_with_pseudo() {
        let css = ".cursor-pointer:hover { cursor: pointer; }";
        assert!(CssParser::class_exists(css, "cursor-pointer"));
    }
    
    #[test]
    fn test_extract_classes() {
        let css = r#"
            .cursor-pointer { cursor: pointer; }
            .text-red { color: red; }
            .hover\:bg-blue:hover { background: blue; }
        "#;
        let classes = CssParser::extract_classes(css);
        assert!(classes.contains(&"cursor-pointer".to_string()));
        assert!(classes.contains(&"text-red".to_string()));
    }
    
    // TEAM-DX-002: Tests for style extraction
    #[test]
    fn test_extract_styles_for_class() {
        let css = r#"
            .cursor-pointer {
                cursor: pointer;
                display: block;
            }
        "#;
        let styles = CssParser::extract_styles_for_class(css, "cursor-pointer");
        assert_eq!(styles.get("cursor"), Some(&"pointer".to_string()));
        assert_eq!(styles.get("display"), Some(&"block".to_string()));
    }
    
    #[test]
    fn test_extract_styles_inline() {
        let css = ".text-red { color: red; }";
        let styles = CssParser::extract_styles_for_class(css, "text-red");
        assert_eq!(styles.get("color"), Some(&"red".to_string()));
    }
}

// Created by: TEAM-DX-001
// CSS parsing and analysis

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
}

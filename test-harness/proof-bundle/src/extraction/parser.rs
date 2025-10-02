//! Source code metadata extraction
//!
//! Extracts metadata ONLY from test functions (marked with #[test]),
//! NOT from production code.
//!
//! # Example
//!
//! ```rust,ignore
//! /// @priority: critical
//! /// @spec: ORCH-1234
//! #[test]                    // ← Must have #[test] attribute
//! fn test_something() {      // ← Extract metadata from this
//!     assert!(true);
//! }
//!
//! /// This is production code
//! pub fn do_something() {    // ← NOT from this (no #[test])
//!     // ...
//! }
//! ```

use std::collections::HashMap;
use std::fs;
use crate::core::TestMetadata;
use crate::Result;

/// Extract metadata from source files
///
/// Only extracts from functions with #[test] attribute.
pub fn extract_metadata(targets: &[crate::discovery::TestTarget]) -> Result<HashMap<String, TestMetadata>> {
    let mut metadata_map = HashMap::new();
    
    for target in targets {
        // Read source file
        let source = fs::read_to_string(&target.src_path)
            .map_err(|e| crate::core::ProofBundleError::Io {
                operation: format!("read source file {}", target.src_path.display()),
                source: e,
            })?;
        
        // Parse with syn
        let ast = syn::parse_file(&source)
            .map_err(|e| crate::core::ProofBundleError::ParseError {
                context: format!("source file {}", target.src_path.display()),
                source: Box::new(e),
            })?;
        
        // Find test functions and extract metadata
        for item in ast.items {
            if let syn::Item::Fn(func) = item {
                // CRITICAL: Only process functions with #[test] attribute
                if has_test_attribute(&func) {
                    let test_name = func.sig.ident.to_string();
                    let metadata = extract_from_function(&func);
                    metadata_map.insert(test_name, metadata);
                }
            }
        }
    }
    
    Ok(metadata_map)
}

/// Check if a function has the #[test] attribute
fn has_test_attribute(func: &syn::ItemFn) -> bool {
    func.attrs.iter().any(|attr| {
        attr.path().is_ident("test")
    })
}

/// Extract metadata from a test function's doc comments
fn extract_from_function(func: &syn::ItemFn) -> TestMetadata {
    let mut metadata = TestMetadata::default();
    
    // Extract doc comments
    for attr in &func.attrs {
        if attr.path().is_ident("doc") {
            // Parse the doc attribute to get the comment text
            if let syn::Meta::NameValue(meta) = &attr.meta {
                if let syn::Expr::Lit(expr_lit) = &meta.value {
                    if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                        let doc = lit_str.value();
                        metadata = super::annotations::parse_annotations(&doc, metadata);
                    }
                }
            }
        }
    }
    
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_has_test_attribute() {
        let code = r#"
            #[test]
            fn my_test() {}
            
            fn not_a_test() {}
        "#;
        
        let ast = syn::parse_file(code).unwrap();
        
        let mut test_count = 0;
        for item in ast.items {
            if let syn::Item::Fn(func) = item {
                if has_test_attribute(&func) {
                    test_count += 1;
                }
            }
        }
        
        assert_eq!(test_count, 1, "Should only find functions with #[test]");
    }
}

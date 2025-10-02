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

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;
use crate::core::TestMetadata;
use crate::Result;
use super::MetadataCache;

/// Extract metadata from source files
///
/// Only extracts from functions with #[test] attribute.
pub fn extract_metadata(targets: &[crate::discovery::TestTarget]) -> Result<HashMap<String, TestMetadata>> {
    let mut metadata_map = HashMap::new();

    // Determine manifest_dir from the first target (single package per call)
    let manifest_dir: PathBuf = targets
        .get(0)
        .map(|t| t.manifest_dir.clone())
        .unwrap_or_else(|| PathBuf::from("."));

    // Collect candidate source files: target src_path + all under src/ and tests/
    let mut files: HashSet<PathBuf> = HashSet::new();
    for t in targets {
        files.insert(t.src_path.clone());
    }
    for sub in ["src", "tests"].iter() {
        let dir = manifest_dir.join(sub);
        if dir.exists() {
            for entry in WalkDir::new(&dir).into_iter().filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.is_file() && p.extension() == Some(OsStr::new("rs")) {
                    files.insert(p.to_path_buf());
                }
            }
        }
    }

    let sources: Vec<PathBuf> = files.into_iter().collect();

    // Use cache when fresh
    let cache = MetadataCache::new(&manifest_dir);
    if cache.is_fresh(&sources) {
        if let Some(map) = cache.load() {
            return Ok(map);
        }
    }

    for file in &sources {
        // Read source file
        let source = fs::read_to_string(file)
            .map_err(|e| crate::core::ProofBundleError::Io {
                operation: format!("read source file {}", file.display()),
                source: e,
            })?;

        // Parse with syn
        let ast = syn::parse_file(&source)
            .map_err(|e| crate::core::ProofBundleError::ParseError {
                context: format!("source file {}", file.display()),
                source: Box::new(e),
            })?;

        // Find test functions and extract metadata
        for item in ast.items {
            if let syn::Item::Fn(func) = item {
                // Only functions with #[test]
                if has_test_attribute(&func) {
                    let test_name = func.sig.ident.to_string();
                    let metadata = extract_from_function(&func);
                    metadata_map.insert(test_name, metadata);
                }
            }
        }
    }

    // Save cache for future runs
    let _ = cache.save(&metadata_map);

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

// Debug metadata extraction to see what's happening
use proof_bundle::discovery;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

fn main() -> anyhow::Result<()> {
    println!("🔍 Debugging metadata extraction...\n");
    
    // 1. Discover test targets
    let targets = discovery::discover_tests("proof-bundle")?;
    let manifest_dir = targets.get(0).unwrap().manifest_dir.clone();
    
    println!("Manifest dir: {}", manifest_dir.display());
    
    // 2. Collect all source files (same logic as parser)
    let mut files: HashSet<PathBuf> = HashSet::new();
    for t in &targets {
        files.insert(t.src_path.clone());
    }
    for sub in ["src", "tests"].iter() {
        let dir = manifest_dir.join(sub);
        if dir.exists() {
            println!("\nScanning directory: {}", dir.display());
            for entry in WalkDir::new(&dir).into_iter().filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.is_file() && p.extension() == Some(OsStr::new("rs")) {
                    files.insert(p.to_path_buf());
                }
            }
        }
    }
    
    println!("\n📁 Found {} Rust source files:", files.len());
    let mut sorted_files: Vec<_> = files.iter().collect();
    sorted_files.sort();
    for (i, file) in sorted_files.iter().enumerate() {
        println!("   {}. {}", i + 1, file.strip_prefix(&manifest_dir).unwrap_or(file).display());
    }
    
    // 3. Parse one file to see if we find test functions
    let status_file = manifest_dir.join("src/core/status.rs");
    if status_file.exists() {
        println!("\n🔬 Parsing src/core/status.rs...");
        let source = fs::read_to_string(&status_file)?;
        let ast = syn::parse_file(&source)?;
        
        let mut test_count = 0;
        for item in &ast.items {
            if let syn::Item::Mod(module) = item {
                if module.ident == "tests" {
                    println!("   Found 'tests' module!");
                    if let Some((_, items)) = &module.content {
                        for item in items {
                            if let syn::Item::Fn(func) = item {
                                // Check for #[test] attribute
                                let has_test = func.attrs.iter().any(|attr| {
                                    attr.path().is_ident("test")
                                });
                                if has_test {
                                    test_count += 1;
                                    println!("      - {} (has #[test])", func.sig.ident);
                                    
                                    // Check for doc comments
                                    let doc_comments: Vec<_> = func.attrs.iter()
                                        .filter_map(|attr| {
                                            if attr.path().is_ident("doc") {
                                                attr.meta.require_name_value().ok()
                                                    .and_then(|nv| {
                                                        if let syn::Expr::Lit(lit) = &nv.value {
                                                            if let syn::Lit::Str(s) = &lit.lit {
                                                                return Some(s.value());
                                                            }
                                                        }
                                                        None
                                                    })
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    
                                    if !doc_comments.is_empty() {
                                        println!("        Doc comments:");
                                        for comment in &doc_comments {
                                            println!("          {}", comment.trim());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("   Total test functions found: {}", test_count);
    }
    
    Ok(())
}

// PKGBUILD parser for worker installation
// Parses Arch Linux PKGBUILD format to extract build instructions

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct PkgBuild {
    /// Package name
    pub pkgname: String,
    
    /// Package version
    pub pkgver: String,
    
    /// Package release number
    pub pkgrel: String,
    
    /// Package description
    pub pkgdesc: String,
    
    /// Architecture (x86_64, aarch64, any)
    pub arch: Vec<String>,
    
    /// License (GPL-3.0-or-later, MIT, etc.)
    pub license: Vec<String>,
    
    /// Runtime dependencies
    pub depends: Vec<String>,
    
    /// Build dependencies
    pub makedepends: Vec<String>,
    
    /// Source URLs
    pub source: Vec<String>,
    
    /// SHA256 checksums
    pub sha256sums: Vec<String>,
    
    /// Build function body
    pub build_fn: Option<String>,
    
    /// Package function body
    pub package_fn: Option<String>,
    
    /// Raw variables (for custom fields)
    pub variables: HashMap<String, String>,
}

impl PkgBuild {
    /// Parse a PKGBUILD file from string content
    pub fn parse(content: &str) -> Result<Self, ParseError> {
        let mut parser = Parser::new(content);
        parser.parse()
    }
    
    /// Parse a PKGBUILD file from path
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ParseError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ParseError::IoError(e.to_string()))?;
        Self::parse(&content)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("IO error: {0}")]
    IoError(String),
    
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Invalid array syntax at line {0}")]
    InvalidArray(usize),
    
    #[error("Invalid function syntax at line {0}")]
    InvalidFunction(usize),
}

struct Parser {
    lines: Vec<String>,
    current_line: usize,
}

impl Parser {
    fn new(content: &str) -> Self {
        let lines = content
            .lines()
            .map(|l| l.trim().to_string())
            .collect();
        
        Self {
            lines,
            current_line: 0,
        }
    }
    
    fn parse(&mut self) -> Result<PkgBuild, ParseError> {
        let mut pkgname = None;
        let mut pkgver = None;
        let mut pkgrel = None;
        let mut pkgdesc = None;
        let mut arch = Vec::new();
        let mut license = Vec::new();
        let mut depends = Vec::new();
        let mut makedepends = Vec::new();
        let mut source = Vec::new();
        let mut sha256sums = Vec::new();
        let mut build_fn = None;
        let mut package_fn = None;
        let mut variables = HashMap::new();
        
        while self.current_line < self.lines.len() {
            let line = &self.lines[self.current_line].clone();
            
            // Skip comments and empty lines
            if line.starts_with('#') || line.is_empty() {
                self.current_line += 1;
                continue;
            }
            
            // Parse variable assignments
            if let Some((key, value)) = self.parse_assignment(line) {
                match key.as_str() {
                    "pkgname" => pkgname = Some(self.unquote(&value)),
                    "pkgver" => pkgver = Some(self.unquote(&value)),
                    "pkgrel" => pkgrel = Some(self.unquote(&value)),
                    "pkgdesc" => pkgdesc = Some(self.unquote(&value)),
                    "arch" => arch = self.parse_array(&value)?,
                    "license" => license = self.parse_array(&value)?,
                    "depends" => depends = self.parse_array(&value)?,
                    "makedepends" => makedepends = self.parse_array(&value)?,
                    "source" => source = self.parse_array(&value)?,
                    "sha256sums" => sha256sums = self.parse_array(&value)?,
                    _ => {
                        variables.insert(key, self.unquote(&value));
                    }
                }
                self.current_line += 1;
                continue;
            }
            
            // Parse functions
            if line.starts_with("build()") {
                build_fn = Some(self.parse_function()?);
                continue;
            }
            
            if line.starts_with("package()") {
                package_fn = Some(self.parse_function()?);
                continue;
            }
            
            self.current_line += 1;
        }
        
        // Validate required fields
        let pkgname = pkgname.ok_or_else(|| ParseError::MissingField("pkgname".to_string()))?;
        let pkgver = pkgver.ok_or_else(|| ParseError::MissingField("pkgver".to_string()))?;
        let pkgrel = pkgrel.ok_or_else(|| ParseError::MissingField("pkgrel".to_string()))?;
        let pkgdesc = pkgdesc.unwrap_or_default();
        
        Ok(PkgBuild {
            pkgname,
            pkgver,
            pkgrel,
            pkgdesc,
            arch,
            license,
            depends,
            makedepends,
            source,
            sha256sums,
            build_fn,
            package_fn,
            variables,
        })
    }
    
    fn parse_assignment(&self, line: &str) -> Option<(String, String)> {
        if let Some(pos) = line.find('=') {
            let key = line[..pos].trim().to_string();
            let value = line[pos + 1..].trim().to_string();
            Some((key, value))
        } else {
            None
        }
    }
    
    fn parse_array(&self, value: &str) -> Result<Vec<String>, ParseError> {
        let value = value.trim();
        
        // Single-line array: arch=('x86_64')
        if value.starts_with('(') && value.ends_with(')') {
            let inner = &value[1..value.len() - 1];
            return Ok(inner
                .split_whitespace()
                .map(|s| self.unquote(s))
                .filter(|s| !s.is_empty())
                .collect());
        }
        
        // Multi-line array (not implemented yet, would need lookahead)
        // For now, treat as single value
        Ok(vec![self.unquote(value)])
    }
    
    fn parse_function(&mut self) -> Result<String, ParseError> {
        let start_line = self.current_line;
        self.current_line += 1; // Skip function declaration
        
        let mut body = String::new();
        let mut brace_count = 1;
        
        while self.current_line < self.lines.len() {
            let line = &self.lines[self.current_line];
            
            // Count braces
            for ch in line.chars() {
                match ch {
                    '{' => brace_count += 1,
                    '}' => brace_count -= 1,
                    _ => {}
                }
            }
            
            if brace_count == 0 {
                self.current_line += 1;
                break;
            }
            
            body.push_str(line);
            body.push('\n');
            self.current_line += 1;
        }
        
        if brace_count != 0 {
            return Err(ParseError::InvalidFunction(start_line));
        }
        
        Ok(body.trim().to_string())
    }
    
    fn unquote(&self, s: &str) -> String {
        let s = s.trim();
        if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
            s[1..s.len() - 1].to_string()
        } else {
            s.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Parse complete PKGBUILD with all fields
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_parse_complete_pkgbuild_extracts_all_fields() {
        let content = r#"
# Maintainer: Test <test@example.com>
pkgname=llm-worker-rbee-cpu
pkgver=0.1.0
pkgrel=1
pkgdesc="LLM worker for rbee system (CPU-only)"
arch=('x86_64' 'aarch64')
license=('GPL-3.0-or-later')
depends=('gcc')
makedepends=('rust' 'cargo')
source=("https://github.com/user/llama-orch.git")
sha256sums=('SKIP')

build() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    cargo build --release --features cpu --bin llm-worker-rbee-cpu
}

package() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    install -Dm755 target/release/llm-worker-rbee-cpu "$pkgdir/usr/local/bin/llm-worker-rbee-cpu"
}
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        // Verify all fields extracted correctly
        assert_eq!(pkgbuild.pkgname, "llm-worker-rbee-cpu");
        assert_eq!(pkgbuild.pkgver, "0.1.0");
        assert_eq!(pkgbuild.pkgrel, "1");
        assert_eq!(pkgbuild.pkgdesc, "LLM worker for rbee system (CPU-only)");
        assert_eq!(pkgbuild.arch, vec!["x86_64", "aarch64"]);
        assert_eq!(pkgbuild.license, vec!["GPL-3.0-or-later"]);
        assert_eq!(pkgbuild.depends, vec!["gcc"]);
        assert_eq!(pkgbuild.makedepends, vec!["rust", "cargo"]);
        assert_eq!(pkgbuild.source, vec!["https://github.com/user/llama-orch.git"]);
        assert_eq!(pkgbuild.sha256sums, vec!["SKIP"]);
        
        // Verify functions extracted
        assert!(pkgbuild.build_fn.is_some());
        assert!(pkgbuild.package_fn.is_some());
        
        let build_fn = pkgbuild.build_fn.unwrap();
        assert!(build_fn.contains("cargo build"));
        assert!(build_fn.contains("--features cpu"));
        
        let package_fn = pkgbuild.package_fn.unwrap();
        assert!(package_fn.contains("install -Dm755"));
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Comments are ignored
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_comments_are_ignored() {
        let content = r#"
# This is a comment
pkgname=test-pkg
# Another comment
pkgver=1.0.0
# pkgrel=999  <- This should be ignored
pkgrel=1
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert_eq!(pkgbuild.pkgname, "test-pkg");
        assert_eq!(pkgbuild.pkgver, "1.0.0");
        assert_eq!(pkgbuild.pkgrel, "1");  // Not 999
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Quote removal works for single and double quotes
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_quote_removal_for_all_quote_types() {
        let content = r#"
pkgname="double-quoted"
pkgver='single-quoted'
pkgrel=unquoted
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert_eq!(pkgbuild.pkgname, "double-quoted");
        assert_eq!(pkgbuild.pkgver, "single-quoted");
        assert_eq!(pkgbuild.pkgrel, "unquoted");
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Array parsing with multiple elements
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_array_parsing_with_multiple_elements() {
        let content = r#"
pkgname=test
pkgver=1.0.0
pkgrel=1
arch=('x86_64' 'aarch64' 'armv7h')
depends=('gcc' 'glibc' 'zlib')
makedepends=('rust' 'cargo' 'cmake' 'make')
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert_eq!(pkgbuild.arch, vec!["x86_64", "aarch64", "armv7h"]);
        assert_eq!(pkgbuild.depends, vec!["gcc", "glibc", "zlib"]);
        assert_eq!(pkgbuild.makedepends, vec!["rust", "cargo", "cmake", "make"]);
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Empty arrays are handled
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_empty_arrays_are_handled() {
        let content = r#"
pkgname=test
pkgver=1.0.0
pkgrel=1
arch=()
depends=()
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert_eq!(pkgbuild.arch, Vec::<String>::new());
        assert_eq!(pkgbuild.depends, Vec::<String>::new());
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Build function with nested braces
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_build_function_with_nested_braces() {
        let content = r#"
pkgname=test
pkgver=1.0.0
pkgrel=1

build() {
    if [ -f "Makefile" ]; then
        make
    else {
        echo "No Makefile"
    }
    fi
}
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        let build_fn = pkgbuild.build_fn.unwrap();
        assert!(build_fn.contains("if [ -f \"Makefile\" ]"));
        assert!(build_fn.contains("make"));
        assert!(build_fn.contains("echo \"No Makefile\""));
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Package function is optional
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_package_function_is_optional() {
        let content = r#"
pkgname=test
pkgver=1.0.0
pkgrel=1

build() {
    echo "Building"
}
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert!(pkgbuild.build_fn.is_some());
        assert!(pkgbuild.package_fn.is_none());
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Build function is optional
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_build_function_is_optional() {
        let content = r#"
pkgname=test
pkgver=1.0.0
pkgrel=1

package() {
    echo "Packaging"
}
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert!(pkgbuild.build_fn.is_none());
        assert!(pkgbuild.package_fn.is_some());
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Missing required field returns error
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_missing_pkgname_returns_error() {
        let content = r#"
pkgver=1.0.0
pkgrel=1
"#;
        
        let result = PkgBuild::parse(content);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ParseError::MissingField(field) => {
                assert_eq!(field, "pkgname");
            }
            _ => panic!("Expected MissingField error"),
        }
    }
    
    #[test]
    fn test_missing_pkgver_returns_error() {
        let content = r#"
pkgname=test
pkgrel=1
"#;
        
        let result = PkgBuild::parse(content);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ParseError::MissingField(field) => {
                assert_eq!(field, "pkgver");
            }
            _ => panic!("Expected MissingField error"),
        }
    }
    
    #[test]
    fn test_missing_pkgrel_returns_error() {
        let content = r#"
pkgname=test
pkgver=1.0.0
"#;
        
        let result = PkgBuild::parse(content);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ParseError::MissingField(field) => {
                assert_eq!(field, "pkgrel");
            }
            _ => panic!("Expected MissingField error"),
        }
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Custom variables are captured
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_custom_variables_are_captured() {
        let content = r#"
pkgname=test
pkgver=1.0.0
pkgrel=1
_custom_var="custom value"
_another_var='another value'
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert_eq!(pkgbuild.variables.get("_custom_var"), Some(&"custom value".to_string()));
        assert_eq!(pkgbuild.variables.get("_another_var"), Some(&"another value".to_string()));
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Empty lines and whitespace are handled
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_empty_lines_and_whitespace_are_handled() {
        let content = r#"

pkgname=test

pkgver=1.0.0

pkgrel=1

"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert_eq!(pkgbuild.pkgname, "test");
        assert_eq!(pkgbuild.pkgver, "1.0.0");
        assert_eq!(pkgbuild.pkgrel, "1");
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Real-world CUDA worker PKGBUILD
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_real_world_cuda_worker_pkgbuild() {
        let content = r#"
# Maintainer: rbee team
pkgname=llm-worker-rbee-cuda
pkgver=0.1.0
pkgrel=1
pkgdesc="LLM worker for rbee system (NVIDIA CUDA)"
arch=('x86_64')
license=('GPL-3.0-or-later')
depends=('gcc' 'cuda')
makedepends=('rust' 'cargo')
source=("https://github.com/user/llama-orch.git")
sha256sums=('SKIP')

build() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    cargo build --release --features cuda --bin llm-worker-rbee-cuda
}

package() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    install -Dm755 target/release/llm-worker-rbee-cuda \
        "$pkgdir/usr/local/bin/llm-worker-rbee-cuda"
}
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        assert_eq!(pkgbuild.pkgname, "llm-worker-rbee-cuda");
        assert_eq!(pkgbuild.arch, vec!["x86_64"]);
        assert_eq!(pkgbuild.depends, vec!["gcc", "cuda"]);
        
        let build_fn = pkgbuild.build_fn.unwrap();
        assert!(build_fn.contains("--features cuda"));
        assert!(build_fn.contains("llm-worker-rbee-cuda"));
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: Description defaults to empty if missing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_optional_fields_default_to_empty() {
        let content = r#"
pkgname=test
pkgver=1.0.0
pkgrel=1
"#;
        
        let pkgbuild = PkgBuild::parse(content).unwrap();
        
        // Optional fields should be empty/default
        assert_eq!(pkgbuild.pkgdesc, "");
        assert_eq!(pkgbuild.arch, Vec::<String>::new());
        assert_eq!(pkgbuild.license, Vec::<String>::new());
        assert_eq!(pkgbuild.depends, Vec::<String>::new());
        assert_eq!(pkgbuild.makedepends, Vec::<String>::new());
        assert_eq!(pkgbuild.source, Vec::<String>::new());
        assert_eq!(pkgbuild.sha256sums, Vec::<String>::new());
        assert!(pkgbuild.build_fn.is_none());
        assert!(pkgbuild.package_fn.is_none());
    }
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BEHAVIOR: File reading works
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    #[test]
    fn test_from_file_reads_and_parses() {
        use std::io::Write;
        use tempfile::NamedTempFile;
        
        let content = r#"
pkgname=file-test
pkgver=2.0.0
pkgrel=5
"#;
        
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(content.as_bytes()).unwrap();
        temp_file.flush().unwrap();
        
        let pkgbuild = PkgBuild::from_file(temp_file.path()).unwrap();
        
        assert_eq!(pkgbuild.pkgname, "file-test");
        assert_eq!(pkgbuild.pkgver, "2.0.0");
        assert_eq!(pkgbuild.pkgrel, "5");
    }
}

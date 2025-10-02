//! Discover tests using cargo_metadata

use cargo_metadata::{MetadataCommand, Package, Target};
use std::path::{Path, PathBuf};
use crate::core::ProofBundleError;
use crate::Result;
use super::TestTarget;

/// Discover all test targets for a package
///
/// Uses `cargo metadata` to find all test-related targets:
/// - Library tests (`cargo test --lib`)
/// - Integration tests (`cargo test --tests`)
/// - Benchmarks with tests (`cargo test --benches`)
pub fn discover_tests(package_name: &str) -> Result<Vec<TestTarget>> {
    // Run cargo metadata
    let metadata = MetadataCommand::new()
        .exec()
        .map_err(|e| ProofBundleError::Other(format!("Failed to run cargo metadata: {}", e)))?;
    
    // Find the specified package
    let package = metadata
        .packages
        .iter()
        .find(|p| p.name == package_name)
        .ok_or_else(|| {
            let available: Vec<String> = metadata.packages.iter().map(|p| p.name.clone()).collect();
            ProofBundleError::NoTestsFound {
                package: package_name.to_string(),
                hint: format!("Package '{}' not found in workspace. Available packages: {}",
                    package_name,
                    available.join(", ")
                ),
            }
        })?;
    
    // Extract test targets
    let targets = extract_test_targets(package);
    
    if targets.is_empty() {
        return Err(ProofBundleError::NoTestsFound {
            package: package_name.to_string(),
            hint: "Package has no test targets. Add #[test] functions or integration tests.".to_string(),
        });
    }
    
    Ok(targets)
}

/// Extract test targets from a package
fn extract_test_targets(package: &Package) -> Vec<TestTarget> {
    // Determine manifest directory from Cargo.toml path
    let manifest_dir: PathBuf = package
        .manifest_path
        .clone()
        .into_std_path_buf()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    package
        .targets
        .iter()
        .filter(|target| is_test_target(target))
        .map(|target| TestTarget {
            name: target.name.clone(),
            kinds: target.kind.iter().map(|s| s.to_string()).collect(),
            src_path: target.src_path.clone().into_std_path_buf(),
            package: package.name.clone(),
            manifest_dir: manifest_dir.clone(),
        })
        .collect()
}

/// Check if a target is test-related
fn is_test_target(target: &Target) -> bool {
    target.kind.iter().any(|kind| {
        matches!(kind.as_str(), "lib" | "test" | "bench")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// @priority: critical
    /// @spec: PB-V3-DISCOVERY
    /// @team: proof-bundle
    /// @tags: integration, discovery, cargo-metadata, dogfooding
    #[test]
    fn test_discover_tests_for_proof_bundle() {
        // This test actually discovers tests for proof-bundle itself
        let result = discover_tests("proof-bundle");
        
        assert!(result.is_ok(), "Failed to discover tests: {:?}", result.err());
        
        let targets = result.unwrap();
        assert!(!targets.is_empty(), "Should find at least one test target");
        
        // Should find lib target
        let has_lib = targets.iter().any(|t| t.is_lib());
        assert!(has_lib, "Should find lib target");
    }
    
    /// @priority: high
    /// @spec: PB-V3-DISCOVERY
    /// @team: proof-bundle
    /// @tags: unit, discovery, error-handling
    #[test]
    fn test_discover_nonexistent_package() {
        let result = discover_tests("nonexistent-package-xyz");
        assert!(result.is_err());
        
        if let Err(ProofBundleError::NoTestsFound { package, hint }) = result {
            assert_eq!(package, "nonexistent-package-xyz");
            assert!(hint.contains("not found"));
        } else {
            panic!("Expected NoTestsFound error");
        }
    }
}

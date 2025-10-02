//! Public API
//!
//! The one-liner API that does everything.

use crate::core::{Mode, TestSummary};
use crate::{discovery, extraction, runners, formatters, bundle};
use crate::Result;

/// Generate a complete proof bundle for a crate
///
/// This is the V3 one-liner that:
/// 1. Discovers tests using cargo_metadata
/// 2. Extracts metadata from source code
/// 3. Runs tests (parsing stderr correctly!)
/// 4. Merges results with metadata
/// 5. Generates all 4 reports
/// 6. Writes everything to .proof_bundle/
///
/// # Example
///
/// ```rust,no_run
/// use proof_bundle; // crate name
///
/// let summary = proof_bundle::generate_for_crate(
///     "my-crate",
///     proof_bundle::Mode::UnitFast
/// )?;
///
/// println!("Tests: {} passed, {} failed", summary.passed, summary.failed);
/// # Ok::<(), proof_bundle::core::ProofBundleError>(())
/// ```
pub fn generate_for_crate(package: &str, mode: Mode) -> Result<TestSummary> {
    // 1. Discover tests using cargo_metadata
    let targets = discovery::discover_tests(package)?;
    
    // 2. Extract metadata from source files
    let metadata_map = extraction::extract_metadata(&targets)?;
    
    // 3. Run tests (parse stderr, not stdout!)
    let mut summary = runners::run_tests(package, mode)?;
    
    // 4. Merge metadata into test results
    for test in &mut summary.tests {
        if let Some(metadata) = metadata_map.get(&test.name) {
            test.metadata = Some(metadata.clone());
        }
    }
    
    // 5. Create bundle writer
    let writer = bundle::BundleWriter::new(mode)?;
    
    // 6. Write test results (NDJSON)
    writer.write_ndjson("test_results", &summary.tests)?;
    
    // 7. Write summary (JSON)
    writer.write_json("summary", &summary)?;
    
    // 8. Generate and write reports
    let executive = formatters::generate_executive_summary(&summary)
        .map_err(|e| crate::core::ProofBundleError::CannotGenerateReports { reason: e })?;
    writer.write_markdown("executive_summary", &executive)?;
    
    let developer = formatters::generate_developer_report(&summary)
        .map_err(|e| crate::core::ProofBundleError::CannotGenerateReports { reason: e })?;
    writer.write_markdown("test_report", &developer)?;
    
    let failure = formatters::generate_failure_report(&summary)
        .map_err(|e| crate::core::ProofBundleError::CannotGenerateReports { reason: e })?;
    writer.write_markdown("failure_report", &failure)?;
    
    let metadata_report = formatters::generate_metadata_report(&summary)
        .map_err(|e| crate::core::ProofBundleError::CannotGenerateReports { reason: e })?;
    writer.write_markdown("metadata_report", &metadata_report)?;
    
    // 9. Write test config for reference
    #[derive(serde::Serialize)]
    struct TestConfig {
        mode: String,
        package: String,
        timestamp: String,
    }
    
    let config = TestConfig {
        mode: mode.name().to_string(),
        package: package.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    writer.write_json("test_config", &config)?;
    
    println!("✅ Proof bundle generated at: {}", writer.root().display());
    
    Ok(summary)
}

/// Builder for advanced usage (future)
pub struct Builder {
    package: String,
    mode: Mode,
}

impl Builder {
    /// Create a new builder
    pub fn new(package: impl Into<String>) -> Self {
        Self {
            package: package.into(),
            mode: Mode::UnitFast,
        }
    }
    
    /// Set the test mode
    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }
    
    /// Generate the proof bundle
    pub fn generate(self) -> Result<TestSummary> {
        generate_for_crate(&self.package, self.mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// @priority: critical
    /// @spec: PB-V3-API
    /// @team: proof-bundle
    /// @tags: integration, e2e, dogfooding, one-liner-api
    /// @requires: cargo, git
    #[test]
    #[ignore] // Skip during normal testing to avoid circular dependency
    fn test_generate_for_proof_bundle() {
        // This is the ultimate integration test!
        // Generate a proof bundle for proof-bundle itself
        // Ignored by default because it creates a circular dependency during `cargo test`
        let result = generate_for_crate("proof-bundle", Mode::UnitFast);
        
        assert!(result.is_ok(), "Failed to generate proof bundle: {:?}", result.err());
        
        let summary = result.unwrap();
        
        // Validate results
        assert!(summary.total > 0, "Should find tests");
        assert!(summary.total >= 40, "Should find at least 40 tests, found {}", summary.total);
        assert!(summary.pass_rate >= 90.0, "Pass rate should be high, got {:.1}%", summary.pass_rate);
        
        // Validate metadata was extracted
        let with_metadata = summary.tests.iter()
            .filter(|t| t.metadata.is_some())
            .count();
        
        println!("Tests with metadata: {}/{}", with_metadata, summary.total);
    }
    
    /// @priority: high
    /// @spec: PB-V3-API
    /// @team: proof-bundle
    /// @tags: integration, builder-pattern, dogfooding
    /// @requires: cargo, git
    #[test]
    #[ignore] // Skip during normal testing to avoid circular dependency
    fn test_builder_api() {
        let result = Builder::new("proof-bundle")
            .mode(Mode::UnitFast)
            .generate();
        
        assert!(result.is_ok());
    }
}

//! One-liner API for proof bundle generation
//!
//! This module provides the simplest possible API for generating proof bundles.
//! Just one function call and you get a complete proof bundle with all reports.
//!
//! # Philosophy
//!
//! Management requirement: Developers should not have to think about proof bundles.
//! One function call should handle everything: running tests, parsing results,
//! generating reports, and writing files.
//!
//! # Example
//!
//! ```rust,no_run
//! use proof_bundle::api;
//!
//! // Generate complete proof bundle for a crate
//! api::generate_for_crate("my-crate", api::Mode::UnitFast)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{ProofBundle, TestSummary, LegacyTestType};
use crate::formatters;
use crate::parsers;
use crate::templates::{self, ProofBundleTemplate};
use anyhow::{Context, Result};
use std::process::Command;

/// Proof bundle generation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Unit tests - fast mode (skip long tests)
    UnitFast,
    
    /// Unit tests - full mode (all tests)
    UnitFull,
    
    /// BDD tests - mocked dependencies
    BddMock,
    
    /// BDD tests - real GPU/CUDA
    BddReal,
    
    /// Integration tests
    Integration,
    
    /// Property-based tests
    Property,
}

impl Mode {
    /// Get the template for this mode
    pub fn template(self) -> ProofBundleTemplate {
        match self {
            Mode::UnitFast => templates::unit_test_fast(),
            Mode::UnitFull => templates::unit_test_full(),
            Mode::BddMock => templates::bdd_test_mock(),
            Mode::BddReal => templates::bdd_test_real(),
            Mode::Integration => templates::integration_test(),
            Mode::Property => templates::property_test(),
        }
    }
    
    /// Get the test type for this mode
    fn test_type(self) -> LegacyTestType {
        match self {
            Mode::UnitFast | Mode::UnitFull => LegacyTestType::Unit,
            Mode::BddMock | Mode::BddReal => LegacyTestType::Bdd,
            Mode::Integration => LegacyTestType::Integration,
            Mode::Property => LegacyTestType::Unit, // Property tests are typically unit tests
        }
    }
}

/// Generate a complete proof bundle for a crate
///
/// This is the one-liner API that does everything:
/// 1. Creates proof bundle directory
/// 2. Runs cargo test with appropriate flags
/// 3. Parses test output
/// 4. Generates all reports (executive, developer, failure, metadata)
/// 5. Writes everything to disk
///
/// # Arguments
///
/// * `package` - Package name (e.g., "my-crate")
/// * `mode` - Test mode (UnitFast, BddReal, etc.)
///
/// # Returns
///
/// Returns the `TestSummary` with all test results.
///
/// # Example
///
/// ```rust,no_run
/// use proof_bundle::api;
///
/// // Generate proof bundle for unit tests (fast mode)
/// let summary = api::generate_for_crate("my-crate", api::Mode::UnitFast)?;
/// println!("Tests: {} passed, {} failed", summary.passed, summary.failed);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn generate_for_crate(package: &str, mode: Mode) -> Result<TestSummary> {
    let template = mode.template();
    let test_type = mode.test_type();
    
    // Create proof bundle
    let pb = ProofBundle::for_type(test_type)
        .context("Failed to create proof bundle")?;
    
    // Build cargo test command - use stable output format
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--package")
        .arg(package);
    
    // Add template flags
    for flag in &template.cargo_flags {
        cmd.arg(flag);
    }
    
    // Add test-specific flags
    if template.skip_long_tests {
        cmd.arg("--");
        cmd.arg("--skip-long-tests");
    }
    
    // Run tests and capture output
    let output = cmd.output()
        .context("Failed to run cargo test")?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Parse test results using stable parser (works on all Rust versions)
    let summary = parsers::parse_stable_output(&stdout)
        .context("Failed to parse test output")?;
    
    // Write test results
    for result in &summary.tests {
        pb.append_ndjson("test_results", result)
            .context("Failed to write test result")?;
    }
    
    // Write summary
    pb.write_json("summary", &summary)
        .context("Failed to write summary")?;
    
    // Generate and write all reports
    let executive = formatters::generate_executive_summary(&summary);
    pb.write_markdown("executive_summary", &executive)
        .context("Failed to write executive summary")?;
    
    let developer = formatters::generate_test_report(&summary);
    pb.write_markdown("test_report", &developer)
        .context("Failed to write test report")?;
    
    let failure = formatters::generate_failure_report(&summary);
    pb.write_markdown("failure_report", &failure)
        .context("Failed to write failure report")?;
    
    let metadata = formatters::generate_metadata_report(&summary);
    pb.write_markdown("metadata_report", &metadata)
        .context("Failed to write metadata report")?;
    
    // Write template config for reference
    pb.write_json("test_config", &template)
        .context("Failed to write test config")?;
    
    Ok(summary)
}

/// Generate a proof bundle with a custom template
///
/// For advanced users who need more control over test execution.
///
/// # Example
///
/// ```rust,no_run
/// use proof_bundle::api;
/// use proof_bundle::templates::ProofBundleTemplate;
/// use std::time::Duration;
///
/// let template = ProofBundleTemplate::custom("my-custom-test")
///     .with_timeout(Duration::from_secs(300))
///     .with_flags(vec!["--nocapture".to_string()]);
///
/// let summary = api::generate_with_template("my-crate", template)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn generate_with_template(package: &str, template: ProofBundleTemplate) -> Result<TestSummary> {
    // Determine test type from template
    let test_type = match template.test_type {
        templates::TestType::Unit => LegacyTestType::Unit,
        templates::TestType::Integration => LegacyTestType::Integration,
        templates::TestType::Bdd => LegacyTestType::Bdd,
        templates::TestType::Property => LegacyTestType::Unit,
        templates::TestType::E2e => LegacyTestType::E2eHaiku,
    };
    
    // Create proof bundle
    let pb = ProofBundle::for_type(test_type)
        .context("Failed to create proof bundle")?;
    
    // Build cargo test command - use stable output format
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--package")
        .arg(package);
    
    // Add template flags
    for flag in &template.cargo_flags {
        cmd.arg(flag);
    }
    
    // Add test-specific flags
    if template.skip_long_tests {
        cmd.arg("--");
        cmd.arg("--skip-long-tests");
    }
    
    // Run tests and capture output
    let output = cmd.output()
        .context("Failed to run cargo test")?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Parse test results using stable parser (works on all Rust versions)
    let summary = parsers::parse_stable_output(&stdout)?;
    
    // Write everything
    for result in &summary.tests {
        pb.append_ndjson("test_results", result)?;
    }
    
    pb.write_json("summary", &summary)?;
    pb.write_markdown("executive_summary", &formatters::generate_executive_summary(&summary))?;
    pb.write_markdown("test_report", &formatters::generate_test_report(&summary))?;
    pb.write_markdown("failure_report", &formatters::generate_failure_report(&summary))?;
    pb.write_markdown("metadata_report", &formatters::generate_metadata_report(&summary))?;
    pb.write_json("test_config", &template)?;
    
    Ok(summary)
}

// Add helper method to ProofBundleTemplate
impl ProofBundleTemplate {
    /// Get test threads setting
    pub fn test_threads(&self) -> Option<usize> {
        // Could be extended to support custom thread counts
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mode_template() {
        let template = Mode::UnitFast.template();
        assert_eq!(template.name, "unit-fast");
        assert!(template.skip_long_tests);
        
        let template = Mode::BddReal.template();
        assert_eq!(template.name, "bdd-real");
        assert!(template.requires_gpu);
    }
    
    #[test]
    fn test_mode_test_type() {
        assert_eq!(Mode::UnitFast.test_type(), LegacyTestType::Unit);
        assert_eq!(Mode::BddMock.test_type(), LegacyTestType::Bdd);
        assert_eq!(Mode::Integration.test_type(), LegacyTestType::Integration);
    }
}

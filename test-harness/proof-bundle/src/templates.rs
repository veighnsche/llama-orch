//! Test templates for standard proof bundle configurations
//!
//! This module provides pre-configured templates for different test types,
//! ensuring consistency across the repository.
//!
//! # Philosophy
//!
//! Management requirement: Developers should not have to think about proof bundle
//! configuration. Templates provide sensible defaults for common test scenarios.
//!
//! # Example
//!
//! ```rust
//! use proof_bundle::templates;
//!
//! // Unit test template (fast mode)
//! let template = templates::unit_test_fast();
//! assert_eq!(template.name, "unit-fast");
//! assert!(template.skip_long_tests);
//!
//! // BDD test template (mock mode)
//! let template = templates::bdd_test_mock();
//! assert_eq!(template.name, "bdd-mock");
//! assert!(template.mock_external_services);
//! ```

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Proof bundle template configuration
///
/// Defines standard settings for different test scenarios.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProofBundleTemplate {
    /// Template name (e.g., "unit-fast", "bdd-real")
    pub name: String,
    
    /// Human-readable description
    pub description: String,
    
    /// Test type category
    pub test_type: TestType,
    
    /// Skip long-running tests
    pub skip_long_tests: bool,
    
    /// Mock external services (databases, APIs, etc.)
    pub mock_external_services: bool,
    
    /// Require GPU/CUDA
    pub requires_gpu: bool,
    
    /// Expected test timeout
    pub timeout: Duration,
    
    /// Capture stdout
    pub capture_stdout: bool,
    
    /// Capture stderr
    pub capture_stderr: bool,
    
    /// Generate all reports
    pub generate_all_reports: bool,
    
    /// Additional cargo test flags
    pub cargo_flags: Vec<String>,
}

/// Test type category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestType {
    /// Unit tests
    Unit,
    
    /// Integration tests
    Integration,
    
    /// BDD/Cucumber tests
    Bdd,
    
    /// Property-based tests
    Property,
    
    /// End-to-end tests
    E2e,
}

impl ProofBundleTemplate {
    /// Create a custom template
    pub fn custom(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: "Custom template".to_string(),
            test_type: TestType::Unit,
            skip_long_tests: false,
            mock_external_services: false,
            requires_gpu: false,
            timeout: Duration::from_secs(300),
            capture_stdout: true,
            capture_stderr: true,
            generate_all_reports: true,
            cargo_flags: vec![],
        }
    }
    
    /// Add cargo test flags
    pub fn with_flags(mut self, flags: Vec<String>) -> Self {
        self.cargo_flags = flags;
        self
    }
    
    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// Unit test template - Fast mode
///
/// Runs unit tests with long tests skipped.
/// Ideal for quick feedback during development.
///
/// # Configuration
///
/// - Skips long tests
/// - Mocks external services
/// - No GPU required
/// - 60 second timeout
///
/// # Example
///
/// ```rust
/// use proof_bundle::templates;
///
/// let template = templates::unit_test_fast();
/// assert!(template.skip_long_tests);
/// assert!(template.mock_external_services);
/// ```
pub fn unit_test_fast() -> ProofBundleTemplate {
    ProofBundleTemplate {
        name: "unit-fast".to_string(),
        description: "Unit tests with long tests skipped (fast feedback)".to_string(),
        test_type: TestType::Unit,
        skip_long_tests: true,
        mock_external_services: true,
        requires_gpu: false,
        timeout: Duration::from_secs(60),
        capture_stdout: true,
        capture_stderr: true,
        generate_all_reports: true,
        cargo_flags: vec!["--lib".to_string()],
    }
}

/// Unit test template - Full mode
///
/// Runs all unit tests including long-running ones.
/// Ideal for CI/CD and pre-merge validation.
///
/// # Configuration
///
/// - Includes long tests
/// - Mocks external services
/// - No GPU required
/// - 300 second timeout
///
/// # Example
///
/// ```rust
/// use proof_bundle::templates;
///
/// let template = templates::unit_test_full();
/// assert!(!template.skip_long_tests);
/// ```
pub fn unit_test_full() -> ProofBundleTemplate {
    ProofBundleTemplate {
        name: "unit-full".to_string(),
        description: "All unit tests including long-running tests".to_string(),
        test_type: TestType::Unit,
        skip_long_tests: false,
        mock_external_services: true,
        requires_gpu: false,
        timeout: Duration::from_secs(300),
        capture_stdout: true,
        capture_stderr: true,
        generate_all_reports: true,
        cargo_flags: vec!["--lib".to_string()],
    }
}

/// BDD test template - Mock mode
///
/// Runs BDD tests with mocked external dependencies.
/// Ideal for testing business logic without infrastructure.
///
/// # Configuration
///
/// - Skips long tests
/// - Mocks external services
/// - No GPU required
/// - 120 second timeout
///
/// # Example
///
/// ```rust
/// use proof_bundle::templates;
///
/// let template = templates::bdd_test_mock();
/// assert!(template.mock_external_services);
/// ```
pub fn bdd_test_mock() -> ProofBundleTemplate {
    ProofBundleTemplate {
        name: "bdd-mock".to_string(),
        description: "BDD tests with mocked external services".to_string(),
        test_type: TestType::Bdd,
        skip_long_tests: true,
        mock_external_services: true,
        requires_gpu: false,
        timeout: Duration::from_secs(120),
        capture_stdout: true,
        capture_stderr: true,
        generate_all_reports: true,
        cargo_flags: vec!["--test".to_string()],
    }
}

/// BDD test template - Real mode
///
/// Runs BDD tests with real GPU/CUDA and external services.
/// Ideal for end-to-end validation before deployment.
///
/// # Configuration
///
/// - Includes long tests
/// - Real external services
/// - GPU required
/// - 600 second timeout
///
/// # Example
///
/// ```rust
/// use proof_bundle::templates;
///
/// let template = templates::bdd_test_real();
/// assert!(!template.mock_external_services);
/// assert!(template.requires_gpu);
/// ```
pub fn bdd_test_real() -> ProofBundleTemplate {
    ProofBundleTemplate {
        name: "bdd-real".to_string(),
        description: "BDD tests with real GPU/CUDA and external services".to_string(),
        test_type: TestType::Bdd,
        skip_long_tests: false,
        mock_external_services: false,
        requires_gpu: true,
        timeout: Duration::from_secs(600),
        capture_stdout: true,
        capture_stderr: true,
        generate_all_reports: true,
        cargo_flags: vec!["--test".to_string()],
    }
}

/// Integration test template
///
/// Runs integration tests with real services but mocked GPU.
/// Ideal for testing service interactions.
///
/// # Configuration
///
/// - Includes long tests
/// - Real external services
/// - No GPU required
/// - 300 second timeout
///
/// # Example
///
/// ```rust
/// use proof_bundle::templates;
///
/// let template = templates::integration_test();
/// assert!(!template.mock_external_services);
/// assert!(!template.requires_gpu);
/// ```
pub fn integration_test() -> ProofBundleTemplate {
    ProofBundleTemplate {
        name: "integration".to_string(),
        description: "Integration tests with real services".to_string(),
        test_type: TestType::Integration,
        skip_long_tests: false,
        mock_external_services: false,
        requires_gpu: false,
        timeout: Duration::from_secs(300),
        capture_stdout: true,
        capture_stderr: true,
        generate_all_reports: true,
        cargo_flags: vec!["--test".to_string()],
    }
}

/// Property-based test template
///
/// Runs property tests with many iterations.
/// Ideal for finding edge cases.
///
/// # Configuration
///
/// - Includes long tests
/// - Mocks external services
/// - No GPU required
/// - 600 second timeout
///
/// # Example
///
/// ```rust
/// use proof_bundle::templates;
///
/// let template = templates::property_test();
/// assert_eq!(template.test_type, templates::TestType::Property);
/// ```
pub fn property_test() -> ProofBundleTemplate {
    ProofBundleTemplate {
        name: "property".to_string(),
        description: "Property-based tests with many iterations".to_string(),
        test_type: TestType::Property,
        skip_long_tests: false,
        mock_external_services: true,
        requires_gpu: false,
        timeout: Duration::from_secs(600),
        capture_stdout: true,
        capture_stderr: true,
        generate_all_reports: true,
        cargo_flags: vec!["--test".to_string()],
    }
}

/// Get all standard templates
///
/// Returns a vector of all predefined templates.
///
/// # Example
///
/// ```rust
/// use proof_bundle::templates;
///
/// let all = templates::all_templates();
/// assert_eq!(all.len(), 6);
/// ```
pub fn all_templates() -> Vec<ProofBundleTemplate> {
    vec![
        unit_test_fast(),
        unit_test_full(),
        bdd_test_mock(),
        bdd_test_real(),
        integration_test(),
        property_test(),
    ]
}

/// Find template by name
///
/// # Example
///
/// ```rust
/// use proof_bundle::templates;
///
/// let template = templates::find_by_name("unit-fast").unwrap();
/// assert_eq!(template.name, "unit-fast");
///
/// assert!(templates::find_by_name("nonexistent").is_none());
/// ```
pub fn find_by_name(name: &str) -> Option<ProofBundleTemplate> {
    all_templates().into_iter().find(|t| t.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unit_test_fast() {
        let template = unit_test_fast();
        assert_eq!(template.name, "unit-fast");
        assert!(template.skip_long_tests);
        assert!(template.mock_external_services);
        assert!(!template.requires_gpu);
        assert_eq!(template.timeout, Duration::from_secs(60));
    }
    
    #[test]
    fn test_unit_test_full() {
        let template = unit_test_full();
        assert_eq!(template.name, "unit-full");
        assert!(!template.skip_long_tests);
        assert!(template.mock_external_services);
    }
    
    #[test]
    fn test_bdd_test_mock() {
        let template = bdd_test_mock();
        assert_eq!(template.name, "bdd-mock");
        assert!(template.mock_external_services);
        assert!(!template.requires_gpu);
    }
    
    #[test]
    fn test_bdd_test_real() {
        let template = bdd_test_real();
        assert_eq!(template.name, "bdd-real");
        assert!(!template.mock_external_services);
        assert!(template.requires_gpu);
        assert_eq!(template.timeout, Duration::from_secs(600));
    }
    
    #[test]
    fn test_integration_test() {
        let template = integration_test();
        assert_eq!(template.name, "integration");
        assert!(!template.mock_external_services);
        assert!(!template.requires_gpu);
    }
    
    #[test]
    fn test_property_test() {
        let template = property_test();
        assert_eq!(template.name, "property");
        assert_eq!(template.test_type, TestType::Property);
    }
    
    #[test]
    fn test_all_templates() {
        let templates = all_templates();
        assert_eq!(templates.len(), 6);
        
        let names: Vec<_> = templates.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"unit-fast"));
        assert!(names.contains(&"unit-full"));
        assert!(names.contains(&"bdd-mock"));
        assert!(names.contains(&"bdd-real"));
        assert!(names.contains(&"integration"));
        assert!(names.contains(&"property"));
    }
    
    #[test]
    fn test_find_by_name() {
        let template = find_by_name("unit-fast").unwrap();
        assert_eq!(template.name, "unit-fast");
        
        let template = find_by_name("bdd-real").unwrap();
        assert_eq!(template.name, "bdd-real");
        
        assert!(find_by_name("nonexistent").is_none());
    }
    
    #[test]
    fn test_custom_template() {
        let template = ProofBundleTemplate::custom("my-custom")
            .with_description("My custom test")
            .with_timeout(Duration::from_secs(120))
            .with_flags(vec!["--nocapture".to_string()]);
        
        assert_eq!(template.name, "my-custom");
        assert_eq!(template.description, "My custom test");
        assert_eq!(template.timeout, Duration::from_secs(120));
        assert_eq!(template.cargo_flags, vec!["--nocapture"]);
    }
}

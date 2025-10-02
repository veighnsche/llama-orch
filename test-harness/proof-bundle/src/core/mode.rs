//! Test execution modes

use serde::{Deserialize, Serialize};

/// Test execution mode
///
/// Determines which tests to run and how to run them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    /// Unit tests - fast mode (skip long-running tests)
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
    /// Get the cargo test flags for this mode
    pub fn cargo_flags(&self) -> Vec<&'static str> {
        match self {
            Mode::UnitFast => vec!["--lib"],
            Mode::UnitFull => vec!["--lib"],
            Mode::BddMock => vec!["--tests"],
            Mode::BddReal => vec!["--tests"],
            Mode::Integration => vec!["--tests"],
            Mode::Property => vec!["--lib"],
        }
    }
    
    /// Check if this mode should skip long tests
    pub fn skip_long_tests(&self) -> bool {
        matches!(self, Mode::UnitFast)
    }
    
    /// Check if this mode requires GPU
    pub fn requires_gpu(&self) -> bool {
        matches!(self, Mode::BddReal)
    }
    
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Mode::UnitFast => "unit-fast",
            Mode::UnitFull => "unit-full",
            Mode::BddMock => "bdd-mock",
            Mode::BddReal => "bdd-real",
            Mode::Integration => "integration",
            Mode::Property => "property",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// @priority: critical
    /// @spec: PB-V3-CORE
    /// @team: proof-bundle
    /// @tags: unit, core, mode
    #[test]
    fn test_mode_names() {
        assert_eq!(Mode::UnitFast.name(), "unit-fast");
        assert_eq!(Mode::BddReal.name(), "bdd-real");
    }
    
    /// @priority: high
    /// @spec: PB-V3-CORE
    /// @team: proof-bundle
    /// @tags: unit, mode, fast-tests
    #[test]
    fn test_skip_long_tests() {
        assert!(Mode::UnitFast.skip_long_tests());
        assert!(!Mode::UnitFull.skip_long_tests());
    }
    
    /// @priority: high
    /// @spec: PB-V3-CORE
    /// @team: proof-bundle
    /// @tags: unit, mode, gpu-detection
    #[test]
    fn test_requires_gpu() {
        assert!(Mode::BddReal.requires_gpu());
        assert!(!Mode::BddMock.requires_gpu());
        assert!(!Mode::UnitFast.requires_gpu());
    }
}

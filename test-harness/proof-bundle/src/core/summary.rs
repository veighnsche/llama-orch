//! Test summary aggregation

use serde::{Deserialize, Serialize};
use super::TestResult;

/// Summary of all test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total number of tests
    pub total: usize,
    
    /// Number of passed tests
    pub passed: usize,
    
    /// Number of failed tests
    pub failed: usize,
    
    /// Number of ignored tests
    pub ignored: usize,
    
    /// Total duration in seconds
    pub duration_secs: f64,
    
    /// Pass rate (0.0 - 100.0)
    pub pass_rate: f64,
    
    /// All test results
    pub tests: Vec<TestResult>,
}

impl TestSummary {
    /// Calculate pass rate from counts
    pub fn calculate_pass_rate(passed: usize, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        (passed as f64 / total as f64) * 100.0
    }
    
    /// Create a new test summary
    pub fn new(tests: Vec<TestResult>) -> Self {
        let total = tests.len();
        let passed = tests.iter().filter(|t| t.status.is_passing()).count();
        let failed = tests.iter().filter(|t| t.status.is_failing()).count();
        let ignored = total - passed - failed;
        let duration_secs = tests.iter().map(|t| t.duration_secs).sum();
        let pass_rate = Self::calculate_pass_rate(passed, total);
        
        Self {
            total,
            passed,
            failed,
            ignored,
            duration_secs,
            pass_rate,
            tests,
        }
    }
    
    /// Validate this summary
    pub fn validate(&self) -> Result<(), String> {
        if self.total == 0 {
            return Err("TestSummary has 0 tests".to_string());
        }
        
        if self.total != self.tests.len() {
            return Err(format!(
                "TestSummary total ({}) doesn't match tests.len() ({})",
                self.total,
                self.tests.len()
            ));
        }
        
        let actual_passed = self.tests.iter().filter(|t| t.status.is_passing()).count();
        if self.passed != actual_passed {
            return Err(format!(
                "TestSummary passed count ({}) doesn't match actual ({})",
                self.passed, actual_passed
            ));
        }
        
        Ok(())
    }
}

impl Default for TestSummary {
    fn default() -> Self {
        Self {
            total: 0,
            passed: 0,
            failed: 0,
            ignored: 0,
            duration_secs: 0.0,
            pass_rate: 0.0,
            tests: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{TestStatus, TestResult};
    
    /// @priority: critical
    /// @spec: PB-V3-CORE
    /// @team: proof-bundle
    /// @tags: unit, core, math, pass-rate
    #[test]
    fn test_calculate_pass_rate() {
        assert_eq!(TestSummary::calculate_pass_rate(10, 10), 100.0);
        assert_eq!(TestSummary::calculate_pass_rate(5, 10), 50.0);
        assert_eq!(TestSummary::calculate_pass_rate(0, 10), 0.0);
        assert_eq!(TestSummary::calculate_pass_rate(0, 0), 0.0);
    }
    
    /// @priority: critical
    /// @spec: PB-V3-CORE
    /// @team: proof-bundle
    /// @tags: unit, core, aggregation
    #[test]
    fn test_new_summary() {
        let tests = vec![
            TestResult::new("test1".to_string(), TestStatus::Passed),
            TestResult::new("test2".to_string(), TestStatus::Failed),
            TestResult::new("test3".to_string(), TestStatus::Passed),
        ];
        
        let summary = TestSummary::new(tests);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.pass_rate, (2.0 / 3.0) * 100.0);
    }
    
    /// @priority: critical
    /// @spec: PB-V3-VALIDATION
    /// @team: proof-bundle
    /// @tags: unit, validation, zero-tests-bug-fix
    /// @scenario: Validating test summary to detect empty test suites before report generation
    /// @threat: Empty test suite gives false confidence - reports show "success" when nothing was tested
    /// @failure_mode: Proof bundle generated for 0 tests looks identical to a passing test suite
    /// @edge_case: Must explicitly reject summaries with total=0 to prevent misleading audit trails
    #[test]
    fn test_validate() {
        let tests = vec![
            TestResult::new("test1".to_string(), TestStatus::Passed),
        ];
        
        let summary = TestSummary::new(tests);
        assert!(summary.validate().is_ok());
        
        // Empty summary should fail validation
        let empty = TestSummary::default();
        assert!(empty.validate().is_err());
    }
}

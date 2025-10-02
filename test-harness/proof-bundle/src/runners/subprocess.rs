//! Run tests as subprocess
//!
//! This module runs `cargo test` as a subprocess and parses the output.
//!
//! # CRITICAL BUG FIX
//!
//! cargo test writes test output to **STDERR**, not STDOUT!
//! STDOUT contains warnings and compilation messages.
//!
//! We parse STDERR to get actual test results.

use crate::core::{Mode, TestResult, TestStatus, TestSummary, ProofBundleError};
use crate::Result;
use std::process::Command;

/// Run cargo test as subprocess
///
/// # Bug Fix
///
/// Previous implementation parsed stdout (wrong!).
/// This correctly parses stderr where test output actually is.
pub fn run_tests(package: &str, mode: Mode) -> Result<TestSummary> {
    // Build cargo test command
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--package")
        .arg(package);
    
    // Add mode-specific flags
    for flag in mode.cargo_flags() {
        cmd.arg(flag);
    }
    
    // Add skip-long-tests flag if needed
    if mode.skip_long_tests() {
        cmd.arg("--");
        cmd.arg("--skip-long-tests");
    }
    
    // Run command and capture output
    let output = cmd.output()
        .map_err(|e| ProofBundleError::CargoTestFailed {
            exit_code: None,
            message: format!("Failed to execute cargo test: {}", e),
        })?;
    
    // BUG FIX: Parse STDERR, not STDOUT!
    let test_output = String::from_utf8_lossy(&output.stderr);
    
    // Parse test results
    let summary = parse_test_output(&test_output)?;
    
    // Validate we found tests
    if summary.total == 0 {
        return Err(ProofBundleError::NoTestsFound {
            package: package.to_string(),
            hint: format!(
                "No tests found in output. Check that package '{}' has tests. \
                 cargo test exit code: {:?}",
                package,
                output.status.code()
            ),
        });
    }
    
    Ok(summary)
}

/// Parse test output from cargo test
///
/// Parses the human-readable format:
/// ```text
/// running 3 tests
/// test test_foo ... ok
/// test test_bar ... FAILED
/// test test_baz ... ignored
///
/// test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured
/// ```
fn parse_test_output(output: &str) -> Result<TestSummary> {
    let mut tests = Vec::new();
    let mut total_passed = 0;
    let mut total_failed = 0;
    let mut total_ignored = 0;

    // Parse individual test lines
    for line in output.lines() {
        let line = line.trim();

        // Match lines like: "test test_name ... ok"
        if line.starts_with("test ") && (line.contains(" ... ok") || line.contains(" ... FAILED") || line.contains(" ... ignored")) {
            if let Some(result) = parse_test_line(line) {
                match result.status {
                    TestStatus::Passed => total_passed += 1,
                    TestStatus::Failed => total_failed += 1,
                    TestStatus::Ignored => total_ignored += 1,
                }
                tests.push(result);
            }
        }
    }

    // Try to extract summary from "test result:" line for validation
    for line in output.lines() {
        if line.contains("test result:") {
            // Extract numbers for validation
            if let Some(passed) = extract_number(line, "passed") {
                total_passed = passed;
            }
            if let Some(failed) = extract_number(line, "failed") {
                total_failed = failed;
            }
            if let Some(ignored) = extract_number(line, "ignored") {
                total_ignored = ignored;
            }
            break;
        }
    }

    // Attach failure details if present in output
    if total_failed > 0 {
        attach_failure_details(output, &mut tests);
    }

    let total = tests.len();
    let pass_rate = TestSummary::calculate_pass_rate(total_passed, total);

    Ok(TestSummary {
        total,
        passed: total_passed,
        failed: total_failed,
        ignored: total_ignored,
        duration_secs: 0.0, // Not available in text output
        pass_rate,
        tests,
    })
}

/// Parse failure detail sections and attach to failed tests as error_message
fn attach_failure_details(output: &str, tests: &mut [TestResult]) {
    use std::collections::HashMap;
    let mut by_name: HashMap<String, usize> = HashMap::new();
    for (i, t) in tests.iter().enumerate() {
        by_name.insert(t.name.clone(), i);
    }

    let lines: Vec<&str> = output.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim();
        // Look for blocks like: "---- module::test_name stdout ----"
        if line.starts_with("---- ") && line.ends_with(" stdout ----") {
            // Extract test name between markers
            let inner = &line[5..line.len() - " stdout ----".len()];
            let test_name = inner.trim();
            // Accumulate lines until next block or summary
            let mut buf = String::new();
            i += 1;
            while i < lines.len() {
                let l = lines[i];
                let t = l.trim();
                if t.starts_with("---- ") || t.starts_with("failures:") || t.starts_with("test result:") {
                    i -= 1; // step back one so outer loop reprocesses header
                    break;
                }
                buf.push_str(l);
                buf.push('\n');
                i += 1;
            }
            // Attach to matching test result if exists
            if let Some(&idx) = by_name.get(test_name) {
                if tests[idx].error_message.is_none() {
                    tests[idx].error_message = Some(buf.trim().to_string());
                }
            }
        }
        i += 1;
    }
}

/// Parse a single test line
///
/// Format: "test test_name ... ok"
fn parse_test_line(line: &str) -> Option<TestResult> {
    let parts: Vec<&str> = line.split(" ... ").collect();
    if parts.len() != 2 {
        return None;
    }
    
    let name = parts[0].strip_prefix("test ")?.trim().to_string();
    let result_str = parts[1].trim();
    
    let status = if result_str.starts_with("ok") {
        TestStatus::Passed
    } else if result_str.starts_with("FAILED") {
        TestStatus::Failed
    } else if result_str.starts_with("ignored") {
        TestStatus::Ignored
    } else {
        return None;
    };
    
    Some(TestResult::new(name, status))
}

/// Extract a number from a line like "1 passed; 2 failed"
fn extract_number(line: &str, keyword: &str) -> Option<usize> {
    let keyword_pos = line.find(keyword)?;
    let before = &line[..keyword_pos];
    let parts: Vec<&str> = before.split_whitespace().collect();
    parts.last()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// @priority: critical
    /// @spec: PB-V3-RUNNER
    /// @team: proof-bundle
    /// @tags: unit, runner, parser, stderr-bug-fix
    #[test]
    fn test_parse_test_line() {
        let line = "test my_module::test_foo ... ok";
        let result = parse_test_line(line).unwrap();
        assert_eq!(result.name, "my_module::test_foo");
        assert_eq!(result.status, TestStatus::Passed);
    }
    
    /// @priority: critical
    /// @spec: PB-V3-RUNNER
    /// @team: proof-bundle
    /// @tags: unit, runner, parser, failure-handling
    #[test]
    fn test_parse_test_line_failed() {
        let line = "test test_bar ... FAILED";
        let result = parse_test_line(line).unwrap();
        assert_eq!(result.name, "test_bar");
        assert_eq!(result.status, TestStatus::Failed);
    }
    
    /// @priority: critical
    /// @spec: PB-V3-RUNNER
    /// @team: proof-bundle
    /// @tags: unit, runner, parser, integration
    #[test]
    fn test_parse_test_output() {
        let output = r#"
running 3 tests
test test_foo ... ok
test test_bar ... FAILED
test test_baz ... ignored

failures:
    test_bar

test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out
"#;
        
        let summary = parse_test_output(output).unwrap();
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.ignored, 1);
    }
    
    /// @priority: high
    /// @spec: PB-V3-RUNNER
    /// @team: proof-bundle
    /// @tags: unit, runner, parser, summary-extraction
    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("1 passed; 2 failed", "passed"), Some(1));
        assert_eq!(extract_number("1 passed; 2 failed", "failed"), Some(2));
        assert_eq!(extract_number("10 passed; 0 failed", "passed"), Some(10));
    }
    
    /// @priority: critical
    /// @spec: PB-V3-RUNNER
    /// @team: proof-bundle
    /// @tags: integration, runner, dogfooding, e2e
    #[test]
    #[ignore] // Skip during normal testing to avoid circular dependency
    fn test_run_tests_on_proof_bundle() {
        // This actually runs cargo test on proof-bundle
        // Ignored by default because it creates a circular dependency during `cargo test`
        let result = run_tests("proof-bundle", Mode::UnitFast);
        
        // Should succeed
        assert!(result.is_ok(), "Failed to run tests: {:?}", result.err());
        
        let summary = result.unwrap();
        
        // Should find tests (we have 43+ tests)
        assert!(summary.total > 0, "Should find tests");
        assert!(summary.total >= 40, "Should find at least 40 tests, found {}", summary.total);
        
        // Should have high pass rate
        assert!(summary.pass_rate >= 90.0, "Pass rate should be high, got {:.1}%", summary.pass_rate);
    }
}


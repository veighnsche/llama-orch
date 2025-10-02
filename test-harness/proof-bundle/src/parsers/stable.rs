//! Stable output parser for standard `cargo test` output

use crate::{TestResult, TestStatus, TestSummary};
use anyhow::Result;

/// Parse standard cargo test output
///
/// Parses the human-readable output from `cargo test` and extracts
/// test results. This is a fallback for when JSON output is not available.
///
/// # Format
///
/// Standard cargo test output looks like:
/// ```text
/// running 3 tests
/// test test_foo ... ok
/// test test_bar ... FAILED
/// test test_baz ... ignored
///
/// failures:
///     test_bar
///
/// test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out
/// ```
///
/// # Example
///
/// ```rust
/// use proof_bundle::parsers;
///
/// let output = r#"running 2 tests
/// test test_foo ... ok
/// test test_bar ... FAILED
///
/// test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"#;
///
/// let summary = parsers::parse_stable_output(output).unwrap();
/// assert_eq!(summary.total, 2);
/// assert_eq!(summary.passed, 1);
/// assert_eq!(summary.failed, 1);
/// ```
pub fn parse_stable_output(stdout: &str) -> Result<TestSummary> {
    let mut test_results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;
    let mut ignored = 0;
    
    // Parse individual test lines
    for line in stdout.lines() {
        let line = line.trim();
        
        // Match lines like: "test test_name ... ok"
        if line.starts_with("test ") && (line.contains(" ... ok") || line.contains(" ... FAILED") || line.contains(" ... ignored")) {
            let parts: Vec<&str> = line.split(" ... ").collect();
            if parts.len() == 2 {
                let name = parts[0].strip_prefix("test ").unwrap_or(parts[0]).to_string();
                let result = parts[1];
                
                let status = if result.starts_with("ok") {
                    passed += 1;
                    TestStatus::Passed
                } else if result.starts_with("FAILED") {
                    failed += 1;
                    TestStatus::Failed
                } else if result.starts_with("ignored") {
                    ignored += 1;
                    TestStatus::Ignored
                } else {
                    continue;
                };
                
                test_results.push(TestResult {
                    name,
                    status,
                    duration_secs: 0.0, // Not available in stable output
                    stdout: None,
                    stderr: None,
                    error_message: None,
                    metadata: None,
                });
            }
        }
    }
    
    // Try to extract summary from "test result:" line
    // Format: "test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out"
    for line in stdout.lines() {
        if line.contains("test result:") {
            // Extract numbers from the summary line
            if let Some(passed_str) = extract_number(line, "passed") {
                passed = passed_str;
            }
            if let Some(failed_str) = extract_number(line, "failed") {
                failed = failed_str;
            }
            if let Some(ignored_str) = extract_number(line, "ignored") {
                ignored = ignored_str;
            }
            break;
        }
    }
    
    let total = test_results.len();
    let pass_rate = TestSummary::calculate_pass_rate(passed, total);
    
    Ok(TestSummary {
        total,
        passed,
        failed,
        ignored,
        duration_secs: 0.0, // Not available in stable output
        pass_rate,
        tests: test_results,
    })
}

/// Extract a number from a line like "1 passed; 2 failed"
fn extract_number(line: &str, keyword: &str) -> Option<usize> {
    // Find the keyword
    let keyword_pos = line.find(keyword)?;
    
    // Look backwards for the number
    let before = &line[..keyword_pos];
    let parts: Vec<&str> = before.split_whitespace().collect();
    
    // The number should be the last token before the keyword
    if let Some(last) = parts.last() {
        last.parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_stable_output_success() {
        let output = r#"running 2 tests
test test_foo ... ok
test test_bar ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"#;
        
        let summary = parse_stable_output(output).unwrap();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 0);
    }
    
    #[test]
    fn test_parse_stable_output_with_failures() {
        let output = r#"running 3 tests
test test_pass ... ok
test test_fail ... FAILED
test test_ignored ... ignored

failures:
    test_fail

test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out"#;
        
        let summary = parse_stable_output(output).unwrap();
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.ignored, 1);
    }
    
    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("1 passed; 2 failed", "passed"), Some(1));
        assert_eq!(extract_number("1 passed; 2 failed", "failed"), Some(2));
        assert_eq!(extract_number("10 passed; 0 failed", "passed"), Some(10));
        assert_eq!(extract_number("no numbers here", "passed"), None);
    }
}

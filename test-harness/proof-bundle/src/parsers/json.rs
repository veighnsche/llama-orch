//! JSON output parser for `cargo test --format json`

use crate::{TestResult, TestStatus, TestSummary};
use anyhow::{Context, Result};
use serde_json::Value;

/// Parse JSON-formatted cargo test output
///
/// Parses the output from `cargo test -- --format json` and extracts
/// test results into a structured summary.
///
/// # Format
///
/// Each line is a JSON object with:
/// - `type`: "test", "suite", etc.
/// - `event`: "ok", "failed", "ignored", "timeout"
/// - `name`: Test name
/// - `exec_time`: Execution time in seconds
/// - `stdout`/`stderr`: Optional output
///
/// # Example
///
/// ```rust
/// use proof_bundle::parsers;
///
/// let output = r#"{"type":"test","event":"ok","name":"test_foo","exec_time":0.001}
/// {"type":"test","event":"failed","name":"test_bar","exec_time":0.5}"#;
///
/// let summary = parsers::parse_json_output(output).unwrap();
/// assert_eq!(summary.total, 2);
/// assert_eq!(summary.passed, 1);
/// assert_eq!(summary.failed, 1);
/// ```
pub fn parse_json_output(stdout: &str) -> Result<TestSummary> {
    let mut test_results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;
    let mut ignored = 0;
    let mut total_duration = 0.0;
    
    for line in stdout.lines() {
        if let Ok(json) = serde_json::from_str::<Value>(line) {
            if json["type"] == "test" {
                if let Some(event) = json["event"].as_str() {
                    let name = json["name"].as_str().unwrap_or("unknown").to_string();
                    let exec_time = json["exec_time"].as_f64().unwrap_or(0.0);
                    
                    let (status, should_count) = match event {
                        "ok" => {
                            passed += 1;
                            (TestStatus::Passed, true)
                        }
                        "failed" => {
                            failed += 1;
                            (TestStatus::Failed, true)
                        }
                        "ignored" => {
                            ignored += 1;
                            (TestStatus::Ignored, false)
                        }
                        "timeout" => {
                            failed += 1;
                            (TestStatus::Timeout, true)
                        }
                        _ => continue,
                    };
                    
                    if should_count {
                        total_duration += exec_time;
                    }
                    
                    // Extract stdout/stderr if available
                    let stdout_val = json["stdout"].as_str().map(|s| s.to_string());
                    let stderr_val = json["stderr"].as_str().map(|s| s.to_string());
                    
                    // Extract error message for failures
                    let error_message = if status == TestStatus::Failed {
                        stderr_val.clone().or_else(|| {
                            json["message"].as_str().map(|s| s.to_string())
                        })
                    } else {
                        None
                    };
                    
                    test_results.push(TestResult {
                        name,
                        status,
                        duration_secs: exec_time,
                        stdout: stdout_val,
                        stderr: stderr_val,
                        error_message,
                        metadata: None,
                    });
                }
            }
        }
    }
    
    let total = test_results.len();
    let pass_rate = TestSummary::calculate_pass_rate(passed, total);
    
    Ok(TestSummary {
        total,
        passed,
        failed,
        ignored,
        duration_secs: total_duration,
        pass_rate,
        tests: test_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_json_output_success() {
        let output = r#"{"type":"test","event":"ok","name":"test_foo","exec_time":0.001}
{"type":"test","event":"ok","name":"test_bar","exec_time":0.002}"#;
        
        let summary = parse_json_output(output).unwrap();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.duration_secs, 0.003);
    }
    
    #[test]
    fn test_parse_json_output_with_failures() {
        let output = r#"{"type":"test","event":"ok","name":"test_pass","exec_time":0.001}
{"type":"test","event":"failed","name":"test_fail","exec_time":0.5,"stderr":"assertion failed"}"#;
        
        let summary = parse_json_output(output).unwrap();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.pass_rate, 50.0);
        
        let failed_test = summary.tests.iter().find(|t| t.name == "test_fail").unwrap();
        assert_eq!(failed_test.status, TestStatus::Failed);
        assert!(failed_test.error_message.is_some());
    }
    
    #[test]
    fn test_parse_json_output_with_ignored() {
        let output = r#"{"type":"test","event":"ok","name":"test_pass","exec_time":0.001}
{"type":"test","event":"ignored","name":"test_ignored"}"#;
        
        let summary = parse_json_output(output).unwrap();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.ignored, 1);
    }
    
    #[test]
    fn test_parse_json_output_empty() {
        let output = "";
        let summary = parse_json_output(output).unwrap();
        assert_eq!(summary.total, 0);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.failed, 0);
    }
}

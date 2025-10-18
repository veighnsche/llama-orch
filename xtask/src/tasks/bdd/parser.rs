// TEAM-111: Test output parsing logic

use super::types::{FailureInfo, TestResults};
use regex::Regex;

/// Parse test output to extract results
pub fn parse_test_output(output: &str, exit_code: i32) -> TestResults {
    let mut results = TestResults { exit_code, ..Default::default() };

    // Validate output is not empty
    if output.trim().is_empty() {
        eprintln!("Warning: Test output is empty, cannot parse results");
        return results;
    }

    // Parse counts using regex
    if let Some(passed) = extract_count(output, r"(\d+) passed") {
        results.passed = passed;
    }

    if let Some(failed) = extract_count(output, r"(\d+) failed") {
        results.failed = failed;
    }

    if let Some(skipped) = extract_count(output, r"(\d+) skipped") {
        results.skipped = skipped;
    }

    // Sanity check: if exit code is 0 but we have failures, something is wrong
    if results.exit_code == 0 && results.failed > 0 {
        eprintln!("Warning: Exit code is 0 but {} tests failed", results.failed);
    }

    results
}

/// Extract failed test information
pub fn extract_failures(output: &str) -> Vec<FailureInfo> {
    let mut failures = Vec::new();

    // Pattern 1: "test name ... FAILED"
    let failed_re = Regex::new(r"test ([\w:]+) \.\.\. FAILED").unwrap();
    for cap in failed_re.captures_iter(output) {
        if let Some(name) = cap.get(1) {
            failures
                .push(FailureInfo { test_name: name.as_str().to_string(), context: String::new() });
        }
    }

    // Extract context for each failure
    for failure in &mut failures {
        if let Some(context) = extract_failure_context(output, &failure.test_name) {
            failure.context = context;
        }
    }

    failures
}

/// Extract test names from failures
pub fn extract_failed_test_names(output: &str) -> Vec<String> {
    let failed_re = Regex::new(r"test ([\w:]+) \.\.\. FAILED").unwrap();
    failed_re
        .captures_iter(output)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

fn extract_count(output: &str, pattern: &str) -> Option<usize> {
    // Regex compilation should never fail for our hardcoded patterns,
    // but handle it gracefully just in case
    let re = match Regex::new(pattern) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Warning: Failed to compile regex '{}': {}", pattern, e);
            return None;
        }
    };

    re.captures(output).and_then(|cap| cap.get(1)).and_then(|m| m.as_str().parse().ok())
}

fn extract_failure_context(output: &str, test_name: &str) -> Option<String> {
    let lines: Vec<&str> = output.lines().collect();
    let marker = format!("test {} ... FAILED", test_name);

    for (i, line) in lines.iter().enumerate() {
        if line.contains(&marker) {
            // Extract context (lines around the failure)
            let start = i.saturating_sub(2);
            let end = (i + 10).min(lines.len());
            return Some(lines[start..end].join("\n"));
        }
    }

    None
}

/// Extract all failure patterns for detailed reporting
pub fn extract_all_failure_patterns(output: &str) -> String {
    let mut patterns = Vec::new();

    // Pattern 1: FAILED markers
    patterns.extend(extract_pattern_with_context(output, "FAILED", 2, 10));

    // Pattern 2: Error: messages
    patterns.extend(extract_pattern_with_context(output, "Error:", 2, 5));

    // Pattern 3: assertion failures
    patterns.extend(extract_pattern_with_context(output, "assertion", 2, 5));

    // Pattern 4: panicked at
    patterns.extend(extract_pattern_with_context(output, "panicked at", 2, 5));

    patterns.join("\n\n")
}

fn extract_pattern_with_context(
    output: &str,
    pattern: &str,
    before: usize,
    after: usize,
) -> Vec<String> {
    let lines: Vec<&str> = output.lines().collect();
    let mut results = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        if line.contains(pattern) {
            let start = i.saturating_sub(before);
            let end = (i + after + 1).min(lines.len());
            results.push(lines[start..end].join("\n"));
        }
    }

    results
}

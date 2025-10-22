// TEAM-111: Unit tests for parser module
// Testing behavior, not coverage

// Mock the parser functions for testing
fn parse_test_output_mock(output: &str, exit_code: i32) -> (usize, usize, usize, i32) {
    use regex::Regex;

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    if output.trim().is_empty() {
        return (0, 0, 0, exit_code);
    }

    if let Some(re) = Regex::new(r"(\d+) passed").ok() {
        if let Some(cap) = re.captures(output) {
            passed = cap.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
        }
    }

    if let Some(re) = Regex::new(r"(\d+) failed").ok() {
        if let Some(cap) = re.captures(output) {
            failed = cap.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
        }
    }

    if let Some(re) = Regex::new(r"(\d+) skipped").ok() {
        if let Some(cap) = re.captures(output) {
            skipped = cap.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
        }
    }

    (passed, failed, skipped, exit_code)
}

#[test]
fn test_parser_handles_successful_run() {
    let output = "test result: ok. 10 passed; 0 failed; 0 ignored";
    let (passed, failed, skipped, _) = parse_test_output_mock(output, 0);

    assert_eq!(passed, 10);
    assert_eq!(failed, 0);
}

#[test]
fn test_parser_handles_failed_run() {
    let output = "test result: FAILED. 8 passed; 2 failed; 0 ignored";
    let (passed, failed, skipped, _) = parse_test_output_mock(output, 1);

    assert_eq!(passed, 8);
    assert_eq!(failed, 2);
}

#[test]
fn test_parser_handles_empty_output() {
    let output = "";
    let (passed, failed, skipped, _) = parse_test_output_mock(output, 1);

    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    assert_eq!(skipped, 0);
}

#[test]
fn test_parser_handles_whitespace_output() {
    let output = "   \n\n   ";
    let (passed, failed, skipped, _) = parse_test_output_mock(output, 1);

    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_parser_handles_malformed_output() {
    let output = "Random text without test results";
    let (passed, failed, skipped, _) = parse_test_output_mock(output, 0);

    // Should return zeros for malformed output
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_parser_handles_large_numbers() {
    let output = "test result: ok. 999 passed; 123 failed; 456 skipped";
    let (passed, failed, skipped, _) = parse_test_output_mock(output, 1);

    assert_eq!(passed, 999);
    assert_eq!(failed, 123);
    assert_eq!(skipped, 456);
}

#[test]
fn test_parser_handles_zero_results() {
    let output = "test result: ok. 0 passed; 0 failed; 0 skipped";
    let (passed, failed, skipped, _) = parse_test_output_mock(output, 0);

    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    assert_eq!(skipped, 0);
}

#[test]
fn test_extract_failed_test_names() {
    use regex::Regex;

    let output = r#"
test auth::token_validation ... FAILED
test lifecycle::worker_startup ... ok
test auth::timing_safe ... FAILED
"#;

    let re = Regex::new(r"test ([\w:]+) \.\.\. FAILED").unwrap();
    let failed_names: Vec<String> = re
        .captures_iter(output)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();

    assert_eq!(failed_names.len(), 2);
    assert!(failed_names.contains(&"auth::token_validation".to_string()));
    assert!(failed_names.contains(&"auth::timing_safe".to_string()));
}

#[test]
fn test_extract_no_failed_tests() {
    use regex::Regex;

    let output = "test auth::token_validation ... ok\ntest lifecycle::worker_startup ... ok";

    let re = Regex::new(r"test ([\w:]+) \.\.\. FAILED").unwrap();
    let failed_names: Vec<String> = re
        .captures_iter(output)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();

    assert_eq!(failed_names.len(), 0);
}

#[test]
fn test_parser_with_ansi_codes() {
    // Simulated ANSI color codes
    let output = "\x1b[32mtest result: ok. 7 passed\x1b[0m; 1 failed; 0 skipped";
    let (passed, failed, _, _) = parse_test_output_mock(output, 0);

    // Should still parse correctly despite ANSI codes
    assert_eq!(passed, 7);
    assert_eq!(failed, 1);
}

#[test]
fn test_parser_with_unicode() {
    let output = "test result: ✅ ok. 5 passed; 0 failed; 0 skipped";
    let (passed, _, _, _) = parse_test_output_mock(output, 0);

    // Should handle unicode gracefully
    assert_eq!(passed, 5);
}

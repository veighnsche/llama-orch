// TEAM-111: Parser behavior tests

#[cfg(test)]
mod tests {
    use super::super::parser::*;

    #[test]
    fn test_parse_successful_test_run() {
        let output = r#"
running 10 tests
test auth::token_validation ... ok
test auth::timing_safe ... ok
test lifecycle::worker_startup ... ok

test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
"#;

        let results = parse_test_output(output, 0);

        assert_eq!(results.passed, 10);
        assert_eq!(results.failed, 0);
        assert_eq!(results.skipped, 0);
        assert_eq!(results.exit_code, 0);
    }

    #[test]
    fn test_parse_failed_test_run() {
        let output = r#"
running 10 tests
test auth::token_validation ... ok
test auth::timing_safe ... FAILED
test lifecycle::worker_startup ... ok

failures:

---- auth::timing_safe stdout ----
thread 'auth::timing_safe' panicked at 'assertion failed'

test result: FAILED. 8 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out
"#;

        let results = parse_test_output(output, 1);

        assert_eq!(results.passed, 8);
        assert_eq!(results.failed, 2);
        assert_eq!(results.skipped, 0);
        assert_eq!(results.exit_code, 1);
    }

    #[test]
    fn test_parse_with_skipped_tests() {
        let output = r#"
test result: ok. 5 passed; 0 failed; 3 skipped; 0 measured; 0 filtered out
"#;

        let results = parse_test_output(output, 0);

        assert_eq!(results.passed, 5);
        assert_eq!(results.failed, 0);
        assert_eq!(results.skipped, 3);
    }

    #[test]
    fn test_parse_empty_output() {
        let output = "";

        let results = parse_test_output(output, 1);

        // Should return defaults when output is empty
        assert_eq!(results.passed, 0);
        assert_eq!(results.failed, 0);
        assert_eq!(results.skipped, 0);
        assert_eq!(results.exit_code, 1);
    }

    #[test]
    fn test_parse_whitespace_only_output() {
        let output = "   \n\n   \t  \n  ";

        let results = parse_test_output(output, 1);

        assert_eq!(results.passed, 0);
        assert_eq!(results.failed, 0);
        assert_eq!(results.skipped, 0);
    }

    #[test]
    fn test_parse_malformed_output() {
        let output = "Some random text without test results";

        let results = parse_test_output(output, 0);

        // Should handle gracefully
        assert_eq!(results.passed, 0);
        assert_eq!(results.failed, 0);
        assert_eq!(results.skipped, 0);
    }

    #[test]
    fn test_parse_large_numbers() {
        let output = "test result: ok. 999 passed; 123 failed; 456 skipped";

        let results = parse_test_output(output, 1);

        assert_eq!(results.passed, 999);
        assert_eq!(results.failed, 123);
        assert_eq!(results.skipped, 456);
    }

    #[test]
    fn test_parse_zero_results() {
        let output = "test result: ok. 0 passed; 0 failed; 0 skipped";

        let results = parse_test_output(output, 0);

        assert_eq!(results.passed, 0);
        assert_eq!(results.failed, 0);
        assert_eq!(results.skipped, 0);
    }

    #[test]
    fn test_extract_failed_test_names() {
        let output = r#"
test auth::token_validation ... FAILED
test lifecycle::worker_startup ... ok
test auth::timing_safe ... FAILED
test scheduling::priority ... FAILED
"#;

        let failed_names = extract_failed_test_names(output);

        assert_eq!(failed_names.len(), 3);
        assert!(failed_names.contains(&"auth::token_validation".to_string()));
        assert!(failed_names.contains(&"auth::timing_safe".to_string()));
        assert!(failed_names.contains(&"scheduling::priority".to_string()));
    }

    #[test]
    fn test_extract_failed_test_names_none() {
        let output = r#"
test auth::token_validation ... ok
test lifecycle::worker_startup ... ok
"#;

        let failed_names = extract_failed_test_names(output);

        assert_eq!(failed_names.len(), 0);
    }

    #[test]
    fn test_extract_failed_test_names_with_special_chars() {
        let output = "test my_module::test_with_underscores ... FAILED";

        let failed_names = extract_failed_test_names(output);

        assert_eq!(failed_names.len(), 1);
        assert_eq!(failed_names[0], "my_module::test_with_underscores");
    }

    #[test]
    fn test_extract_all_failure_patterns() {
        let output = r#"
test auth::token_validation ... FAILED
Error: Authentication failed
assertion failed: expected true, got false
thread 'main' panicked at 'explicit panic'
"#;

        let patterns = extract_all_failure_patterns(output);

        // Should contain all failure patterns
        assert!(patterns.contains("FAILED"));
        assert!(patterns.contains("Error:"));
        assert!(patterns.contains("assertion"));
        assert!(patterns.contains("panicked"));
    }

    #[test]
    fn test_extract_all_failure_patterns_empty() {
        let output = "Everything is fine, no failures here!";

        let patterns = extract_all_failure_patterns(output);

        // Should be empty or minimal
        assert!(patterns.is_empty() || patterns.trim().is_empty());
    }

    #[test]
    fn test_parse_inconsistent_exit_code() {
        // Exit code 0 but tests failed - should warn
        let output = "test result: FAILED. 5 passed; 3 failed; 0 skipped";

        let results = parse_test_output(output, 0);

        assert_eq!(results.failed, 3);
        assert_eq!(results.exit_code, 0);
        // This should trigger a warning in the implementation
    }

    #[test]
    fn test_parse_multiple_result_lines() {
        // Some test runners might output multiple summary lines
        let output = r#"
test result: ok. 5 passed; 0 failed; 0 skipped
Some other text
test result: ok. 10 passed; 2 failed; 1 skipped
"#;

        let results = parse_test_output(output, 1);

        // Should pick up the first match
        assert!(results.passed > 0);
    }

    #[test]
    fn test_parse_with_unicode() {
        let output = "test result: ✅ ok. 5 passed; 0 failed; 0 skipped";

        let results = parse_test_output(output, 0);

        // Should handle unicode gracefully
        assert_eq!(results.passed, 5);
    }

    #[test]
    fn test_parse_with_ansi_codes() {
        // Simulated ANSI color codes
        let output = "\x1b[32mtest result: ok. 7 passed\x1b[0m; 1 failed; 0 skipped";

        let results = parse_test_output(output, 0);

        // Should still parse correctly despite ANSI codes
        assert_eq!(results.passed, 7);
        assert_eq!(results.failed, 1);
    }
}

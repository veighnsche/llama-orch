// TEAM-111: File generation behavior tests

#[cfg(test)]
mod tests {
    use super::super::files::*;
    use super::super::types::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_paths(temp_dir: &TempDir, timestamp: &str) -> OutputPaths {
        OutputPaths::new(temp_dir.path().to_path_buf(), timestamp)
    }

    #[test]
    fn test_generate_summary_file() {
        let temp_dir = TempDir::new().unwrap();
        let paths = create_test_paths(&temp_dir, "20251018_220000");
        
        let test_cmd = "cargo test --test cucumber";
        let results = TestResults {
            passed: 10,
            failed: 2,
            skipped: 1,
            exit_code: 1,
        };
        
        let result = generate_summary_file(&paths, test_cmd, &results);
        assert!(result.is_ok());
        
        // Verify file was created
        assert!(paths.results_file.exists());
        
        // Verify content
        let content = fs::read_to_string(&paths.results_file).unwrap();
        assert!(content.contains("BDD Test Results"));
        assert!(content.contains("cargo test --test cucumber"));
        assert!(content.contains("FAILED"));
        assert!(content.contains("Passed:  10"));
        assert!(content.contains("Failed:  2"));
        assert!(content.contains("Skipped: 1"));
    }

    #[test]
    fn test_generate_summary_file_success() {
        let temp_dir = TempDir::new().unwrap();
        let paths = create_test_paths(&temp_dir, "20251018_220000");
        
        let test_cmd = "cargo test --test cucumber";
        let results = TestResults {
            passed: 15,
            failed: 0,
            skipped: 0,
            exit_code: 0,
        };
        
        generate_summary_file(&paths, test_cmd, &results).unwrap();
        
        let content = fs::read_to_string(&paths.results_file).unwrap();
        assert!(content.contains("PASSED"));
        assert!(content.contains("Passed:  15"));
        assert!(content.contains("Failed:  0"));
    }

    #[test]
    fn test_generate_failure_files_no_failures() {
        let temp_dir = TempDir::new().unwrap();
        let mut paths = create_test_paths(&temp_dir, "20251018_220000");
        paths.set_failure_files("20251018_220000");
        
        // Create empty test output file
        fs::write(&paths.test_output, "").unwrap();
        
        let results = TestResults {
            passed: 10,
            failed: 0,
            skipped: 0,
            exit_code: 0,
        };
        
        let result = generate_failure_files(&paths, &results);
        assert!(result.is_ok());
        
        // Should not create failure files when no failures
        assert!(!paths.failures_file.as_ref().unwrap().exists());
    }

    #[test]
    fn test_generate_failure_files_with_failures() {
        let temp_dir = TempDir::new().unwrap();
        let mut paths = create_test_paths(&temp_dir, "20251018_220000");
        paths.set_failure_files("20251018_220000");
        
        // Create test output with failures
        let test_output = r#"
test auth::token_validation ... FAILED
test lifecycle::worker_startup ... ok
test auth::timing_safe ... FAILED

failures:

---- auth::token_validation stdout ----
Error: Authentication failed
thread 'auth::token_validation' panicked at 'assertion failed'
"#;
        fs::write(&paths.test_output, test_output).unwrap();
        
        let results = TestResults {
            passed: 8,
            failed: 2,
            skipped: 0,
            exit_code: 1,
        };
        
        let result = generate_failure_files(&paths, &results);
        assert!(result.is_ok());
        
        // Verify failures file was created
        let failures_file = paths.failures_file.as_ref().unwrap();
        assert!(failures_file.exists());
        
        // Verify content
        let content = fs::read_to_string(failures_file).unwrap();
        assert!(content.contains("FAILURE DETAILS"));
        assert!(content.contains("Failed Tests: 2"));
        assert!(content.contains("FAILED"));
        assert!(content.contains("Error:"));
        assert!(content.contains("Panics:"));
    }

    #[test]
    fn test_generate_rerun_command_file() {
        let temp_dir = TempDir::new().unwrap();
        let mut paths = create_test_paths(&temp_dir, "20251018_220000");
        paths.set_failure_files("20251018_220000");
        
        // Create test output with failed tests
        let test_output = r#"
test auth::token_validation ... FAILED
test auth::timing_safe ... FAILED
test scheduling::priority ... FAILED
"#;
        fs::write(&paths.test_output, test_output).unwrap();
        
        let results = TestResults {
            passed: 7,
            failed: 3,
            skipped: 0,
            exit_code: 1,
        };
        
        generate_failure_files(&paths, &results).unwrap();
        
        // Verify rerun command file was created
        let rerun_file = paths.rerun_file.as_ref().unwrap();
        assert!(rerun_file.exists());
        
        // Verify content
        let content = fs::read_to_string(rerun_file).unwrap();
        assert!(content.contains("Re-run failed tests"));
        assert!(content.contains("cd test-harness/bdd"));
        assert!(content.contains("cargo test --test cucumber"));
        assert!(content.contains("auth::token_validation"));
        assert!(content.contains("auth::timing_safe"));
        assert!(content.contains("scheduling::priority"));
        assert!(content.contains("--nocapture"));
    }

    #[test]
    fn test_generate_failure_files_missing_test_output() {
        let temp_dir = TempDir::new().unwrap();
        let mut paths = create_test_paths(&temp_dir, "20251018_220000");
        paths.set_failure_files("20251018_220000");
        
        // Don't create test output file
        
        let results = TestResults {
            passed: 0,
            failed: 1,
            skipped: 0,
            exit_code: 1,
        };
        
        let result = generate_failure_files(&paths, &results);
        
        // Should return error when test output file doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_summary_with_special_characters() {
        let temp_dir = TempDir::new().unwrap();
        let paths = create_test_paths(&temp_dir, "20251018_220000");
        
        let test_cmd = "cargo test --test cucumber -- --tags '@auth & @p0'";
        let results = TestResults {
            passed: 5,
            failed: 0,
            skipped: 0,
            exit_code: 0,
        };
        
        generate_summary_file(&paths, test_cmd, &results).unwrap();
        
        let content = fs::read_to_string(&paths.results_file).unwrap();
        assert!(content.contains("@auth"));
        assert!(content.contains("@p0"));
    }

    #[test]
    fn test_generate_failure_files_with_unicode() {
        let temp_dir = TempDir::new().unwrap();
        let mut paths = create_test_paths(&temp_dir, "20251018_220000");
        paths.set_failure_files("20251018_220000");
        
        let test_output = "test unicode::test_emoji ... FAILED\nError: 🚨 Something went wrong! 🔥";
        fs::write(&paths.test_output, test_output).unwrap();
        
        let results = TestResults {
            passed: 0,
            failed: 1,
            skipped: 0,
            exit_code: 1,
        };
        
        let result = generate_failure_files(&paths, &results);
        assert!(result.is_ok());
        
        // Should handle unicode gracefully
        let content = fs::read_to_string(paths.failures_file.as_ref().unwrap()).unwrap();
        assert!(content.contains("FAILED"));
    }

    #[test]
    fn test_generate_rerun_command_no_failed_tests() {
        let temp_dir = TempDir::new().unwrap();
        let mut paths = create_test_paths(&temp_dir, "20251018_220000");
        paths.set_failure_files("20251018_220000");
        
        // Test output with no FAILED markers
        let test_output = "test auth::token_validation ... ok\ntest lifecycle::worker_startup ... ok";
        fs::write(&paths.test_output, test_output).unwrap();
        
        let results = TestResults {
            passed: 2,
            failed: 0,
            skipped: 0,
            exit_code: 0,
        };
        
        generate_failure_files(&paths, &results).unwrap();
        
        // Rerun file should not be created if no failed tests found
        let rerun_file = paths.rerun_file.as_ref().unwrap();
        assert!(!rerun_file.exists());
    }

    #[test]
    fn test_generate_summary_with_long_command() {
        let temp_dir = TempDir::new().unwrap();
        let paths = create_test_paths(&temp_dir, "20251018_220000");
        
        let test_cmd = "cargo test --test cucumber -- --tags @auth --tags @p0 --tags @critical --feature lifecycle --feature authentication --nocapture --test-threads 1";
        let results = TestResults::default();
        
        generate_summary_file(&paths, test_cmd, &results).unwrap();
        
        let content = fs::read_to_string(&paths.results_file).unwrap();
        assert!(content.contains(test_cmd));
    }
}

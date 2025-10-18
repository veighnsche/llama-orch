// TEAM-111: Types behavior tests

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use std::path::PathBuf;

    #[test]
    fn test_output_paths_creation() {
        let log_dir = PathBuf::from("/tmp/test-logs");
        let timestamp = "20251018_220000";
        
        let paths = OutputPaths::new(log_dir.clone(), timestamp);
        
        assert_eq!(paths.log_dir, log_dir);
        assert!(paths.compile_log.to_string_lossy().contains("compile-20251018_220000.log"));
        assert!(paths.test_output.to_string_lossy().contains("test-output-20251018_220000.log"));
        assert!(paths.full_log.to_string_lossy().contains("bdd-test-20251018_220000.log"));
        assert!(paths.results_file.to_string_lossy().contains("bdd-results-20251018_220000.txt"));
        assert!(paths.failures_file.is_none());
        assert!(paths.rerun_file.is_none());
    }

    #[test]
    fn test_output_paths_set_failure_files() {
        let log_dir = PathBuf::from("/tmp/test-logs");
        let timestamp = "20251018_220000";
        
        let mut paths = OutputPaths::new(log_dir.clone(), timestamp);
        paths.set_failure_files(timestamp);
        
        assert!(paths.failures_file.is_some());
        assert!(paths.rerun_file.is_some());
        
        let failures_file = paths.failures_file.unwrap();
        assert!(failures_file.to_string_lossy().contains("failures-20251018_220000.txt"));
        
        let rerun_file = paths.rerun_file.unwrap();
        assert!(rerun_file.to_string_lossy().contains("rerun-failures-cmd.txt"));
    }

    #[test]
    fn test_test_results_default() {
        let results = TestResults::default();
        
        assert_eq!(results.passed, 0);
        assert_eq!(results.failed, 0);
        assert_eq!(results.skipped, 0);
        assert_eq!(results.exit_code, 0);
    }

    #[test]
    fn test_test_results_creation() {
        let results = TestResults {
            passed: 10,
            failed: 2,
            skipped: 1,
            exit_code: 1,
        };
        
        assert_eq!(results.passed, 10);
        assert_eq!(results.failed, 2);
        assert_eq!(results.skipped, 1);
        assert_eq!(results.exit_code, 1);
    }

    #[test]
    fn test_bdd_config_creation() {
        let config = BddConfig {
            tags: Some("@auth".to_string()),
            feature: Some("lifecycle".to_string()),
            quiet: true,
        };
        
        assert_eq!(config.tags, Some("@auth".to_string()));
        assert_eq!(config.feature, Some("lifecycle".to_string()));
        assert_eq!(config.quiet, true);
    }

    #[test]
    fn test_bdd_config_no_filters() {
        let config = BddConfig {
            tags: None,
            feature: None,
            quiet: false,
        };
        
        assert!(config.tags.is_none());
        assert!(config.feature.is_none());
        assert_eq!(config.quiet, false);
    }

    #[test]
    fn test_failure_info_creation() {
        let failure = FailureInfo {
            test_name: "auth::token_validation".to_string(),
            context: "assertion failed: expected true".to_string(),
        };
        
        assert_eq!(failure.test_name, "auth::token_validation");
        assert!(failure.context.contains("assertion failed"));
    }

    #[test]
    fn test_output_paths_with_special_chars_in_timestamp() {
        let log_dir = PathBuf::from("/tmp/logs");
        let timestamp = "2025-10-18_22:00:00"; // Colons in timestamp
        
        let paths = OutputPaths::new(log_dir, timestamp);
        
        // Should handle special characters
        assert!(paths.compile_log.to_string_lossy().contains(timestamp));
    }

    #[test]
    fn test_output_paths_with_empty_timestamp() {
        let log_dir = PathBuf::from("/tmp/logs");
        let timestamp = "";
        
        let paths = OutputPaths::new(log_dir, timestamp);
        
        // Should still create valid paths
        assert!(paths.compile_log.to_string_lossy().contains("compile-"));
    }

    #[test]
    fn test_test_results_large_numbers() {
        let results = TestResults {
            passed: 999999,
            failed: 123456,
            skipped: 789012,
            exit_code: 1,
        };
        
        assert_eq!(results.passed, 999999);
        assert_eq!(results.failed, 123456);
        assert_eq!(results.skipped, 789012);
    }

    #[test]
    fn test_bdd_config_clone() {
        let config1 = BddConfig {
            tags: Some("@p0".to_string()),
            feature: None,
            quiet: false,
        };
        
        let config2 = config1.clone();
        
        assert_eq!(config1.tags, config2.tags);
        assert_eq!(config1.feature, config2.feature);
        assert_eq!(config1.quiet, config2.quiet);
    }
}

// TEAM-111: Type definitions for BDD test runner

use std::path::PathBuf;

/// Configuration for BDD test execution
#[derive(Debug, Clone)]
pub struct BddConfig {
    pub tags: Option<String>,
    pub feature: Option<String>,
    pub quiet: bool,
    pub really_quiet: bool,
    pub show_quiet_warning: bool,
    pub run_all: bool,  // If true, run all tests. If false, run only failing tests from last run.
}

/// Test execution results
#[derive(Debug, Default)]
pub struct TestResults {
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub exit_code: i32,
}

/// Information about a failed test
#[derive(Debug, Clone)]
pub struct FailureInfo {
    pub test_name: String,
    pub context: String,
}

/// Paths for all output files
#[derive(Debug)]
pub struct OutputPaths {
    pub log_dir: PathBuf,
    pub compile_log: PathBuf,
    pub test_output: PathBuf,
    pub full_log: PathBuf,
    pub results_file: PathBuf,
    pub failures_file: Option<PathBuf>,
    pub rerun_file: Option<PathBuf>,
}

impl OutputPaths {
    pub fn new(log_dir: PathBuf, timestamp: &str) -> Self {
        Self {
            compile_log: log_dir.join(format!("compile-{}.log", timestamp)),
            test_output: log_dir.join(format!("test-output-{}.log", timestamp)),
            full_log: log_dir.join(format!("bdd-test-{}.log", timestamp)),
            results_file: log_dir.join(format!("bdd-results-{}.txt", timestamp)),
            failures_file: None,
            rerun_file: None,
            log_dir,
        }
    }

    pub fn set_failure_files(&mut self, timestamp: &str) {
        self.failures_file = Some(self.log_dir.join(format!("failures-{}.txt", timestamp)));
        self.rerun_file = Some(self.log_dir.join("rerun-failures-cmd.txt"));
    }
}

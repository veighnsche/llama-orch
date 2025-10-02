//! Test capture builder and implementation

use crate::fs::ProofBundle;
use super::types::{TestResult, TestStatus, TestSummary};
use anyhow::{Context, Result};
use serde_json::Value;
use std::process::Command;

/// Builder for capturing test results
pub struct TestCaptureBuilder<'a> {
    pb: &'a ProofBundle,
    package: String,
    lib: bool,
    tests: bool,
    benches: bool,
    doc: bool,
    features: Vec<String>,
    no_fail_fast: bool,
    test_threads: Option<usize>,
}

impl<'a> TestCaptureBuilder<'a> {
    /// Create new test capture builder
    pub fn new(pb: &'a ProofBundle, package: &str) -> Self {
        Self {
            pb,
            package: package.to_string(),
            lib: false,
            tests: false,
            benches: false,
            doc: false,
            features: Vec::new(),
            no_fail_fast: false,
            test_threads: None,
        }
    }
    
    /// Include unit tests (--lib)
    pub fn lib(mut self) -> Self {
        self.lib = true;
        self
    }
    
    /// Include integration tests (--tests)
    pub fn tests(mut self) -> Self {
        self.tests = true;
        self
    }
    
    /// Include benchmarks (--benches)
    pub fn benches(mut self) -> Self {
        self.benches = true;
        self
    }
    
    /// Include doc tests (--doc)
    pub fn doc(mut self) -> Self {
        self.doc = true;
        self
    }
    
    /// Include all test types
    pub fn all(mut self) -> Self {
        self.lib = true;
        self.tests = true;
        self.benches = true;
        self.doc = true;
        self
    }
    
    /// Enable specific features
    pub fn features(mut self, features: &[&str]) -> Self {
        self.features = features.iter().map(|s| s.to_string()).collect();
        self
    }
    
    /// Don't stop on first failure
    pub fn no_fail_fast(mut self) -> Self {
        self.no_fail_fast = true;
        self
    }
    
    /// Set test thread count
    pub fn test_threads(mut self, n: usize) -> Self {
        self.test_threads = Some(n);
        self
    }
    
    /// Run tests and capture results
    ///
    /// This will:
    /// 1. Run cargo test with --format json
    /// 2. Parse test results
    /// 3. Write to test_results.ndjson
    /// 4. Write summary.json
    /// 5. Generate test_report.md
    /// 6. Return TestSummary
    ///
    /// **IMPORTANT**: Results are captured even if tests fail (per PB-002 policy)
    pub fn run(self) -> Result<TestSummary> {
        // Build cargo test command
        let mut cmd = Command::new("cargo");
        cmd.arg("test");
        cmd.arg("-p");
        cmd.arg(&self.package);
        
        // Add test type flags
        if self.lib {
            cmd.arg("--lib");
        }
        if self.tests {
            cmd.arg("--tests");
        }
        if self.benches {
            cmd.arg("--benches");
        }
        if self.doc {
            cmd.arg("--doc");
        }
        
        // Add features
        if !self.features.is_empty() {
            cmd.arg("--features");
            cmd.arg(self.features.join(","));
        }
        
        // Add test arguments
        cmd.arg("--");
        
        // Try JSON format (requires nightly or -Z unstable-options)
        cmd.arg("--format");
        cmd.arg("json");
        cmd.arg("-Z");
        cmd.arg("unstable-options");
        
        if self.no_fail_fast {
            cmd.arg("--no-fail-fast");
        }
        
        if let Some(n) = self.test_threads {
            cmd.arg("--test-threads");
            cmd.arg(n.to_string());
        }
        
        // Disable backtrace for cleaner output
        cmd.env("RUST_BACKTRACE", "0");
        
        // Run command (don't check exit code - capture results even on failure)
        let output = cmd.output()
            .context("Failed to run cargo test")?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // Check if JSON format failed (unstable options not available)
        if stderr.contains("unstable") || stderr.contains("--format") {
            return Err(anyhow::anyhow!(
                "cargo test --format json requires nightly Rust or -Z unstable-options.\n\
                 Please run with: cargo +nightly test\n\
                 Or use stable Rust with manual test capture."
            ));
        }
        
        // Parse JSON test results
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
        
        // Write test results to NDJSON
        for result in &test_results {
            self.pb.append_ndjson("test_results", result)
                .context("Failed to write test result")?;
        }
        
        // Create summary
        let summary = TestSummary {
            total,
            passed,
            failed,
            ignored,
            duration_secs: total_duration,
            pass_rate,
            tests: test_results,
        };
        
        // Write summary JSON
        self.pb.write_json("summary", &summary)
            .context("Failed to write summary")?;
        
        // Generate human-readable report
        let report = self.generate_report(&summary);
        self.pb.write_markdown("test_report.md", &report)
            .context("Failed to write test report")?;
        
        Ok(summary)
    }
    
    /// Generate human-readable test report
    fn generate_report(&self, summary: &TestSummary) -> String {
        let status_emoji = if summary.failed == 0 {
            "✅"
        } else {
            "❌"
        };
        
        let mut report = format!(
            "# AUTOGENERATED: Proof Bundle\n\n\
            # Test Report - {}\n\n\
            **Status**: {} {}\n\
            **Total Tests**: {}\n\
            **Passed**: {} ({:.1}%)\n\
            **Failed**: {}\n\
            **Ignored**: {}\n\
            **Duration**: {:.2}s\n\n\
            ---\n\n",
            self.package,
            status_emoji,
            if summary.failed == 0 { "ALL TESTS PASSED" } else { "TESTS FAILED" },
            summary.total,
            summary.passed,
            summary.pass_rate,
            summary.failed,
            summary.ignored,
            summary.duration_secs,
        );
        
        // Add test type coverage
        report.push_str("## Test Coverage\n\n");
        if self.lib {
            report.push_str("- ✅ Unit tests (--lib)\n");
        }
        if self.tests {
            report.push_str("- ✅ Integration tests (--tests)\n");
        }
        if self.benches {
            report.push_str("- ✅ Benchmarks (--benches)\n");
        }
        if self.doc {
            report.push_str("- ✅ Doc tests (--doc)\n");
        }
        report.push_str("\n---\n\n");
        
        // Add failed tests section if any
        if summary.failed > 0 {
            report.push_str("## Failed Tests\n\n");
            for test in &summary.tests {
                if test.status == TestStatus::Failed {
                    report.push_str(&format!("### ❌ {}\n\n", test.name));
                    if let Some(ref err) = test.error_message {
                        report.push_str("**Error**:\n```\n");
                        report.push_str(err);
                        report.push_str("\n```\n\n");
                    }
                    report.push_str(&format!("**Duration**: {:.3}s\n\n", test.duration_secs));
                }
            }
            report.push_str("---\n\n");
        }
        
        // Add data files section
        report.push_str("## Evidence Files\n\n");
        report.push_str("- `test_results.ndjson` - All test results with timing\n");
        report.push_str("- `summary.json` - Aggregate statistics\n");
        report.push_str("- `test_report.md` - This human-readable report\n\n");
        
        report.push_str("---\n\n");
        report.push_str(&format!("**Package**: {}\n", self.package));
        report.push_str(&format!("**Tests Captured**: {}\n", summary.total));
        
        report
    }
}

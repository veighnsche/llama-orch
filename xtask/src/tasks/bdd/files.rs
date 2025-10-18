// TEAM-111: File generation logic

use super::parser;
use super::types::{OutputPaths, TestResults};
use anyhow::{Context, Result};
use std::fs;
use std::io::Write;

/// Generate failures file with detailed information
pub fn generate_failure_files(paths: &OutputPaths, results: &TestResults) -> Result<()> {
    if results.failed == 0 {
        return Ok(());
    }

    let output_content = fs::read_to_string(&paths.test_output)
        .context(format!("Failed to read test output from {}", paths.test_output.display()))?;

    // Generate failures file
    if let Some(ref failures_file) = paths.failures_file {
        let mut file = fs::File::create(failures_file)
            .context(format!("Failed to create failures file at {}", failures_file.display()))?;

        writeln!(file, "FAILURE DETAILS")?;
        writeln!(file, "========================================")?;
        writeln!(file)?;
        writeln!(file, "Failed Tests: {}", results.failed)?;
        writeln!(file)?;
        writeln!(file, "========================================")?;
        writeln!(file)?;

        // Extract all failure patterns
        let patterns = parser::extract_all_failure_patterns(&output_content);
        writeln!(file, "{}", patterns)?;

        writeln!(file)?;
        writeln!(file, "========================================")?;
        writeln!(file, "Errors:")?;
        writeln!(file, "========================================")?;

        for line in output_content.lines() {
            if line.contains("Error:") {
                writeln!(file, "{}", line)?;
            }
        }

        writeln!(file)?;
        writeln!(file, "========================================")?;
        writeln!(file, "Panics:")?;
        writeln!(file, "========================================")?;

        for line in output_content.lines() {
            if line.contains("panicked at") {
                writeln!(file, "{}", line)?;
            }
        }
    }

    // Generate rerun command file
    if let Some(ref rerun_file) = paths.rerun_file {
        let failed_tests = parser::extract_failed_test_names(&output_content);

        if !failed_tests.is_empty() {
            let mut file = fs::File::create(rerun_file)?;

            writeln!(file, "# Re-run failed tests")?;
            writeln!(file, "# Copy and paste the command below:")?;
            writeln!(file)?;
            writeln!(file, "cd test-harness/bdd")?;

            let tests_str = failed_tests.join(" ");
            writeln!(file, "cargo test --test cucumber {} -- --nocapture", tests_str)?;
        }
    }

    Ok(())
}

/// Generate summary file
pub fn generate_summary_file(
    paths: &OutputPaths,
    test_cmd: &str,
    results: &TestResults,
) -> Result<()> {
    let mut file = fs::File::create(&paths.results_file)?;

    writeln!(file, "BDD Test Results")?;
    writeln!(file, "================================")?;
    writeln!(file)?;
    writeln!(file, "Command: {}", test_cmd)?;
    writeln!(file, "Status: {}", if results.exit_code == 0 { "PASSED" } else { "FAILED" })?;
    writeln!(file)?;
    writeln!(file, "Summary:")?;
    writeln!(file, "  Passed:  {}", results.passed)?;
    writeln!(file, "  Failed:  {}", results.failed)?;
    writeln!(file, "  Skipped: {}", results.skipped)?;
    writeln!(file)?;
    writeln!(file, "Full log: {}", paths.full_log.display())?;

    Ok(())
}

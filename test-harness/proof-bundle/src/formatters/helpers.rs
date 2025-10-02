//! Helper functions for formatters

use crate::{TestResult, TestStatus};
use std::collections::HashMap;

/// Categorize tests by type based on naming conventions
pub(super) fn categorize_tests(tests: &[TestResult]) -> HashMap<String, Vec<TestResult>> {
    let mut categories: HashMap<String, Vec<TestResult>> = HashMap::new();
    
    for test in tests {
        let category = if test.name.contains("property") || test.name.contains("proptest") {
            "Property Tests".to_string()
        } else if test.name.contains("integration") || test.name.contains("e2e") {
            "Integration Tests".to_string()
        } else if test.name.contains("bdd") || test.name.contains("scenario") {
            "BDD Tests".to_string()
        } else if test.name.contains("stress") || test.name.contains("load") {
            "Stress Tests".to_string()
        } else if test.name.contains("bench") {
            "Benchmarks".to_string()
        } else {
            "Unit Tests".to_string()
        };
        
        categories.entry(category).or_insert_with(Vec::new).push(test.clone());
    }
    
    categories
}

/// Simplify technical error messages for management audience
pub(super) fn simplify_error(error: &str) -> String {
    // Remove technical details, keep core message
    if error.contains("assertion") {
        "Test expectation not met".to_string()
    } else if error.contains("panicked") {
        "Test encountered unexpected condition".to_string()
    } else if error.contains("timeout") || error.contains("timed out") {
        "Test exceeded time limit".to_string()
    } else if error.contains("memory") || error.contains("OOM") {
        "Memory allocation issue".to_string()
    } else {
        // Take first line, truncate to 80 chars
        error.lines().next()
            .unwrap_or("Unknown error")
            .chars().take(80).collect::<String>()
    }
}

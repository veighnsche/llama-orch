//! BDD step implementation analyzer
//!
//! Analyzes step definitions to identify stubs, TODOs, and implementation status
//!
//! Created by: TEAM-124

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Analysis results for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAnalysis {
    pub name: String,
    pub path: PathBuf,
    pub total_functions: usize,
    pub unused_world: usize,
    pub todos: usize,
    pub stub_count: usize,
    pub stub_percentage: f64,
    pub stub_functions: Vec<StubFunction>,
}

/// Information about a stub function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StubFunction {
    pub name: String,
    pub line_number: usize,
    pub has_unused_world: bool,
    pub has_todo: bool,
    pub signature: String,
}

/// Overall analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResults {
    pub total_files: usize,
    pub total_functions: usize,
    pub total_unused_world: usize,
    pub total_todos: usize,
    pub total_stubs: usize,
    pub implementation_percentage: f64,
    pub files: Vec<FileAnalysis>,
    pub timestamp: String,
}

impl AnalysisResults {
    /// Get files sorted by stub count (descending)
    pub fn files_by_stub_count(&self) -> Vec<&FileAnalysis> {
        let mut files: Vec<&FileAnalysis> = self.files.iter().collect();
        files.sort_by(|a, b| b.stub_count.cmp(&a.stub_count));
        files
    }

    /// Get files with stubs only
    pub fn files_with_stubs(&self) -> Vec<&FileAnalysis> {
        self.files.iter().filter(|f| f.stub_count > 0).collect()
    }

    /// Get files without stubs
    pub fn complete_files(&self) -> Vec<&FileAnalysis> {
        self.files.iter().filter(|f| f.stub_count == 0).collect()
    }

    /// Save to JSON file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let results = serde_json::from_str(&json)?;
        Ok(results)
    }
}

/// Analyze all step files in the BDD directory
pub fn analyze_bdd_steps(steps_dir: &Path) -> Result<AnalysisResults> {
    let mut files = Vec::new();
    let mut total_functions = 0;
    let mut total_unused_world = 0;
    let mut total_todos = 0;

    // Find all .rs files in steps directory
    for entry in fs::read_dir(steps_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            if let Ok(analysis) = analyze_file(&path) {
                total_functions += analysis.total_functions;
                total_unused_world += analysis.unused_world;
                total_todos += analysis.todos;
                files.push(analysis);
            }
        }
    }

    let total_stubs = total_unused_world + total_todos;
    let implementation_percentage = if total_functions > 0 {
        ((total_functions - total_stubs) as f64 / total_functions as f64) * 100.0
    } else {
        0.0
    };

    Ok(AnalysisResults {
        total_files: files.len(),
        total_functions,
        total_unused_world,
        total_todos,
        total_stubs,
        implementation_percentage,
        files,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

/// Analyze a single step file
fn analyze_file(path: &Path) -> Result<FileAnalysis> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();

    let mut total_functions = 0;
    let mut unused_world = 0;
    let mut todos = 0;
    let mut stub_functions = Vec::new();

    let mut current_function: Option<(String, usize, String)> = None; // (name, line, signature)
    let mut function_has_unused_world = false;
    let mut function_has_todo = false;

    for (line_num, line) in lines.iter().enumerate() {
        let line_num = line_num + 1; // 1-indexed

        // Detect function start
        if line.contains("pub async fn") || line.contains("pub fn") {
            // Save previous function if it was a stub
            if let Some((name, start_line, sig)) = current_function.take() {
                if function_has_unused_world || function_has_todo {
                    stub_functions.push(StubFunction {
                        name,
                        line_number: start_line,
                        has_unused_world: function_has_unused_world,
                        has_todo: function_has_todo,
                        signature: sig,
                    });
                }
            }

            // Start tracking new function
            total_functions += 1;
            function_has_unused_world = false;
            function_has_todo = false;

            // Extract function name
            if let Some(name) = extract_function_name(line) {
                current_function = Some((name, line_num, line.trim().to_string()));
            }
        }

        // Check for unused world parameter
        if line.contains("_world: &mut World") {
            unused_world += 1;
            function_has_unused_world = true;
        }

        // Check for TODO markers
        if line.contains("TODO:") || line.contains("TODO ") {
            todos += 1;
            function_has_todo = true;
        }
    }

    // Save last function if it was a stub
    if let Some((name, start_line, sig)) = current_function {
        if function_has_unused_world || function_has_todo {
            stub_functions.push(StubFunction {
                name,
                line_number: start_line,
                has_unused_world: function_has_unused_world,
                has_todo: function_has_todo,
                signature: sig,
            });
        }
    }

    let stub_count = unused_world + todos;
    let stub_percentage = if total_functions > 0 {
        (stub_count as f64 / total_functions as f64) * 100.0
    } else {
        0.0
    };

    Ok(FileAnalysis {
        name: path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
        path: path.to_path_buf(),
        total_functions,
        unused_world,
        todos,
        stub_count,
        stub_percentage,
        stub_functions,
    })
}

/// Extract function name from a function declaration line
fn extract_function_name(line: &str) -> Option<String> {
    // Match patterns like: pub async fn function_name(
    let parts: Vec<&str> = line.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "fn" && i + 1 < parts.len() {
            let name = parts[i + 1];
            // Remove parameter list
            if let Some(paren_pos) = name.find('(') {
                return Some(name[..paren_pos].to_string());
            } else if let Some(angle_pos) = name.find('<') {
                return Some(name[..angle_pos].to_string());
            } else {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Print analysis results in text format
pub fn print_text_report(results: &AnalysisResults, detailed: bool, stubs_only: bool) {
    println!("=== BDD STEP IMPLEMENTATION ANALYSIS ===\n");
    println!("Analyzed: {}", results.timestamp);
    println!("Total step files: {}", results.total_files);
    println!("Total step functions: {}", results.total_functions);
    println!(
        "Functions with unused _world: {} ({:.1}%)",
        results.total_unused_world,
        (results.total_unused_world as f64 / results.total_functions as f64) * 100.0
    );
    println!(
        "Functions with TODO markers: {} ({:.1}%)",
        results.total_todos,
        (results.total_todos as f64 / results.total_functions as f64) * 100.0
    );
    println!(
        "\nEstimated stub functions: {} ({:.1}%)",
        results.total_stubs,
        (results.total_stubs as f64 / results.total_functions as f64) * 100.0
    );
    println!(
        "✅ Implemented: ~{} functions ({:.1}%)",
        results.total_functions - results.total_stubs,
        results.implementation_percentage
    );

    if detailed {
        println!("\n=== FILES WITH STUBS ===\n");
        println!(
            "{:<40} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "File", "Functions", "Unused", "TODOs", "Stubs", "% Stub"
        );
        println!("{}", "-".repeat(90));

        let files = if stubs_only {
            results.files_with_stubs()
        } else {
            results.files_by_stub_count()
        };

        for file in files {
            if stubs_only && file.stub_count == 0 {
                continue;
            }
            println!(
                "{:<40} {:>10} {:>8} {:>8} {:>8} {:>7.1}%",
                file.name,
                file.total_functions,
                file.unused_world,
                file.todos,
                file.stub_count,
                file.stub_percentage
            );
        }
    }

    // Show top 10 files needing work
    println!("\n=== TOP 10 FILES NEEDING WORK ===\n");
    let top_files: Vec<&FileAnalysis> = results
        .files_by_stub_count()
        .into_iter()
        .filter(|f| f.stub_count > 0)
        .take(10)
        .collect();

    for (i, file) in top_files.iter().enumerate() {
        let priority = if file.stub_percentage > 50.0 {
            "🔴 CRITICAL"
        } else if file.stub_percentage > 20.0 {
            "🟡 MODERATE"
        } else {
            "🟢 LOW"
        };

        println!(
            "{}. {} - {} ({} stubs, {:.1}%)",
            i + 1,
            file.name,
            priority,
            file.stub_count,
            file.stub_percentage
        );
    }

    // Show work estimation
    println!("\n=== WORK ESTIMATION ===\n");
    let critical_stubs: usize = results
        .files
        .iter()
        .filter(|f| f.stub_percentage > 50.0)
        .map(|f| f.stub_count)
        .sum();
    let moderate_stubs: usize = results
        .files
        .iter()
        .filter(|f| f.stub_percentage > 20.0 && f.stub_percentage <= 50.0)
        .map(|f| f.stub_count)
        .sum();
    let low_stubs: usize = results
        .files
        .iter()
        .filter(|f| f.stub_percentage > 0.0 && f.stub_percentage <= 20.0)
        .map(|f| f.stub_count)
        .sum();

    println!("🔴 CRITICAL (>50% stubs): {} stubs × 20 min = {:.1} hours", critical_stubs, critical_stubs as f64 * 20.0 / 60.0);
    println!("🟡 MODERATE (20-50% stubs): {} stubs × 15 min = {:.1} hours", moderate_stubs, moderate_stubs as f64 * 15.0 / 60.0);
    println!("🟢 LOW (<20% stubs): {} stubs × 10 min = {:.1} hours", low_stubs, low_stubs as f64 * 10.0 / 60.0);
    
    let total_hours = (critical_stubs as f64 * 20.0 + moderate_stubs as f64 * 15.0 + low_stubs as f64 * 10.0) / 60.0;
    println!("\n📊 TOTAL ESTIMATE: {:.1} hours ({:.1} days)", total_hours, total_hours / 8.0);

    // Show complete files
    let complete = results.complete_files();
    if !complete.is_empty() {
        println!("\n=== COMPLETE FILES ({}) ===\n", complete.len());
        for file in complete.iter().take(10) {
            println!("✅ {} ({} functions)", file.name, file.total_functions);
        }
        if complete.len() > 10 {
            println!("... and {} more", complete.len() - 10);
        }
    }
}

/// Print analysis results in JSON format
pub fn print_json_report(results: &AnalysisResults) -> Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    println!("{}", json);
    Ok(())
}

/// Print analysis results in Markdown format
pub fn print_markdown_report(results: &AnalysisResults) {
    println!("# BDD Step Implementation Analysis\n");
    println!("**Generated:** {}\n", results.timestamp);
    println!("## Summary\n");
    println!("- **Total Files:** {}", results.total_files);
    println!("- **Total Functions:** {}", results.total_functions);
    println!(
        "- **Implemented:** {} ({:.1}%)",
        results.total_functions - results.total_stubs,
        results.implementation_percentage
    );
    println!(
        "- **Stubs/TODOs:** {} ({:.1}%)",
        results.total_stubs,
        (results.total_stubs as f64 / results.total_functions as f64) * 100.0
    );

    println!("\n## Top Files Needing Work\n");
    println!("| File | Functions | Stubs | % Stubbed | Priority |");
    println!("|------|-----------|-------|-----------|----------|");

    for file in results.files_by_stub_count().iter().take(10) {
        if file.stub_count == 0 {
            continue;
        }
        let priority = if file.stub_percentage > 50.0 {
            "🔴 CRITICAL"
        } else if file.stub_percentage > 20.0 {
            "🟡 MODERATE"
        } else {
            "🟢 LOW"
        };

        println!(
            "| {} | {} | {} | {:.1}% | {} |",
            file.name, file.total_functions, file.stub_count, file.stub_percentage, priority
        );
    }

    println!("\n## Implementation Status\n");
    println!("```");
    println!(
        "✅ Implemented: {} functions ({:.1}%)",
        results.total_functions - results.total_stubs,
        results.implementation_percentage
    );
    println!(
        "⚠️  Stubs/TODOs: {} functions ({:.1}%)",
        results.total_stubs,
        (results.total_stubs as f64 / results.total_functions as f64) * 100.0
    );
    println!("```");
}

/// Print stub details for a specific file
pub fn print_file_stubs(results: &AnalysisResults, file_name: &str) -> Result<()> {
    let file = results
        .files
        .iter()
        .find(|f| f.name == file_name)
        .ok_or_else(|| anyhow::anyhow!("File not found: {}", file_name))?;

    println!("=== STUB ANALYSIS: {} ===\n", file.name);
    println!("Total functions: {}", file.total_functions);
    println!("Stub functions: {} ({:.1}%)", file.stub_count, file.stub_percentage);
    println!();

    if file.stub_functions.is_empty() {
        println!("✅ No stubs found! This file is complete.");
        return Ok(());
    }

    println!("Stub functions:\n");
    for stub in &file.stub_functions {
        let markers = if stub.has_unused_world && stub.has_todo {
            "unused _world + TODO"
        } else if stub.has_unused_world {
            "unused _world"
        } else {
            "TODO"
        };

        println!("Line {}: {} ({})", stub.line_number, stub.name, markers);
        println!("  {}", stub.signature);
        println!();
    }

    Ok(())
}

/// Compare current analysis with previous run
pub fn compare_progress(current: &AnalysisResults, previous: &AnalysisResults) {
    println!("=== BDD PROGRESS COMPARISON ===\n");
    println!("Previous: {}", previous.timestamp);
    println!("Current:  {}\n", current.timestamp);

    let impl_diff = current.implementation_percentage - previous.implementation_percentage;
    let stubs_diff = current.total_stubs as i32 - previous.total_stubs as i32;

    println!("Implementation: {:.1}% → {:.1}% ({:+.1}%)",
        previous.implementation_percentage,
        current.implementation_percentage,
        impl_diff
    );

    println!("Stubs: {} → {} ({:+})",
        previous.total_stubs,
        current.total_stubs,
        stubs_diff
    );

    if impl_diff > 0.0 {
        println!("\n✅ Progress made! {:.1}% more functions implemented", impl_diff);
    } else if impl_diff < 0.0 {
        println!("\n⚠️  Regression! {:.1}% fewer functions implemented", impl_diff.abs());
    } else {
        println!("\n➡️  No change in implementation percentage");
    }

    // Show files that improved
    let mut improved = Vec::new();
    let mut regressed = Vec::new();

    for current_file in &current.files {
        if let Some(prev_file) = previous.files.iter().find(|f| f.name == current_file.name) {
            let diff = current_file.stub_count as i32 - prev_file.stub_count as i32;
            if diff < 0 {
                improved.push((current_file.name.clone(), diff.abs()));
            } else if diff > 0 {
                regressed.push((current_file.name.clone(), diff));
            }
        }
    }

    if !improved.is_empty() {
        println!("\n=== FILES IMPROVED ===\n");
        for (name, reduction) in improved {
            println!("✅ {} (-{} stubs)", name, reduction);
        }
    }

    if !regressed.is_empty() {
        println!("\n=== FILES REGRESSED ===\n");
        for (name, increase) in regressed {
            println!("⚠️  {} (+{} stubs)", name, increase);
        }
    }
}

// TEAM-123: Duplicate step definition checker
// Scans BDD step files for duplicate cucumber step patterns

use anyhow::{Context, Result};
use colored::Colorize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct StepDefinition {
    pattern: String,
    file: String,
    line: usize,
    function_name: String,
}

pub fn check_duplicate_steps() -> Result<()> {
    println!("{}", "🔍 Checking for duplicate step definitions...".cyan().bold());
    println!();

    let bdd_dir = get_bdd_directory()?;
    let steps_dir = bdd_dir.join("src/steps");

    if !steps_dir.exists() {
        anyhow::bail!("Steps directory not found: {}", steps_dir.display());
    }

    println!("{} {}", "📁 Scanning:".blue(), steps_dir.display());
    println!();

    // Collect all step definitions
    let mut step_map: HashMap<String, Vec<StepDefinition>> = HashMap::new();
    let mut total_files = 0;
    let mut total_steps = 0;

    for entry in walkdir::WalkDir::new(&steps_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
    {
        let path = entry.path();
        let relative_path = path.strip_prefix(&steps_dir).unwrap_or(path);

        total_files += 1;
        let content =
            fs::read_to_string(path).context(format!("Failed to read {}", path.display()))?;

        let steps = extract_step_definitions(&content, relative_path.to_string_lossy().to_string());
        total_steps += steps.len();

        for step in steps {
            step_map.entry(step.pattern.clone()).or_insert_with(Vec::new).push(step);
        }
    }

    println!("{} {} files, {} step definitions", "📊 Found:".green(), total_files, total_steps);
    println!();

    // Find duplicates
    let mut duplicates_found = false;
    let mut duplicate_count = 0;

    for (pattern, definitions) in step_map.iter() {
        if definitions.len() > 1 {
            duplicates_found = true;
            duplicate_count += definitions.len() - 1;

            println!("{} {}", "❌ DUPLICATE STEP:".red().bold(), pattern.yellow());
            println!();

            for def in definitions {
                println!("  {} {}:{}", "→".red(), def.file.cyan(), def.line.to_string().yellow());
                println!("    Function: {}", def.function_name.white());
            }
            println!();
        }
    }

    if duplicates_found {
        println!();
        println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".red());
        println!(
            "{} {} duplicate step definition(s) found!",
            "❌ FAILURE:".red().bold(),
            duplicate_count
        );
        println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".red());
        println!();
        println!("{}", "⚠️  CRITICAL: Duplicate steps cause cucumber to HANG!".yellow().bold());
        println!();
        println!("To fix:");
        println!("  1. Remove or rename one of the duplicate functions");
        println!("  2. Make sure each step pattern is unique across all files");
        println!("  3. Run this check again: cargo xtask bdd:check-duplicates");
        println!();

        std::process::exit(1);
    } else {
        println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".green());
        println!("{} No duplicate step definitions found!", "✅ SUCCESS:".green().bold());
        println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".green());
        println!();
        println!("All step definitions are unique. Safe to run tests!");
        println!();
    }

    Ok(())
}

fn extract_step_definitions(content: &str, file: String) -> Vec<StepDefinition> {
    let mut definitions = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Look for step attributes: #[given(...)] #[when(...)] #[then(...)]
        if trimmed.starts_with("#[given(")
            || trimmed.starts_with("#[when(")
            || trimmed.starts_with("#[then(")
        {
            // Extract the pattern from expr = "..." or regex = r"..."
            if let Some(pattern) = extract_pattern(trimmed) {
                // Next non-empty line should be the function definition
                if let Some(func_name) = find_function_name(&lines, i + 1) {
                    definitions.push(StepDefinition {
                        pattern,
                        file: file.clone(),
                        line: i + 1, // 1-indexed
                        function_name: func_name,
                    });
                }
            }
        }
    }

    definitions
}

fn extract_pattern(attribute_line: &str) -> Option<String> {
    // Handle expr = "pattern"
    if let Some(start) = attribute_line.find("expr = \"") {
        let after_expr = &attribute_line[start + 8..];
        if let Some(end) = after_expr.find('"') {
            return Some(after_expr[..end].to_string());
        }
    }

    // Handle regex = r#"pattern"# or regex = r"pattern" or regex = "pattern"
    if let Some(start) = attribute_line.find("regex = ") {
        let after_regex = &attribute_line[start + 8..].trim_start();

        // Handle r#"..."#
        if after_regex.starts_with("r#\"") {
            let content = &after_regex[3..];
            if let Some(end) = content.find("\"#") {
                return Some(content[..end].to_string());
            }
        }
        // Handle r"..."
        else if after_regex.starts_with("r\"") {
            let content = &after_regex[2..];
            if let Some(end) = content.find('"') {
                return Some(content[..end].to_string());
            }
        }
        // Handle "..."
        else if after_regex.starts_with('"') {
            let content = &after_regex[1..];
            if let Some(end) = content.find('"') {
                return Some(content[..end].to_string());
            }
        }
    }

    None
}

fn find_function_name(lines: &[&str], start_idx: usize) -> Option<String> {
    // Look for "pub async fn function_name" in the next few lines
    for i in start_idx..std::cmp::min(start_idx + 5, lines.len()) {
        let line = lines[i].trim();
        if line.starts_with("pub async fn ")
            || line.starts_with("async fn ")
            || line.starts_with("pub fn ")
        {
            // Extract function name
            let after_fn = if line.starts_with("pub async fn ") {
                &line[13..]
            } else if line.starts_with("async fn ") {
                &line[9..]
            } else {
                &line[7..]
            };

            if let Some(paren_pos) = after_fn.find('(') {
                return Some(after_fn[..paren_pos].to_string());
            }
        }
    }
    None
}

fn get_bdd_directory() -> Result<PathBuf> {
    let root = crate::util::repo_root()?;
    Ok(root.join("test-harness/bdd"))
}

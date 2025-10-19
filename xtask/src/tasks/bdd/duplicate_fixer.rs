// TEAM-123: Automated duplicate step fixer
// Removes duplicate step definitions based on predefined rules

use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

pub fn fix_all_duplicates() -> Result<()> {
    println!("{}", "🔧 Auto-fixing duplicate step definitions...".cyan().bold());
    println!();

    let bdd_dir = get_bdd_directory()?;
    let steps_dir = bdd_dir.join("src/steps");

    // List of duplicates to remove (file, line_number, reason)
    let duplicates_to_remove = vec![
        ("worker_preflight.rs", 354, "validation fails", "Keep error_handling.rs:745"),
        ("integration.rs", 290, "worker returns to idle state", "Keep lifecycle.rs:562"),
        ("error_handling.rs", 1438, "error message does not contain {string}", "Keep validation.rs:353"),
        ("deadline_propagation.rs", 311, "worker is processing inference request", "Keep error_handling.rs:961"),
        ("audit_logging.rs", 589, "queen-rbee logs warning {string}", "Keep audit_logging.rs:385"),
        ("authentication.rs", 28, "I send POST to {string} without Authorization header", "Keep authentication.rs:782"),
        ("authentication.rs", 690, "I send {int} authenticated requests", "Keep authentication.rs:814"),
        ("authentication.rs", 352, "I send GET to {string} without Authorization header", "Keep authentication.rs:798"),
        ("worker_registration.rs", 89, "rbee-hive reports worker {string} with capabilities {string}", "Keep queen_rbee_registry.rs:126"),
        ("error_handling.rs", 1465, "rbee-hive continues running (does NOT crash)", "Keep errors.rs:113"),
        ("secrets.rs", 51, "queen-rbee starts with config:", "Keep configuration_management.rs:655"),
        ("worker_registration.rs", 115, "the response contains {int} worker(s)", "Keep queen_rbee_registry.rs:237"),
        ("pid_tracking.rs", 18, "rbee-hive spawns a worker process", "Keep lifecycle.rs:540"),
        ("error_handling.rs", 524, "rbee-hive detects worker crash", "Keep lifecycle.rs:574"),
        ("authentication.rs", 123, "log contains {string}", "Keep configuration_management.rs:669"),
        ("secrets.rs", 82, "systemd credential exists at {string}", "Keep secrets.rs:363"),
    ];

    let mut fixed_count = 0;

    for (file, line_num, pattern, reason) in duplicates_to_remove {
        let file_path = steps_dir.join(file);
        
        match comment_out_step_definition(&file_path, line_num, pattern, reason) {
            Ok(true) => {
                println!("{} {}:{} - {}", "✅ Fixed:".green(), file, line_num, pattern);
                fixed_count += 1;
            }
            Ok(false) => {
                println!("{} {}:{} - {} (already fixed or not found)", "⏭️  Skipped:".yellow(), file, line_num, pattern);
            }
            Err(e) => {
                println!("{} {}:{} - {}: {}", "❌ Error:".red(), file, line_num, pattern, e);
            }
        }
    }

    println!();
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".green());
    println!("{} {} duplicate step definitions fixed!", "✅ SUCCESS:".green().bold(), fixed_count);
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".green());
    println!();
    println!("Run 'cargo xtask bdd:check-duplicates' to verify all duplicates are fixed.");
    println!();

    Ok(())
}

fn comment_out_step_definition(
    file_path: &PathBuf,
    target_line: usize,
    pattern: &str,
    reason: &str,
) -> Result<bool> {
    let content = fs::read_to_string(file_path)
        .context(format!("Failed to read {}", file_path.display()))?;

    let lines: Vec<&str> = content.lines().collect();
    
    if target_line == 0 || target_line > lines.len() {
        return Ok(false);
    }

    let line_idx = target_line - 1; // Convert to 0-indexed
    let line = lines[line_idx];

    // Check if this line contains a step attribute
    if !line.trim().starts_with("#[given(") && 
       !line.trim().starts_with("#[when(") && 
       !line.trim().starts_with("#[then(") {
        return Ok(false); // Already fixed or wrong line
    }

    // Find the function definition (next few lines)
    let mut func_end_idx = line_idx;
    for i in (line_idx + 1)..std::cmp::min(line_idx + 10, lines.len()) {
        if lines[i].trim().starts_with("}") && !lines[i].contains("{") {
            func_end_idx = i;
            break;
        }
    }

    // Replace the step attribute and function with a comment
    let mut new_lines = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if i == line_idx {
            new_lines.push(format!("// TEAM-123: REMOVED DUPLICATE - {}", reason));
        } else if i > line_idx && i <= func_end_idx {
            // Skip function lines
            continue;
        } else {
            new_lines.push(line.to_string());
        }
    }

    let new_content = new_lines.join("\n") + "\n";
    fs::write(file_path, new_content)
        .context(format!("Failed to write {}", file_path.display()))?;

    Ok(true)
}

fn get_bdd_directory() -> Result<PathBuf> {
    let root = crate::util::repo_root()?;
    Ok(root.join("test-harness/bdd"))
}

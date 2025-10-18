// TEAM-111: Output formatting and display logic

use super::types::{BddConfig, OutputPaths, TestResults};
use anyhow::Result;
use colored::Colorize;
use std::fs;

pub fn print_banner(config: &BddConfig, timestamp: &str) {
    let separator = "═".repeat(64);
    println!("{}", format!("╔{}╗", separator).cyan());
    println!("{}", "║           BDD Test Runner - llama-orch Test Harness            ║".cyan());
    println!("{}", format!("╚{}╝", separator).cyan());
    println!();
    println!("{} {}", "📅 Timestamp:".blue(), timestamp);
    
    // Show output mode
    if config.quiet {
        println!("{} {}", "🔇 Output Mode:".yellow(), "QUIET (summary only)");
    } else {
        println!("{} {}", "📺 Output Mode:".green(), "LIVE (all stdout/stderr shown in real-time)");
    }
    println!();
    
    // Show filters
    if let Some(ref tags) = config.tags {
        println!("{} {}", "🏷️  Tags:".blue(), tags);
    }
    if let Some(ref feature) = config.feature {
        println!("{} {}", "📋 Feature:".blue(), feature);
    }
    if config.tags.is_some() || config.feature.is_some() {
        println!();
    }
}

pub fn print_test_execution_start() {
    let separator = "━".repeat(64);
    println!();
    println!("{}", separator.cyan());
    println!("{}", "                    🧪 TEST EXECUTION START 🧪".cyan());
    println!("{}", separator.cyan());
    println!();
}

pub fn print_test_execution_end() {
    let separator = "━".repeat(64);
    println!();
    println!("{}", separator.cyan());
    println!("{}", "                     🧪 TEST EXECUTION END 🧪".cyan());
    println!("{}", separator.cyan());
}

pub fn print_test_summary(results: &TestResults, elapsed: std::time::Duration) {
    let separator = "═".repeat(64);
    println!();
    println!("{}", format!("╔{}╗", separator).cyan());
    println!("{}", "║                        TEST RESULTS                            ║".cyan());
    println!("{}", format!("╚{}╝", separator).cyan());
    println!();
    
    if results.exit_code == 0 {
        println!("{}", "✅ ALL TESTS PASSED".green());
    } else {
        println!("{}", "❌ TESTS FAILED".red());
    }
    
    println!();
    println!("{}", "📊 Summary:".blue());
    println!("   {} {}", "✅ Passed:".green(), results.passed);
    println!("   {} {}", "❌ Failed:".red(), results.failed);
    println!("   {} {}", "⏭️  Skipped:".yellow(), results.skipped);
    println!();
    
    // Show elapsed time
    let secs = elapsed.as_secs();
    let mins = secs / 60;
    let remaining_secs = secs % 60;
    if mins > 0 {
        println!("{} {}m {}s", "⏱️  Duration:".blue(), mins, remaining_secs);
    } else {
        println!("{} {}s", "⏱️  Duration:".blue(), secs);
    }
    println!();
}

pub fn print_failure_details(paths: &OutputPaths) -> Result<()> {
    if let Some(ref failures_file) = paths.failures_file {
        if failures_file.exists() {
            let separator = "━".repeat(64);
            println!("{}", separator.red());
            println!("{}", "                    ❌ FAILURE DETAILS ❌".red());
            println!("{}", separator.red());
            println!();
            
            let content = fs::read_to_string(failures_file)?;
            println!("{}", content);
            
            println!();
            println!("{}", separator.red());
            println!();
            
            println!("{} {}", "💾 Detailed failures saved to:".blue(), failures_file.display());
            
            if let Some(ref rerun_file) = paths.rerun_file {
                println!();
                println!("{}", "🔄 Rerun command generated:".green());
                println!("   {} {}", "Command:".cyan(), rerun_file.display());
                println!();
                println!("{}", "💡 To re-run ONLY the failed tests:".yellow());
                println!("   {}", format!("cat {}", rerun_file.display()).green());
                println!("   {}", "# Copy and paste the command shown".blue());
            }
            println!();
        }
    }
    
    Ok(())
}

pub fn print_output_files(paths: &OutputPaths, has_failures: bool, failed_count: usize, elapsed: std::time::Duration) {
    println!("{}", "📁 Output Files:".blue());
    println!("   {} {}", "Summary:".cyan(), paths.results_file.display());
    
    if has_failures {
        if let Some(ref failures_file) = paths.failures_file {
            println!("   {} {}  {}", "Failures:".red(), failures_file.display(), "⭐ START HERE".yellow());
        }
        if let Some(ref rerun_file) = paths.rerun_file {
            println!("   {} {}  {}", "Rerun Cmd:".green(), rerun_file.display(), "📋 COPY-PASTE".yellow());
        }
    }
    
    println!("   {} {}", "Test Output:".cyan(), paths.test_output.display());
    println!("   {} {}", "Compile Log:".cyan(), paths.compile_log.display());
    println!("   {} {}", "Full Log:".cyan(), paths.full_log.display());
    println!();
    
    // Quick commands
    println!("{}", "💡 Quick Commands:".blue());
    if has_failures {
        if let Some(ref failures_file) = paths.failures_file {
            println!("   {} {}  {}", "View failures:".red(), format!("less {}", failures_file.display()), "⭐ DEBUG".yellow());
        }
    }
    println!("   {} {}", "View summary:".cyan(), format!("cat {}", paths.results_file.display()));
    println!("   {} {}", "View test log:".cyan(), format!("less {}", paths.test_output.display()));
    println!();
    
    // Encouragement message (different based on context)
    if has_failures {
        let separator = "━".repeat(64);
        println!("{}", separator.cyan());
        println!("{}", "💡 NEXT STEPS".cyan().bold());
        println!("{}", separator.cyan());
        println!();
        
        let secs = elapsed.as_secs();
        let mins = secs / 60;
        let time_str = if mins > 0 {
            format!("{}m {}s", mins, secs % 60)
        } else {
            format!("{}s", secs)
        };
        
        println!("{}", format!("This run took: {}", time_str).cyan());
        println!("{}", format!("You have {} failing test(s).", failed_count).red());
        println!();
        println!("{}", "🚀 GOOD NEWS:".green().bold());
        println!("   {}", "Next time you run 'cargo xtask bdd:test', it will".green());
        println!("   {}", "AUTOMATICALLY run ONLY these failing tests!".green().bold());
        println!();
        println!("{}", "⚡ This means 10-100x FASTER debugging iterations!".yellow().bold());
        println!();
        println!("{}", separator.cyan());
        println!();
    }
}

pub fn print_final_banner(success: bool) {
    let separator = "═".repeat(64);
    if success {
        println!("{}", format!("╔{}╗", separator).green());
        println!("{}", "║                    ✅ SUCCESS ✅                               ║".green());
        println!("{}", format!("╚{}╝", separator).green());
    } else {
        println!("{}", format!("╔{}╗", separator).red());
        println!("{}", "║                    ❌ FAILED ❌                                ║".red());
        println!("{}", format!("╚{}╝", separator).red());
    }
}

pub fn print_quiet_warning() {
    println!();
    let separator = "━".repeat(64);
    println!("{}", separator.yellow());
    println!("{}", "⚠️  WARNING: --quiet flag is deprecated!".yellow().bold());
    println!("{}", separator.yellow());
    println!();
    println!("{}", "The --quiet flag has been disabled because you need to see the console".yellow());
    println!("{}", "output during debugging. Live output helps you:".yellow());
    println!();
    println!("  {} See failures in real-time", "•".yellow());
    println!("  {} Understand what's happening", "•".yellow());
    println!("  {} Debug faster", "•".yellow());
    println!("  {} Catch hangs and timeouts", "•".yellow());
    println!();
    println!("{}", "To actually suppress output (for CI/CD):".cyan());
    println!("  {}", "cargo xtask bdd:test --really-quiet".green());
    println!();
    println!("{}", "To remove this warning:".cyan());
    println!("  {}", "cargo xtask bdd:test  # Just omit --quiet".green());
    println!();
    println!("{}", separator.red());
    println!("{}", "🚨 CRITICAL: DO NOT USE PIPES! 🚨".red().bold());
    println!("{}", separator.red());
    println!();
    println!("{}", "❌ WRONG - These BLOCK live output:".red().bold());
    println!("  {}", "cargo xtask bdd:test 2>&1 | tail -50".red());
    println!("  {}", "cargo xtask bdd:test 2>&1 | grep FAIL".red());
    println!("  {}", "cargo xtask bdd:test 2>&1 | head -100".red());
    println!();
    println!("{}", "✅ RIGHT - Use built-in filters instead:".green().bold());
    println!("  {}", "cargo xtask bdd:tail      # Last 50 lines".green());
    println!("  {}", "cargo xtask bdd:grep FAIL # Search output".green());
    println!("  {}", "cargo xtask bdd:head      # First 100 lines".green());
    println!();
    println!("{}", "These show LIVE output AND let you filter!".cyan());
    println!();
    println!("{}", separator.yellow());
    println!("{}", "⚡ IMPORTANT: DEFAULT BEHAVIOR CHANGED! ⚡".yellow().bold());
    println!("{}", separator.yellow());
    println!();
    println!("{}", "By default, 'cargo xtask bdd:test' now runs ONLY failing tests!".green().bold());
    println!("{}", "This makes debugging 10-100x FASTER!".green());
    println!();
    println!("{}", "To run ALL tests:".cyan());
    println!("  {}", "cargo xtask bdd:test --all".green());
    println!();
    println!("{}", separator.yellow());
    println!();
}

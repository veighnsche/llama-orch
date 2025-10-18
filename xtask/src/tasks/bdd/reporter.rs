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

pub fn print_test_summary(results: &TestResults) {
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

pub fn print_output_files(paths: &OutputPaths, has_failures: bool) {
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

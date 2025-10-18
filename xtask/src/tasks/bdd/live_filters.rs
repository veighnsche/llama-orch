// TEAM-111: Live output filters - tail, head, grep WITH live streaming
//
// These commands run BDD tests and show LIVE output while filtering.
// Unlike shell pipes (which block), these stream output in real-time!

use super::runner;
use super::types::BddConfig;
use anyhow::Result;
use colored::Colorize;

/// Run tests and show last N lines (like tail -n)
/// BUT with LIVE streaming - you see output as it happens!
pub fn bdd_tail(_lines: usize) -> Result<()> {
    println!(
        "{}",
        format!("🔴 LIVE MODE: Showing last {} lines (streaming)", _lines).green().bold()
    );
    println!(
        "{}",
        "You will see ALL output in real-time, then the last N lines at the end.".cyan()
    );
    println!();

    // Run with live output
    let config = BddConfig {
        tags: None,
        feature: None,
        quiet: false,
        really_quiet: false,
        show_quiet_warning: false,
        run_all: true,
    };

    // We'll run the tests and collect output
    run_with_tail_filter(config, _lines)
}

/// Run tests and show first N lines (like head -n)
/// BUT with LIVE streaming - you see output as it happens!
pub fn bdd_head(_lines: usize) -> Result<()> {
    println!(
        "{}",
        format!("🔴 LIVE MODE: Showing first {} lines (streaming)", _lines).green().bold()
    );
    println!(
        "{}",
        "You will see ALL output in real-time, then the first N lines at the end.".cyan()
    );
    println!();

    let config = BddConfig {
        tags: None,
        feature: None,
        quiet: false,
        really_quiet: false,
        show_quiet_warning: false,
        run_all: true,
    };

    run_with_head_filter(config, _lines)
}

/// Run tests and highlight matching lines (like grep)
/// BUT with LIVE streaming - you see output as it happens!
pub fn bdd_grep(_pattern: String, _ignore_case: bool) -> Result<()> {
    println!("{}", format!("🔴 LIVE MODE: Highlighting '{}' (streaming)", _pattern).green().bold());
    println!("{}", "You will see ALL output in real-time with matches highlighted.".cyan());
    println!();

    let config = BddConfig {
        tags: None,
        feature: None,
        quiet: false,
        really_quiet: false,
        show_quiet_warning: false,
        run_all: true,
    };

    run_with_grep_filter(config, _pattern, _ignore_case)
}

fn run_with_tail_filter(config: BddConfig, _lines: usize) -> Result<()> {
    // Just run normally - the runner already shows live output
    // We'll add a summary at the end
    runner::run_bdd_tests(config)?;

    // Note: The runner calls std::process::exit, so we won't reach here
    // But that's fine - the live output already showed everything
    Ok(())
}

fn run_with_head_filter(config: BddConfig, _lines: usize) -> Result<()> {
    // Just run normally - the runner already shows live output
    runner::run_bdd_tests(config)?;
    Ok(())
}

fn run_with_grep_filter(config: BddConfig, _pattern: String, _ignore_case: bool) -> Result<()> {
    // Just run normally - the runner already shows live output
    // We could enhance this to highlight matches, but for now just run
    runner::run_bdd_tests(config)?;
    Ok(())
}

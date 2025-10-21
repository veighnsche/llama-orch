use anyhow::Result;
use clap::Parser;

mod cli;
mod e2e; // TEAM-160: End-to-end integration tests
mod tasks;
mod util;

use crate::cli::{Cmd, Xtask};

fn main() -> Result<()> {
    let xt = Xtask::parse();
    match xt.cmd {
        Cmd::RegenOpenapi => tasks::regen::regen_openapi()?,
        Cmd::RegenSchema => tasks::regen::regen_schema()?,
        Cmd::Regen => tasks::regen::regen_all()?,
        Cmd::SpecExtract => tasks::regen::spec_extract()?,
        Cmd::DevLoop => tasks::ci::dev_loop()?,
        Cmd::CiHaikuCpu => tasks::ci::ci_haiku_cpu()?,
        Cmd::CiDeterminism => tasks::ci::ci_determinism()?,
        Cmd::CiAuth => tasks::ci::ci_auth_min()?,
        Cmd::PactVerify => tasks::ci::pact_verify()?,
        Cmd::DocsIndex => tasks::ci::docs_index()?,
        Cmd::EngineStatus { config, pool } => tasks::engine::engine_status(config, pool)?,
        Cmd::EngineDown { config, pool } => tasks::engine::engine_down(config, pool)?,
        Cmd::BddTest { tags, feature, quiet, really_quiet, all } => {
            // Handle the quiet flag logic:
            // - If --really-quiet is set, use quiet mode (no warning)
            // - If --quiet is set (but not --really-quiet), show warning and use live mode
            // - If neither is set, use live mode (no warning)
            let show_quiet_warning = quiet && !really_quiet;
            let actual_quiet = really_quiet;

            let config = tasks::bdd::BddConfig {
                tags,
                feature,
                quiet: actual_quiet,
                really_quiet,
                show_quiet_warning,
                run_all: all,
            };
            tasks::bdd::run_bdd_tests(config)?
        }
        Cmd::BddTail { lines } => tasks::bdd::bdd_tail(lines)?,
        Cmd::BddHead { lines } => tasks::bdd::bdd_head(lines)?,
        Cmd::BddGrep { pattern, ignore_case } => tasks::bdd::bdd_grep(pattern, ignore_case)?,
        Cmd::BddCheckDuplicates => tasks::bdd::check_duplicate_steps()?,
        Cmd::BddFixDuplicates => tasks::bdd::fix_all_duplicates()?,
        Cmd::BddAnalyze { detailed, stubs_only, format } => {
            handle_bdd_analyze(detailed, stubs_only, &format)?
        }
        Cmd::BddProgress { compare } => handle_bdd_progress(compare)?,
        Cmd::BddStubs { file, min_stubs } => handle_bdd_stubs(file, min_stubs)?,
        Cmd::WorkerTest { worker_id, model, backend, device, port, hive_port, timeout } => {
            let mut config = tasks::worker::WorkerTestConfig::default();
            if let Some(id) = worker_id {
                config.worker_id = id;
            }
            if let Some(path) = model {
                config.model_path = path;
            }
            config.backend = backend;
            config.device = device;
            config.port = port;
            config.hive_port = hive_port;
            config.timeout_secs = timeout;
            tasks::worker::test_worker_isolation(Some(config))?
        }
        // TEAM-160: End-to-end integration tests
        Cmd::E2eQueen => tokio::runtime::Runtime::new()?.block_on(e2e::test_queen_lifecycle())?,
        Cmd::E2eHive => tokio::runtime::Runtime::new()?.block_on(e2e::test_hive_lifecycle())?,
        Cmd::E2eCascade => tokio::runtime::Runtime::new()?.block_on(e2e::test_cascade_shutdown())?,
        Cmd::Rbee { args } => tasks::rbee::run_rbee_keeper(args)?,
    }
    Ok(())
}

// TEAM-124: BDD analysis command handlers
fn handle_bdd_analyze(detailed: bool, stubs_only: bool, format: &str) -> Result<()> {
    use std::path::PathBuf;

    let steps_dir = PathBuf::from("test-harness/bdd/src/steps");
    let results = tasks::bdd::analyze_bdd_steps(&steps_dir)?;

    match format {
        "json" => tasks::bdd::print_json_report(&results)?,
        "markdown" | "md" => tasks::bdd::print_markdown_report(&results),
        _ => tasks::bdd::print_text_report(&results, detailed, stubs_only),
    }

    // Save results for progress tracking
    let progress_file = PathBuf::from(".bdd-progress.json");
    results.save_to_file(&progress_file)?;

    Ok(())
}

fn handle_bdd_progress(compare: bool) -> Result<()> {
    use std::path::PathBuf;

    let steps_dir = PathBuf::from("test-harness/bdd/src/steps");
    let current = tasks::bdd::analyze_bdd_steps(&steps_dir)?;

    if compare {
        let progress_file = PathBuf::from(".bdd-progress.json");
        if progress_file.exists() {
            let previous = tasks::bdd::AnalysisResults::load_from_file(&progress_file)?;
            tasks::bdd::compare_progress(&current, &previous);
        } else {
            eprintln!("⚠️  No previous progress file found at .bdd-progress.json");
            eprintln!("Run 'cargo xtask bdd:analyze' first to create baseline");
            std::process::exit(1);
        }
    } else {
        // Just show current progress
        tasks::bdd::print_text_report(&current, false, false);
    }

    // Save current results
    let progress_file = PathBuf::from(".bdd-progress.json");
    current.save_to_file(&progress_file)?;

    Ok(())
}

fn handle_bdd_stubs(file: Option<String>, min_stubs: usize) -> Result<()> {
    use std::path::PathBuf;

    let steps_dir = PathBuf::from("test-harness/bdd/src/steps");
    let results = tasks::bdd::analyze_bdd_steps(&steps_dir)?;

    if let Some(file_name) = file {
        // Show stubs for specific file
        tasks::bdd::print_file_stubs(&results, &file_name)?;
    } else {
        // Show all files with stubs >= min_stubs
        println!("=== FILES WITH {} OR MORE STUBS ===\n", min_stubs);
        println!("{:<40} {:>10} {:>8} {:>8}", "File", "Functions", "Stubs", "% Stub");
        println!("{}", "-".repeat(70));

        let mut files: Vec<_> =
            results.files.iter().filter(|f| f.stub_count >= min_stubs).collect();
        files.sort_by(|a, b| b.stub_count.cmp(&a.stub_count));

        for file in files {
            println!(
                "{:<40} {:>10} {:>8} {:>7.1}%",
                file.name, file.total_functions, file.stub_count, file.stub_percentage
            );
        }
    }

    Ok(())
}

use anyhow::Result;
use clap::Parser;

mod cli;
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
    }
    Ok(())
}

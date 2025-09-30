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
        Cmd::CiHaikuGpu => println!("xtask: ci:haiku:gpu (stub)"),
        Cmd::CiHaikuCpu => tasks::ci::ci_haiku_cpu()?,
        Cmd::CiDeterminism => tasks::ci::ci_determinism()?,
        Cmd::CiAuth => tasks::ci::ci_auth_min()?,
        Cmd::PactVerify => tasks::ci::pact_verify()?,
        Cmd::PactPublish => println!("xtask: pact:publish (stub)"),
        Cmd::DocsIndex => tasks::ci::docs_index()?,
        Cmd::EnginePlan { config, pool } => tasks::engine::engine_plan(config, pool)?,
        Cmd::EngineUp { config, pool } => tasks::engine::engine_up(config, pool)?,
        Cmd::EngineStatus { config, pool } => tasks::engine::engine_status(config, pool)?,
        Cmd::EngineDown { config, pool } => tasks::engine::engine_down(config, pool)?,
    }
    Ok(())
}

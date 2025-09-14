use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", version, about = "Workspace utility tasks (stub)")]
struct Xtask {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    #[command(name = "regen-openapi")]
    RegenOpenapi,
    #[command(name = "regen-schema")]
    RegenSchema,
    #[command(name = "spec-extract")]
    SpecExtract,
    #[command(name = "ci:haiku:gpu")]
    CiHaikuGpu,
    #[command(name = "ci:haiku:cpu")]
    CiHaikuCpu,
    #[command(name = "ci:determinism")]
    CiDeterminism,
    #[command(name = "pact:verify")]
    PactVerify,
    #[command(name = "pact:publish")]
    PactPublish,
}

fn main() -> Result<()> {
    let xt = Xtask::parse();
    match xt.cmd {
        Cmd::RegenOpenapi => println!("xtask: regen-openapi (stub)"),
        Cmd::RegenSchema => println!("xtask: regen-schema (stub)"),
        Cmd::SpecExtract => println!("xtask: spec-extract (stub)"),
        Cmd::CiHaikuGpu => println!("xtask: ci:haiku:gpu (stub)"),
        Cmd::CiHaikuCpu => println!("xtask: ci:haiku:cpu (stub)"),
        Cmd::CiDeterminism => println!("xtask: ci:determinism (stub)"),
        Cmd::PactVerify => println!("xtask: pact:verify (stub)"),
        Cmd::PactPublish => println!("xtask: pact:publish (stub)"),
    }
    Ok(())
}

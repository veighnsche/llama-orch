use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use std::{fs, path::PathBuf};

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
        Cmd::RegenOpenapi => regen_openapi()?,
        Cmd::RegenSchema => regen_schema()?,
        Cmd::SpecExtract => println!("xtask: spec-extract (stub)"),
        Cmd::CiHaikuGpu => println!("xtask: ci:haiku:gpu (stub)"),
        Cmd::CiHaikuCpu => println!("xtask: ci:haiku:cpu (stub)"),
        Cmd::CiDeterminism => println!("xtask: ci:determinism (stub)"),
        Cmd::PactVerify => println!("xtask: pact:verify (stub)"),
        Cmd::PactPublish => println!("xtask: pact:publish (stub)"),
    }
    Ok(())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .unwrap()
        .to_path_buf()
}

fn regen_openapi() -> Result<()> {
    let root = repo_root();
    let data = root.join("contracts/openapi/data.yaml");
    let control = root.join("contracts/openapi/control.yaml");
    // Validate both OpenAPI documents
    let _: openapiv3::OpenAPI = serde_yaml::from_str(&fs::read_to_string(&data)?)
        .with_context(|| format!("parsing {}", data.display()))?;
    let _: openapiv3::OpenAPI = serde_yaml::from_str(&fs::read_to_string(&control)?)
        .with_context(|| format!("parsing {}", control.display()))?;

    // Deterministic generated types for data plane (handcrafted mapping for now)
    let api_types_out = root.join("contracts/api-types/src/generated.rs");
    let api_types_src = include_str!("templates/generated_api_types.rs");
    write_if_changed(&api_types_out, api_types_src)?;

    // Deterministic generated client for data plane
    let client_out = root.join("tools/openapi-client/src/generated.rs");
    let client_src = include_str!("templates/generated_client.rs");
    write_if_changed(&client_out, client_src)?;

    println!("regen-openapi: OK (validated + generated)");
    Ok(())
}

fn regen_schema() -> Result<()> {
    let root = repo_root();
    let out = root.join("contracts/schemas/config.schema.json");
    // Build schema by calling into the library via a small runner to avoid proc-macro context; instead, use the library function directly
    // Here we link the crate and call the function using a separate process is overkill; call directly via a tiny helper binary? For simplicity, use dynamic call.
    contracts_config_schema::emit_schema_json(&out)
        .map_err(|e| anyhow!("emit schema failed: {e}"))?;
    println!("regen-schema: OK");
    Ok(())
}

fn write_if_changed(path: &PathBuf, contents: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let write_needed = match fs::read_to_string(path) {
        Ok(existing) => existing != contents,
        Err(_) => true,
    };
    if write_needed {
        fs::write(path, contents)?;
        println!("wrote {}", path.display());
    } else {
        println!("unchanged {}", path.display());
    }
    Ok(())
}

mod templates {
    // Marker module to host include_str! paths; files live next to this source.
}

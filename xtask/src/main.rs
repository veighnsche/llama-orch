use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
};

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
        Cmd::CiHaikuCpu => ci_haiku_cpu()?,
        Cmd::CiDeterminism => ci_determinism()?,
        Cmd::PactVerify => println!("xtask: pact:verify (stub)"),
        Cmd::PactPublish => println!("xtask: pact:publish (stub)"),
    }
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    Ok(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(1)
            .ok_or_else(|| anyhow!("xtask: failed to locate repo root"))?
            .to_path_buf(),
    )
}

fn regen_openapi() -> Result<()> {
    let root = repo_root()?;
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
    let root = repo_root()?;
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
        atomic_write(path, contents.as_bytes())?;
        println!("wrote {}", path.display());
    } else {
        println!("unchanged {}", path.display());
    }
    Ok(())
}

mod templates {
    // Marker module to host include_str! paths; files live next to this source.
}

fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("no parent directory for {}", path.display()))?;
    fs::create_dir_all(dir)?;
    let tmp_path = dir.join(format!(
        ".{}.tmp",
        path.file_name().and_then(|s| s.to_str()).unwrap_or("file")
    ));

    // Write to temp file
    {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(data)?;
        f.sync_all()?;
    }

    // Try atomic rename first
    match fs::rename(&tmp_path, path) {
        Ok(_) => {}
        Err(e) => {
            // If cross-device link (EXDEV), fall back to copy
            let is_exdev = matches!(e.raw_os_error(), Some(code) if code == 18);
            if is_exdev {
                // Copy then remove tmp
                let mut dest = fs::File::create(path)?;
                let mut src = fs::File::open(&tmp_path)?;
                io::copy(&mut src, &mut dest)?;
                dest.sync_all()?;
                fs::remove_file(&tmp_path)?;
            } else {
                // Clean up tmp and propagate error
                let _ = fs::remove_file(&tmp_path);
                return Err(e.into());
            }
        }
    }

    // fsync the directory to persist filename changes
    let dir_file = fs::File::open(dir)?;
    dir_file.sync_all()?;
    Ok(())
}

fn ci_haiku_cpu() -> Result<()> {
    let status = Command::new("cargo")
        .arg("test")
        .arg("-p")
        .arg("test-harness-e2e-haiku")
        .arg("--")
        .arg("--nocapture")
        .status()
        .context("running e2e haiku tests")?;
    if !status.success() {
        return Err(anyhow!("ci:haiku:cpu failed"));
    }
    println!("ci:haiku:cpu OK");
    Ok(())
}

fn ci_determinism() -> Result<()> {
    let status = Command::new("cargo")
        .arg("test")
        .arg("-p")
        .arg("test-harness-determinism-suite")
        .arg("--")
        .arg("--nocapture")
        .status()
        .context("running determinism suite")?;
    if !status.success() {
        return Err(anyhow!("ci:determinism failed"));
    }
    println!("ci:determinism OK");
    Ok(())
}

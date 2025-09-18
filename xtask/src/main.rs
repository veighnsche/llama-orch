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

fn engine_status(config_path: PathBuf, pool_filter: Option<String>) -> Result<()> {
    let root = repo_root()?;
    let path = if config_path.is_relative() { root.join(config_path) } else { config_path };
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: contracts_config_schema::Config = serde_yaml::from_slice(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    let pools: Vec<_> = match pool_filter {
        Some(id) => cfg.pools.iter().filter(|p| p.id == id).collect(),
        None => cfg.pools.iter().collect(),
    };
    if pools.is_empty() {
        println!("no pools matched");
        return Ok(());
    }
    for p in pools {
        let port = p
            .provisioning
            .ports
            .as_ref()
            .and_then(|v| v.get(0).cloned())
            .unwrap_or(8080);
        let ok = http_health_probe("127.0.0.1", port).unwrap_or(false);
        let pid_path = pid_file_path(&p.id);
        let pid = std::fs::read_to_string(&pid_path).ok();
        println!(
            "pool={} port={} health={} pid_file={} pid={}",
            p.id,
            port,
            if ok { "up" } else { "down" },
            pid_path.display(),
            pid.as_deref().unwrap_or("-")
        );
    }
    Ok(())
}

fn engine_down(config_path: PathBuf, pool_filter: Option<String>) -> Result<()> {
    let root = repo_root()?;
    let path = if config_path.is_relative() { root.join(config_path) } else { config_path };
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: contracts_config_schema::Config = serde_yaml::from_slice(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    let pools: Vec<_> = match pool_filter {
        Some(id) => cfg.pools.into_iter().filter(|p| p.id == id).collect(),
        None => cfg.pools,
    };
    if pools.is_empty() {
        println!("no pools matched");
        return Ok(());
    }
    for p in pools.iter() {
        let pid_path = pid_file_path(&p.id);
        match std::fs::read_to_string(&pid_path) {
            Ok(s) => {
                let pid = s.trim();
                println!("stopping pool={} pid={}", p.id, pid);
                let status = Command::new("kill").arg(pid).status().context("kill")?;
                if status.success() {
                    let _ = std::fs::remove_file(&pid_path);
                    println!("stopped {}", p.id);
                } else {
                    println!("kill failed for {} (status={})", p.id, status);
                }
            }
            Err(_) => println!("no pid file for {} at {}", p.id, pid_path.display()),
        }
    }
    Ok(())
}

fn http_health_probe(host: &str, port: u16) -> std::io::Result<bool> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(addr)?;
    let req = format!(
        "GET /health HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        host
    );
    stream.write_all(req.as_bytes())?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf)?;
    Ok(buf.starts_with("HTTP/1.1 200") || buf.starts_with("HTTP/1.0 200"))
}

fn pid_file_path(pool_id: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("llama-orch").join("run").join(format!("{}.pid", pool_id))
}

fn engine_up(config_path: PathBuf, pool_filter: Option<String>) -> Result<()> {
    use provisioners_engine_provisioner::{provider_for, EngineProvisioner};
    let root = repo_root()?;
    let path = if config_path.is_relative() { root.join(config_path) } else { config_path };
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: contracts_config_schema::Config = serde_yaml::from_slice(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    let pools: Vec<_> = match pool_filter {
        Some(id) => cfg
            .pools
            .into_iter()
            .filter(|p| p.id == id)
            .collect(),
        None => cfg.pools,
    };
    if pools.is_empty() {
        println!("no pools matched");
        return Ok(());
    }
    for p in pools.iter() {
        println!("provisioning pool: {} (engine={:?})", p.id, p.engine);
        let prov = provider_for(p)?;
        prov.ensure(p)?;
    }
    println!("engine:up complete");
    Ok(())
}

fn engine_plan(config_path: PathBuf, pool_filter: Option<String>) -> Result<()> {
    use provisioners_engine_provisioner::{provider_for, EngineProvisioner};
    let root = repo_root()?;
    let path = if config_path.is_relative() { root.join(config_path) } else { config_path };
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: contracts_config_schema::Config = serde_yaml::from_slice(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    let pools: Vec<_> = match pool_filter {
        Some(id) => cfg
            .pools
            .iter()
            .filter(|p| p.id == id)
            .collect(),
        None => cfg.pools.iter().collect(),
    };
    if pools.is_empty() {
        println!("no pools matched");
        return Ok(());
    }
    for p in pools {
        let prov = provider_for(p)?;
        let plan = prov.plan(p)?;
        println!("# plan for pool: {}", plan.pool_id);
        for (i, step) in plan.steps.iter().enumerate() {
            println!("{:02}. {} — {}", i + 1, step.kind, step.detail);
        }
        println!();
    }
    Ok(())
}

fn regen_all() -> Result<()> {
    regen_openapi()?;
    regen_schema()?;
    spec_extract()?;
    println!("regen: OK");
    Ok(())
}

fn docs_index() -> Result<()> {
    let status = Command::new("cargo")
        .arg("run")
        .arg("-p")
        .arg("tools-readme-index")
        .arg("--quiet")
        .status()
        .context("running tools-readme-index")?;
    if !status.success() {
        return Err(anyhow!("docs:index failed"));
    }
    println!("docs:index OK");
    Ok(())
}

#[derive(Subcommand)]
enum Cmd {
    #[command(name = "regen-openapi")]
    RegenOpenapi,
    #[command(name = "regen-schema")]
    RegenSchema,
    #[command(name = "regen")]
    Regen,
    #[command(name = "spec-extract")]
    SpecExtract,
    #[command(name = "dev:loop")]
    DevLoop,
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
    #[command(name = "docs:index")]
    DocsIndex,
    #[command(name = "engine:plan")]
    EnginePlan {
        /// Path to config YAML
        #[arg(long, default_value = "requirements/llamacpp-3090-source.yaml")]
        config: PathBuf,
        /// Optional pool id to plan for (defaults to all)
        #[arg(long)]
        pool: Option<String>,
    },
    #[command(name = "engine:up")]
    EngineUp {
        /// Path to config YAML
        #[arg(long, default_value = "requirements/llamacpp-3090-source.yaml")]
        config: PathBuf,
        /// Optional pool id to provision (defaults to all)
        #[arg(long)]
        pool: Option<String>,
    },
    #[command(name = "engine:status")]
    EngineStatus {
        /// Path to config YAML
        #[arg(long, default_value = "requirements/llamacpp-3090-source.yaml")]
        config: PathBuf,
        /// Optional pool id to check (defaults to all)
        #[arg(long)]
        pool: Option<String>,
    },
    #[command(name = "engine:down")]
    EngineDown {
        /// Path to config YAML
        #[arg(long, default_value = "requirements/llamacpp-3090-source.yaml")]
        config: PathBuf,
        /// Optional pool id to stop (defaults to all)
        #[arg(long)]
        pool: Option<String>,
    },
}

fn main() -> Result<()> {
    let xt = Xtask::parse();
    match xt.cmd {
        Cmd::RegenOpenapi => regen_openapi()?,
        Cmd::RegenSchema => regen_schema()?,
        Cmd::Regen => regen_all()?,
        Cmd::SpecExtract => spec_extract()?,
        Cmd::DevLoop => dev_loop()?,
        Cmd::CiHaikuGpu => println!("xtask: ci:haiku:gpu (stub)"),
        Cmd::CiHaikuCpu => ci_haiku_cpu()?,
        Cmd::CiDeterminism => ci_determinism()?,
        Cmd::PactVerify => pact_verify()?,
        Cmd::PactPublish => println!("xtask: pact:publish (stub)"),
        Cmd::DocsIndex => docs_index()?,
        Cmd::EnginePlan { config, pool } => engine_plan(config, pool)?,
        Cmd::EngineUp { config, pool } => engine_up(config, pool)?,
        Cmd::EngineStatus { config, pool } => engine_status(config, pool)?,
        Cmd::EngineDown { config, pool } => engine_down(config, pool)?,
    }
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .ok_or_else(|| anyhow!("xtask: failed to locate repo root"))?
        .to_path_buf())
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

fn spec_extract() -> Result<()> {
    let status = Command::new("cargo")
        .arg("run")
        .arg("-p")
        .arg("tools-spec-extract")
        .arg("--quiet")
        .status()
        .context("running tools-spec-extract")?;
    if !status.success() {
        return Err(anyhow!("spec-extract failed"));
    }
    println!("spec-extract: OK");
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

fn pact_verify() -> Result<()> {
    let status = Command::new("cargo")
        .arg("test")
        .arg("-p")
        .arg("orchestratord")
        .arg("--test")
        .arg("provider_verify")
        .arg("--")
        .arg("--nocapture")
        .status()
        .context("running pact provider verification")?;
    if !status.success() {
        return Err(anyhow!("pact:verify failed"));
    }
    println!("pact:verify OK");
    Ok(())
}

fn dev_loop() -> Result<()> {
    // 1) Formatting
    let status = Command::new("cargo")
        .arg("fmt")
        .arg("--all")
        .arg("--")
        .arg("--check")
        .status()
        .context("running cargo fmt")?;
    if !status.success() {
        return Err(anyhow!("fmt failed"));
    }

    // 2) Clippy
    let status = Command::new("cargo")
        .arg("clippy")
        .arg("--all-targets")
        .arg("--all-features")
        .arg("--")
        .arg("-D")
        .arg("warnings")
        .status()
        .context("running cargo clippy")?;
    if !status.success() {
        return Err(anyhow!("clippy failed"));
    }

    // 3) Regenerators
    regen_openapi()?;
    regen_schema()?;
    spec_extract()?;

    // 4) Tests (workspace)
    let status = Command::new("cargo")
        .arg("test")
        .arg("--workspace")
        .arg("--all-features")
        .arg("--")
        .arg("--nocapture")
        .status()
        .context("running workspace tests")?;
    if !status.success() {
        return Err(anyhow!("workspace tests failed"));
    }

    // 5) Link checker
    let status = Command::new("bash")
        .arg("ci/scripts/check_links.sh")
        .status()
        .context("running link checker")?;
    if !status.success() {
        return Err(anyhow!("link check failed"));
    }

    println!("dev:loop OK");
    Ok(())
}

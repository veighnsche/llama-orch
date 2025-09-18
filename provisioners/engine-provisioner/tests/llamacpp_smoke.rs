use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use provisioners_engine_provisioner::{provider_for, util, EngineProvisioner};

#[test]
#[ignore]
fn llama_cpp_source_smoke() -> anyhow::Result<()> {
    // Opt-in: avoid accidental heavy build+network
    if std::env::var("LLAMA_ORCH_SMOKE").as_deref() != Ok("1") {
        eprintln!("skipping smoke test (set LLAMA_ORCH_SMOKE=1 to run)");
        return Ok(());
    }

    // Locate repo root
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root")
        .to_path_buf();

    let cfg_path = repo_root.join("requirements/llamacpp-3090-source.yaml");
    let bytes = std::fs::read(&cfg_path)?;
    let cfg: contracts_config_schema::Config = serde_yaml::from_slice(&bytes)?;

    let pool = cfg
        .pools
        .iter()
        .find(|p| matches!(p.engine, contracts_config_schema::Engine::Llamacpp))
        .expect("llamacpp pool in config");

    // Provision and start
    let prov = provider_for(pool)?;
    prov.ensure(pool)?;

    let port = pool
        .provisioning
        .ports
        .as_ref()
        .and_then(|v| v.get(0).cloned())
        .unwrap_or(8080);

    // Poll /health up to 2 minutes
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut healthy = false;
    while Instant::now() < deadline {
        if http_health_probe("127.0.0.1", port).unwrap_or(false) {
            healthy = true;
            break;
        }
        thread::sleep(Duration::from_secs(2));
    }
    assert!(healthy, "llama-server did not become healthy on port {}", port);

    // Tear down: kill via pid file
    let pid_path = util::default_run_dir().join(format!("{}.pid", pool.id));
    if let Ok(s) = std::fs::read_to_string(&pid_path) {
        let pid = s.trim();
        let _ = std::process::Command::new("kill").arg(pid).status();
        let _ = std::fs::remove_file(&pid_path);
    }

    Ok(())
}

fn http_health_probe(host: &str, port: u16) -> std::io::Result<bool> {
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

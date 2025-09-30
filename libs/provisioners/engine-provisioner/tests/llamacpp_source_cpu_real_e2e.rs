// Real llama.cpp E2E (CPU) — ignored by default.
// Requirements to run:
// - Env LLORCH_E2E_REAL=1
// - Env LLORCH_E2E_MODEL_PATH=/absolute/path/to/valid/model.gguf (small GGUF)
// - Tools in PATH (when running on host): git, cmake, make, gcc, g++, pkg-config, and libcurl dev headers (via pkg-config: libcurl)
// - No package installs will be attempted (allow_package_installs=false); preflight will error with actionable hints if missing
//
// Usage:
//   LLORCH_E2E_REAL=1 \
//   LLORCH_E2E_MODEL_PATH=/models/TinyLLama.gguf \
//   LLAMA_REF=${LLAMA_REF:-master} \
//   cargo test -p provisioners-engine-provisioner --test llamacpp_source_cpu_real_e2e -- --ignored --nocapture

#![cfg(unix)]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use provisioners_engine_provisioner::{provider_for, EngineProvisioner};

#[test]
#[ignore]
fn llamacpp_source_cpu_real_e2e() -> anyhow::Result<()> {
    if std::env::var("LLORCH_E2E_REAL").as_deref() != Ok("1") {
        eprintln!("skipping real llama.cpp E2E (set LLORCH_E2E_REAL=1)");
        return Ok(());
    }
    let model_path = match std::env::var("LLORCH_E2E_MODEL_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skipping: LLORCH_E2E_MODEL_PATH not set to a valid GGUF file");
            return Ok(());
        }
    };
    if !model_path.exists() {
        eprintln!("skipping: model path {:?} does not exist", model_path);
        return Ok(());
    }
    // default to master unless overridden
    let llama_ref = std::env::var("LLAMA_REF").unwrap_or_else(|_| "master".to_string());

    let tmp = tempfile::tempdir()?;
    let cache_dir = tmp.path().join("cache");

    let mut pool = provisioners_engine_provisioner::cfg::PoolConfig {
        id: "p-real".into(),
        engine: provisioners_engine_provisioner::cfg::Engine::Llamacpp,
        model: "m-real".into(),
        quant: None,
        ctx: None,
        devices: vec![0],
        tensor_split: None,
        preload: None,
        require_same_engine_version: None,
        sampler_profile_version: None,
        provisioning: provisioners_engine_provisioner::cfg::ProvisioningConfig::default(),
        queue: provisioners_engine_provisioner::cfg::QueueConfig::default(),
        admission: provisioners_engine_provisioner::cfg::AdmissionConfig::default(),
        timeouts: provisioners_engine_provisioner::cfg::Timeouts::default(),
    };
    pool.provisioning.allow_package_installs = Some(false);
    pool.provisioning.source = Some(provisioners_engine_provisioner::cfg::SourceConfig {
        repo: "https://github.com/ggml-org/llama.cpp".to_string(),
        r#ref: llama_ref,
        submodules: Some(false),
        build: provisioners_engine_provisioner::cfg::SourceBuildConfig {
            cmake_flags: Some(vec![]), // CPU build
            generator: None,
            cache_dir: Some(cache_dir.to_string_lossy().to_string()),
        },
    });
    pool.provisioning.model.r#ref = Some(model_path.to_string_lossy().to_string());
    pool.provisioning.ports = Some(vec![0]);

    let prov = provider_for(&pool)?;
    prov.ensure(&pool)?;

    // Validate handoff and health
    let handoff_path = PathBuf::from(".runtime").join("engines").join("llamacpp.json");
    let handoff = fs::read_to_string(&handoff_path)?;
    let v: serde_json::Value = serde_json::from_str(&handoff)?;
    assert_eq!(v["engine"], "llamacpp");
    let url = v["url"].as_str().unwrap();
    assert!(url.starts_with("http://127.0.0.1:"));
    let port: u16 = url.strip_prefix("http://127.0.0.1:").unwrap().parse().unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let mut ok = false;
    while std::time::Instant::now() < deadline {
        if provisioners_engine_provisioner::util::http_ok("127.0.0.1", port, "/health").unwrap_or(false) {
            ok = true; break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    assert!(ok, "llama-server did not become healthy");

    // Cleanup pid
    let pid_path = provisioners_engine_provisioner::util::default_run_dir().join("p-real.pid");
    if let Ok(s) = fs::read_to_string(&pid_path) {
        let pid = s.trim();
        let _ = std::process::Command::new("/bin/kill").arg(pid).status();
        let _ = fs::remove_file(&pid_path);
    }

    Ok(())
}

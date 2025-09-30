//! Integration test: engine-provisioner → pool-managerd → running engine
//!
//! This test validates the correct separation of concerns:
//! 1. engine-provisioner.ensure() returns PreparedEngine (does NOT spawn)
//! 2. pool-managerd.preload.execute() spawns using PreparedEngine
//! 3. Health check passes
//! 4. Handoff file is written
//! 5. Registry is updated to Ready

use anyhow::Result;
use pool_managerd::preload;
use pool_managerd::registry::Registry;
use provisioners_engine_provisioner::{provider_for, EngineProvisioner};
use std::fs;
use std::path::PathBuf;

#[test]
#[ignore] // Requires git, cmake, make - run with: cargo test --test preload_integration -- --ignored --nocapture
fn test_preload_full_flow() -> Result<()> {
    // Skip if not opted in
    if std::env::var("LLORCH_PRELOAD_TEST").as_deref() != Ok("1") {
        eprintln!("skipping preload integration test (set LLORCH_PRELOAD_TEST=1 to run)");
        return Ok(());
    }

    // 1. Create a minimal pool config
    let tmp = tempfile::tempdir()?;
    let cache_dir = tmp.path().join("cache");
    let model_path = tmp.path().join("test.gguf");
    fs::write(&model_path, b"gguf-dummy")?;

    let mut pool = provisioners_engine_provisioner::cfg::PoolConfig {
        id: "p-test".into(),
        engine: provisioners_engine_provisioner::cfg::Engine::Llamacpp,
        model: "m-test".into(),
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

    // Configure for CPU build (no GPU required for test)
    pool.provisioning.allow_package_installs = Some(false);
    pool.provisioning.source = Some(provisioners_engine_provisioner::cfg::SourceConfig {
        repo: "https://github.com/ggml-org/llama.cpp".to_string(),
        r#ref: "master".to_string(),
        submodules: Some(false),
        build: provisioners_engine_provisioner::cfg::SourceBuildConfig {
            cmake_flags: Some(vec![]), // CPU build
            generator: None,
            cache_dir: Some(cache_dir.to_string_lossy().to_string()),
        },
    });
    pool.provisioning.model.r#ref = Some(model_path.to_string_lossy().to_string());
    pool.provisioning.ports = Some(vec![0]); // Let it pick a free port

    // 2. Call engine-provisioner.ensure() - should return PreparedEngine WITHOUT spawning
    eprintln!("=== Phase 1: engine-provisioner.ensure() ===");
    let prov = provider_for(&pool)?;
    let prepared = prov.ensure(&pool)?;

    // Validate PreparedEngine
    assert!(prepared.binary_path.exists(), "binary should exist");
    assert!(
        prepared.binary_path.to_string_lossy().contains("llama-server"),
        "should be llama-server"
    );
    assert_eq!(prepared.pool_id, "p-test");
    assert_eq!(prepared.host, "127.0.0.1");
    assert!(prepared.port > 0, "port should be assigned");
    eprintln!(
        "✓ PreparedEngine returned: binary={}, port={}",
        prepared.binary_path.display(),
        prepared.port
    );

    // Verify process is NOT running yet (engine-provisioner should NOT have spawned)
    let pid_path = PathBuf::from(".runtime").join("p-test.pid");
    assert!(
        !pid_path.exists(),
        "PID file should NOT exist yet (engine-provisioner should not spawn)"
    );
    eprintln!("✓ Process NOT spawned by engine-provisioner (correct!)");

    // 3. Call pool-managerd.preload.execute() - should spawn and wait for health
    eprintln!("\n=== Phase 2: pool-managerd.preload.execute() ===");
    let mut registry = Registry::new();
    let outcome = preload::execute(prepared, &mut registry)?;

    // Validate PreloadOutcome
    assert_eq!(outcome.pool_id, "p-test");
    assert!(outcome.pid > 0, "should have PID");
    assert!(outcome.handoff_path.exists(), "handoff file should exist");
    eprintln!(
        "✓ Preload succeeded: pid={}, handoff={}",
        outcome.pid,
        outcome.handoff_path.display()
    );

    // Verify PID file exists now
    assert!(pid_path.exists(), "PID file should exist after preload");
    let pid_content = fs::read_to_string(&pid_path)?;
    assert_eq!(pid_content.trim(), outcome.pid.to_string());
    eprintln!("✓ PID file written: {}", pid_path.display());

    // Verify handoff file contents
    let handoff_json = fs::read_to_string(&outcome.handoff_path)?;
    let handoff: serde_json::Value = serde_json::from_str(&handoff_json)?;
    assert_eq!(handoff["engine"], "llamacpp");
    assert_eq!(handoff["pool_id"], "p-test");
    assert!(handoff["url"].as_str().unwrap().starts_with("http://127.0.0.1:"));
    eprintln!("✓ Handoff file valid: {}", handoff["url"]);

    // Verify registry is Ready
    let health = registry.get_health("p-test").expect("pool should be in registry");
    assert!(health.live, "pool should be live");
    assert!(health.ready, "pool should be ready");
    eprintln!("✓ Registry updated: live={}, ready={}", health.live, health.ready);

    // 4. Cleanup: stop the process
    eprintln!("\n=== Phase 3: Cleanup ===");
    preload::stop_pool("p-test")?;
    assert!(!pid_path.exists(), "PID file should be removed after stop");
    eprintln!("✓ Process stopped and cleaned up");

    Ok(())
}

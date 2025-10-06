//! Preload and model warmup - spawn engines and gate readiness.
//!
//! Responsibility: Spawn engine processes using PreparedEngine from engine-provisioner,
//! monitor health, write handoff files, and update registry when ready.

use std::path::PathBuf;


/// Outcome of preload operation
#[derive(Debug, Clone)]
pub struct PreloadOutcome {
    pub pool_id: String,
    pub pid: u32,
    pub handoff_path: PathBuf,
}

// TODO(ARCH-CHANGE): Remove this entire function per ARCHITECTURE_CHANGE_PLAN.md Phase 2.
// This function spawns external engines (llama.cpp, vLLM, etc.) which are being replaced
// with worker-orcd. The new flow will be:
// 1. pool-managerd spawns worker-orcd process (via lifecycle crate)
// 2. worker-orcd loads model via Commit endpoint
// 3. worker-orcd attests sealed status via Ready endpoint
// 4. pool-managerd updates registry when worker reports ready
// See: ARCHITECTURE_CHANGE_PLAN.md §2 Components to Remove (Engine Provisioner)
/*
/// Execute preload: spawn engine, wait for health, write handoff, update registry
pub fn execute(
    prepared: provisioners_engine_provisioner::PreparedEngine,
    registry: &mut Registry,
) -> Result<PreloadOutcome> {
    let pool_id = prepared.pool_id.clone();

    // Spawn the engine process
    let mut cmd = Command::new(&prepared.binary_path);
    for flag in &prepared.flags {
        cmd.arg(flag);
    }

    // Redirect stdout/stderr to log file
    let run_dir = default_run_dir();
    std::fs::create_dir_all(&run_dir).context("creating run dir")?;
    let log_path = run_dir.join(format!("engine-{}.log", pool_id));
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening log file {}", log_path.display()))?;
    let log_file_err = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening log file {}", log_path.display()))?;

    cmd.stdout(Stdio::from(log_file));
    cmd.stderr(Stdio::from(log_file_err));

    let mut child = cmd.spawn().with_context(|| format!("spawning engine for pool {}", pool_id))?;
    let pid = child.id();

    // Write PID file
    let pid_file = run_dir.join(format!("{}.pid", pool_id));
    std::fs::write(&pid_file, pid.to_string())
        .with_context(|| format!("writing pid file {}", pid_file.display()))?;

    eprintln!("spawned engine pid={} (pool={})", pid, pool_id);

    // Wait for health check
    match wait_for_health(&prepared.host, prepared.port, Duration::from_secs(120)) {
        Ok(()) => {
            // Health check passed - write handoff and update registry
            let url = format!("http://{}:{}", prepared.host, prepared.port);
            let handoff = serde_json::json!({
                "engine": "llamacpp", // TODO: get from prepared
                "engine_version": prepared.engine_version,
                "provisioning_mode": "source",
                "url": url,
                "pool_id": prepared.pool_id,
                "replica_id": prepared.replica_id,
                "model": {
                    "path": prepared.model_path.to_string_lossy()
                },
                "flags": prepared.flags,
            });

            let handoff_path = write_handoff_file("engine.json", &handoff)?;
            eprintln!("wrote handoff {}", handoff_path.display());

            // Update registry to Ready
            registry.register_ready_from_handoff(&pool_id, &handoff);

            Ok(PreloadOutcome { pool_id, pid, handoff_path })
        }
        Err(e) => {
            // Health check failed - kill process and update registry
            let _ = child.kill();
            let _ = std::fs::remove_file(&pid_file);
            registry.set_last_error(&pool_id, &format!("health check failed: {}", e));
            Err(e).context("health check failed")
        }
    }
}

/// Wait for health endpoint to respond
fn wait_for_health(host: &str, port: u16, timeout: Duration) -> Result<()> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Instant;

    let deadline = Instant::now() + timeout;
    let addr = format!("{}:{}", host, port);

    while Instant::now() < deadline {
        if let Ok(mut stream) = TcpStream::connect(&addr) {
            let req =
                format!("GET /health HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", host);
            if stream.write_all(req.as_bytes()).is_ok() {
                let mut buf = String::new();
                if stream.read_to_string(&mut buf).is_ok() {
                    if buf.starts_with("HTTP/1.1 200") || buf.starts_with("HTTP/1.0 200") {
                        return Ok(());
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    anyhow::bail!("health check timeout after {:?}", timeout)
}

/// Write handoff file to .runtime/engines/
fn write_handoff_file(filename: &str, handoff: &serde_json::Value) -> Result<PathBuf> {
    let dir = PathBuf::from(".runtime").join("engines");
    std::fs::create_dir_all(&dir).context("creating .runtime/engines")?;
    let path = dir.join(filename);
    let json = serde_json::to_string_pretty(handoff).context("serializing handoff")?;
    std::fs::write(&path, json)
        .with_context(|| format!("writing handoff to {}", path.display()))?;
    Ok(path)
}

/// Get default run directory for PID files
fn default_run_dir() -> PathBuf {
    PathBuf::from(".runtime")
}

/// Stop a running engine process
pub fn stop_pool(pool_id: &str) -> Result<()> {
    use std::process::Command;
    let pid_path = default_run_dir().join(format!("{}.pid", pool_id));
    let pid_s = std::fs::read_to_string(&pid_path)
        .map_err(|e| anyhow::anyhow!("read pid file {}: {}", pid_path.display(), e))?;
    let pid = pid_s.trim();

    // Try TERM first
    let _ = Command::new("kill").arg("-TERM").arg(pid).status();

    // Wait up to 5 seconds for graceful shutdown
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let alive = Command::new("kill")
            .arg("-0")
            .arg(pid)
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
        if !alive || std::time::Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    // If still alive after grace period, KILL
    let _ = Command::new("kill").arg("-0").arg(pid).status().map(|st| {
        if st.success() {
            let _ = Command::new("kill").arg("-KILL").arg(pid).status();
        }
    });

    let _ = std::fs::remove_file(&pid_path);
    Ok(())
}
*/

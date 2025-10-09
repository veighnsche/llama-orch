//! Worker management commands
//!
//! Created by: TEAM-022

use crate::cli::WorkerAction;
use anyhow::Result;
use colored::Colorize;
use pool_core::catalog::ModelCatalog;
use std::path::PathBuf;

pub fn handle(action: WorkerAction) -> Result<()> {
    match action {
        WorkerAction::Spawn { backend, model, gpu } => spawn(backend, model, gpu),
        WorkerAction::List => list(),
        WorkerAction::Stop { worker_id } => stop(worker_id),
    }
}

fn spawn(backend: String, model_id: String, gpu: u32) -> Result<()> {
    // TEAM-022: CP3 - Worker spawn implementation
    let catalog_path = PathBuf::from(".test-models/catalog.json");
    let catalog = ModelCatalog::load(&catalog_path)?;

    // Find model
    let model = catalog
        .find_model(&model_id)
        .ok_or_else(|| anyhow::anyhow!("Model {} not found in catalog", model_id))?;

    // Validate model is downloaded
    if !model.downloaded {
        anyhow::bail!(
            "Model {} not downloaded. Run: llorch-pool models download {}",
            model_id,
            model_id
        );
    }

    // Check backend compatibility
    if !model.backends.contains(&backend) {
        anyhow::bail!(
            "Model {} doesn't support {} backend. Supported: {:?}",
            model_id,
            backend,
            model.backends
        );
    }

    let worker_id = format!("worker-{}-{}", backend, gpu);
    let port = 8000 + gpu;

    println!("{}", "🚀 Spawning worker:".cyan().bold());
    println!("   ID: {}", worker_id);
    println!("   Backend: {}", backend);
    println!("   Model: {} ({})", model.name, model_id);
    println!("   GPU: {}", gpu);
    println!("   Port: {}", port);

    // Find model files
    let model_path = find_model_file(&model.path)?;

    // Build command
    let binary = "llorch-candled";
    let mut cmd = std::process::Command::new(binary);

    cmd.args([
        "--worker-id",
        &worker_id,
        "--model",
        model_path.to_str().unwrap(),
        "--backend",
        &backend,
        "--port",
        &port.to_string(),
    ]);

    if backend != "cpu" {
        cmd.args(["--gpu", &gpu.to_string()]);
    }

    // Spawn as background process
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                // Create new session to detach from terminal
                nix::libc::setsid();
                Ok(())
            });
        }
    }

    // Redirect stdout/stderr to log file
    let log_path = format!(".runtime/workers/{}.log", worker_id);
    std::fs::create_dir_all(".runtime/workers")?;
    let log_file = std::fs::File::create(&log_path)?;
    cmd.stdout(log_file.try_clone()?);
    cmd.stderr(log_file);

    let child = cmd.spawn()?;

    println!("{}", format!("✅ Worker spawned (PID: {})", child.id()).green());
    println!("   Endpoint: http://localhost:{}", port);
    println!("   Logs: {}", log_path);

    // Save worker info
    save_worker_info(&worker_id, child.id(), &backend, &model_id, port as u16)?;

    Ok(())
}

fn find_model_file(model_dir: &std::path::Path) -> Result<PathBuf> {
    // Look for .safetensors or .gguf files
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "safetensors" || ext == "gguf" {
                return Ok(path);
            }
        }
    }
    anyhow::bail!("No model file found in {}", model_dir.display())
}

fn save_worker_info(
    worker_id: &str,
    pid: u32,
    backend: &str,
    model_id: &str,
    port: u16,
) -> Result<()> {
    let info = serde_json::json!({
        "worker_id": worker_id,
        "pid": pid,
        "backend": backend,
        "model_id": model_id,
        "port": port,
        "started_at": chrono::Utc::now().to_rfc3339(),
    });

    let info_path = format!(".runtime/workers/{}.json", worker_id);
    std::fs::write(info_path, serde_json::to_string_pretty(&info)?)?;

    Ok(())
}

fn list() -> Result<()> {
    let workers_dir = PathBuf::from(".runtime/workers");

    if !workers_dir.exists() {
        println!("{}", "No workers running".yellow());
        return Ok(());
    }

    println!();
    println!("{}", "Running Workers".bold());
    println!("{}", "=".repeat(80));
    println!(
        "{:<20} {:<10} {:<15} {:<10} {:<10}",
        "Worker ID".bold(),
        "PID".bold(),
        "Model".bold(),
        "Backend".bold(),
        "Port".bold()
    );
    println!("{}", "-".repeat(80));

    for entry in std::fs::read_dir(&workers_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = std::fs::read_to_string(&path)?;
            let info: serde_json::Value = serde_json::from_str(&content)?;

            let worker_id = info["worker_id"].as_str().unwrap_or("unknown");
            let pid = info["pid"].as_u64().unwrap_or(0);
            let model_id = info["model_id"].as_str().unwrap_or("unknown");
            let backend = info["backend"].as_str().unwrap_or("unknown");
            let port = info["port"].as_u64().unwrap_or(0);

            // Check if process is still running
            let running = is_process_running(pid as u32);
            let status = if running { "✅" } else { "❌" };

            println!(
                "{:<20} {:<10} {:<15} {:<10} {:<10} {}",
                worker_id, pid, model_id, backend, port, status
            );
        }
    }

    println!("{}", "=".repeat(80));
    println!();

    Ok(())
}

fn stop(worker_id: String) -> Result<()> {
    let info_path = format!(".runtime/workers/{}.json", worker_id);

    if !std::path::Path::new(&info_path).exists() {
        anyhow::bail!("Worker {} not found", worker_id);
    }

    let content = std::fs::read_to_string(&info_path)?;
    let info: serde_json::Value = serde_json::from_str(&content)?;
    let pid = info["pid"].as_u64().unwrap_or(0) as i32;

    println!("🛑 Stopping worker {} (PID: {})", worker_id, pid);

    // Send SIGTERM
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;

        kill(Pid::from_raw(pid), Signal::SIGTERM)?;
    }

    // Wait a bit for graceful shutdown
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Check if still running
    if is_process_running(pid as u32) {
        println!("   Worker still running, sending SIGKILL");
        #[cfg(unix)]
        {
            use nix::sys::signal::{kill, Signal};
            use nix::unistd::Pid;
            kill(Pid::from_raw(pid), Signal::SIGKILL)?;
        }
    }

    // Remove worker info
    std::fs::remove_file(&info_path)?;

    println!("{}", format!("✅ Worker {} stopped", worker_id).green());

    Ok(())
}

#[cfg(unix)]
fn is_process_running(pid: u32) -> bool {
    use nix::sys::signal::kill;
    use nix::unistd::Pid;

    // Signal 0 is used to check if process exists
    kill(Pid::from_raw(pid as i32), None).is_ok()
}

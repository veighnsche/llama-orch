# Checkpoint 3: Automation - Model Downloads + Worker Spawning

**Created by:** TEAM-022  
**Checkpoint:** CP3  
**Duration:** Week 3 (5 days)  
**Status:** Pending  
**Depends On:** CP2 Complete

---

## Objective

<!-- TEAM-023: DEPRECATED - huggingface-cli is deprecated, use `hf` CLI instead! -->
**⚠️ WARNING: `huggingface-cli` is DEPRECATED. Use `hf` instead!**

Implement automation for:
1. Model downloads via hf CLI (NOT deprecated huggingface-cli)
2. Worker spawning via llorch-candled
3. Download Qwen (smallest model) on all pools
4. Test Qwen on Metal and CUDA backends

**Why This Third:** Need automation before we can test multiple models efficiently.

---

## Work Units

### WU3.1: Implement Model Download (Day 1-2)

**Location:** `bin/pool-ctl/src/commands/models.rs`

**Tasks:**
1. Implement hf CLI wrapper (TEAM-023: NOT huggingface-cli - that's deprecated!)
2. Wire download command
3. Update catalog after download
4. Add progress indicators

**Implementation:**
```rust
// TEAM-022: Model download automation
use std::process::Command;
use pool_core::catalog::ModelCatalog;

pub fn handle_download_command(model_id: String) -> Result<()> {
    let catalog_path = PathBuf::from(".test-models/catalog.json");
    let mut catalog = ModelCatalog::load(&catalog_path)?;
    
    // Find model in catalog
    let model = catalog.find_model_mut(&model_id)
        .ok_or_else(|| anyhow::anyhow!("Model {} not in catalog", model_id))?;
    
    if model.downloaded {
        println!("{}", format!("✅ Model {} already downloaded", model_id).green());
        return Ok(());
    }
    
    let repo = model.metadata["repo"].as_str()
        .ok_or_else(|| anyhow::anyhow!("No repo in metadata"))?;
    
    println!("{}", format!("📥 Downloading {} from {}", model.name, repo).cyan());
    println!("   Target: {}", model.path.display());
    
    // Create target directory
    std::fs::create_dir_all(&model.path)?;
    
    // TEAM-024: Use modern hf CLI (huggingface-cli is deprecated)
    let status = Command::new("hf")
        .args(&[
            "download",
            repo,
            "--include", "*.safetensors", "*.json", "tokenizer.model",
            "--local-dir", model.path.to_str().unwrap(),
        ])
        .status()?;
    
    if !status.success() {
        anyhow::bail!("Download failed for {}", model_id);
    }
    
    // Update catalog
    model.downloaded = true;
    
    // Calculate actual size
    let size_bytes = calculate_dir_size(&model.path)?;
    model.size_gb = size_bytes as f64 / 1_000_000_000.0;
    
    catalog.save(&catalog_path)?;
    
    println!("{}", format!("✅ Model {} downloaded successfully ({:.1} GB)", 
        model_id, model.size_gb).green());
    
    Ok(())
}

fn calculate_dir_size(path: &Path) -> Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total += metadata.len();
        } else if metadata.is_dir() {
            total += calculate_dir_size(&entry.path())?;
        }
    }
    Ok(total)
}
```

**Success Criteria:**
- [ ] `llorch-pool models download <model>` works
- [ ] Downloads SafeTensors format
- [ ] Updates catalog automatically
- [ ] Handles already-downloaded models
- [ ] Shows progress/status

---

### WU3.2: Implement Worker Spawning (Day 2-3)

**Location:** `bin/pool-ctl/src/commands/worker.rs`

**Tasks:**
1. Implement worker spawn logic
2. Validate model is downloaded
3. Check backend compatibility
4. Spawn llorch-candled as background process

**Implementation:**
```rust
// TEAM-022: Worker spawning automation
use std::process::Command;
use pool_core::catalog::ModelCatalog;

pub fn handle_spawn_command(
    backend: String,
    model_id: String,
    gpu: u32,
) -> Result<()> {
    let catalog_path = PathBuf::from(".test-models/catalog.json");
    let catalog = ModelCatalog::load(&catalog_path)?;
    
    // Find model
    let model = catalog.find_model(&model_id)
        .ok_or_else(|| anyhow::anyhow!("Model {} not found in catalog", model_id))?;
    
    // Validate model is downloaded
    if !model.downloaded {
        anyhow::bail!(
            "Model {} not downloaded. Run: llorch-pool models download {}", 
            model_id, model_id
        );
    }
    
    // Check backend compatibility
    if !model.backends.contains(&backend) {
        anyhow::bail!(
            "Model {} doesn't support {} backend. Supported: {:?}",
            model_id, backend, model.backends
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
    let binary = format!("llorch-candled");
    let mut cmd = Command::new(&binary);
    
    cmd.args(&[
        "--worker-id", &worker_id,
        "--model", model_path.to_str().unwrap(),
        "--backend", &backend,
        "--port", &port.to_string(),
    ]);
    
    if backend != "cpu" {
        cmd.args(&["--gpu", &gpu.to_string()]);
    }
    
    // Spawn as background process
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                // Create new session to detach from terminal
                libc::setsid();
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
    save_worker_info(&worker_id, child.id(), &backend, &model_id, port)?;
    
    Ok(())
}

fn find_model_file(model_dir: &Path) -> Result<PathBuf> {
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
```

**Success Criteria:**
- [ ] `llorch-pool worker spawn <backend> --model <model>` works
- [ ] Checks model is downloaded
- [ ] Checks backend compatibility
- [ ] Spawns worker as background process
- [ ] Logs to file
- [ ] Saves worker info

---

### WU3.3: Implement Worker Management (Day 3)

**Location:** `bin/pool-ctl/src/commands/worker.rs`

**Tasks:**
1. Implement `llorch-pool worker list`
2. Implement `llorch-pool worker stop`
3. Implement `llorch-pool worker logs`
4. Add worker status checks

**Implementation:**
```rust
// TEAM-022: Worker management commands

pub fn handle_list_command() -> Result<()> {
    let workers_dir = PathBuf::from(".runtime/workers");
    
    if !workers_dir.exists() {
        println!("{}", "No workers running".yellow());
        return Ok(());
    }
    
    println!("\n{}", "Running Workers".bold());
    println!("{}", "=".repeat(80));
    println!("{:<20} {:<10} {:<15} {:<10} {:<10}", 
        "Worker ID".bold(), "PID".bold(), "Model".bold(), "Backend".bold(), "Port".bold());
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
            
            println!("{:<20} {:<10} {:<15} {:<10} {:<10} {}", 
                worker_id, pid, model_id, backend, port, status);
        }
    }
    
    println!("{}", "=".repeat(80));
    println!();
    
    Ok(())
}

pub fn handle_stop_command(worker_id: String) -> Result<()> {
    let info_path = format!(".runtime/workers/{}.json", worker_id);
    
    if !Path::new(&info_path).exists() {
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
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;
    
    kill(Pid::from_raw(pid as i32), Signal::from_c_int(0).unwrap()).is_ok()
}
```

**Success Criteria:**
- [ ] `llorch-pool worker list` shows running workers
- [ ] `llorch-pool worker stop <id>` stops worker
- [ ] Process status is accurate
- [ ] Graceful shutdown works

---

### WU3.4: Download Qwen on All Pools (Day 4)

**Tasks:**
1. Download Qwen on mac.home.arpa
2. Download Qwen on workstation.home.arpa
3. Verify downloads
4. Test catalog updates

**Commands:**
```bash
# On mac.home.arpa
llorch-pool models download qwen-0.5b

# On workstation.home.arpa
llorch-pool models download qwen-0.5b

# Verify
llorch-pool models catalog
```

**Expected Output:**
```
📥 Downloading Qwen2.5 0.5B Instruct from Qwen/Qwen2.5-0.5B-Instruct
   Target: .test-models/qwen-0.5b
Downloading model.safetensors: 100%|████████| 1.0GB/1.0GB [00:30<00:00, 33.3MB/s]
✅ Model qwen-0.5b downloaded successfully (1.0 GB)
```

**Success Criteria:**
- [ ] Qwen downloaded on mac.home.arpa
- [ ] Qwen downloaded on workstation.home.arpa
- [ ] Catalog shows downloaded=true
- [ ] Size is accurate (~1GB)

---

### WU3.5: Test Qwen on Metal and CUDA (Day 5)

**Tasks:**
1. Spawn Qwen worker on Metal (mac)
2. Test inference on Metal
3. Spawn Qwen worker on CUDA (workstation)
4. Test inference on CUDA
5. Document results

**Test Commands:**
```bash
# On mac.home.arpa (Metal)
llorch-pool worker spawn metal --model qwen-0.5b --gpu 0

# Wait for worker to start
sleep 10

# Test inference
curl -X POST http://localhost:8000/execute \
    -H "Content-Type: application/json" \
    -d '{
        "job_id": "test-qwen-metal",
        "prompt": "Hello, how are you?",
        "max_tokens": 20,
        "temperature": 0.0,
        "seed": 42
    }'

# Stop worker
llorch-pool worker stop worker-metal-0
```

```bash
# On workstation.home.arpa (CUDA)
llorch-pool worker spawn cuda --model qwen-0.5b --gpu 0

# Wait for worker to start
sleep 10

# Test inference
curl -X POST http://localhost:8000/execute \
    -H "Content-Type: application/json" \
    -d '{
        "job_id": "test-qwen-cuda",
        "prompt": "Hello, how are you?",
        "max_tokens": 20,
        "temperature": 0.0,
        "seed": 42
    }'

# Stop worker
llorch-pool worker stop worker-cuda-0
```

**Expected Results:**
- Worker spawns successfully
- Model loads into VRAM
- Inference completes without errors
- Tokens are generated
- No broadcasting errors
- Worker stops gracefully

**Success Criteria:**
- [ ] Qwen works on Metal backend
- [ ] Qwen works on CUDA backend
- [ ] No errors in logs
- [ ] Tokens generated correctly
- [ ] Worker lifecycle works (spawn → inference → stop)

---

## Checkpoint Gate: CP3 Verification

**Before proceeding to CP4, verify:**

### Model Downloads
- [ ] `llorch-pool models download` works
- [ ] Qwen downloaded on mac.home.arpa
- [ ] Qwen downloaded on workstation.home.arpa
- [ ] Catalog updated correctly
- [ ] File sizes are accurate

### Worker Spawning
- [ ] `llorch-pool worker spawn` works
- [ ] Workers spawn as background processes
- [ ] Logs are written to files
- [ ] Worker info is saved

### Worker Management
- [ ] `llorch-pool worker list` shows workers
- [ ] `llorch-pool worker stop` stops workers
- [ ] Process status is accurate
- [ ] Cleanup works correctly

### Qwen Testing
- [ ] Qwen works on Metal (mac)
- [ ] Qwen works on CUDA (workstation)
- [ ] No broadcasting errors
- [ ] Inference completes successfully
- [ ] Tokens are generated

### Remote Operations
- [ ] `llorch pool models download --host mac` works
- [ ] `llorch pool worker spawn --host mac` works
- [ ] SSH commands execute correctly

### Code Quality
- [ ] `cargo fmt --all` clean
- [ ] `cargo clippy --all` clean
- [ ] Team signatures added
- [ ] Documentation updated

---

## Deliverables

**Code:**
- `bin/pool-ctl/src/commands/models.rs` (download command)
- `bin/pool-ctl/src/commands/worker.rs` (spawn/list/stop commands)
- `bin/llorch-ctl/src/commands/pool.rs` (remote commands)

**Downloaded Models:**
- `.test-models/qwen-0.5b/` on mac.home.arpa
- `.test-models/qwen-0.5b/` on workstation.home.arpa

**Documentation:**
- Download guide
- Worker spawning guide
- Troubleshooting guide

---

## Dependencies

**Additional Cargo.toml dependencies:**
```toml
# pool-ctl
nix = { version = "0.27", features = ["signal", "process"] }
```

**System Requirements:**
- `hf` CLI installed on all pools (TEAM-023: NOT huggingface-cli - that's deprecated!)
  - Install: `pip install huggingface_hub[cli]`
- SSH access to all pools
- llorch-candled binary built for each backend

---

## Risk Mitigation

**Risk 1:** Download failures (network issues)  
**Mitigation:** Retry logic, resume support, verify checksums

**Risk 2:** Worker spawn failures  
**Mitigation:** Validate before spawn, clear error messages, cleanup on failure

**Risk 3:** Process management issues  
**Mitigation:** Robust PID tracking, graceful shutdown, orphan detection

---

## Next Checkpoint

After CP3 gate passes, proceed to `04_CP4_MULTI_MODEL.md`.

---

**Status:** Ready to start after CP2  
**Estimated Duration:** 5 days  
**Blocking:** CP2 must be complete

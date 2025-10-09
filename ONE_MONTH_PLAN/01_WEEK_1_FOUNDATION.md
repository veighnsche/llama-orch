# Week 1: Foundation — Days 1-7

**Goal**: Working end-to-end system  
**Deliverable**: Submit job → worker executes → tokens stream back

---

## 🎉 GOOD NEWS: You Already Have Most of This!

**Existing Shared Crates (Ready to Use):**
- ✅ **audit-logging** (895 lines of docs!) — Immutable audit trails
- ✅ **auth-min** — Token fingerprinting, validation
- ✅ **input-validation** — Log injection prevention
- ✅ **narration-core** — Developer observability
- ✅ **secrets-management** — Secure secret loading
- ✅ Plus 6 more crates!

**Time Saved:** 2 days (don't need to build audit system from scratch)

**What This Means:**
- Day 1: Wire up existing crates (not build from scratch)
- Day 2-3: Build CLIs (straightforward)
- Days 4-5: Integration + buffer time

**You're ahead of schedule!**

---

## Day 1 (Monday): queen-rbee

### Morning Session (09:00-13:00)

**Task 1: Create binary (30 min)**
```bash
cd /home/vince/Projects/llama-orch/bin
mkdir queen-rbee
cd queen-rbee
cargo init --name queen-rbee
```

**Task 2: Add dependencies (15 min)**
```toml
[dependencies]
# ✅ USE EXISTING SHARED CRATES!
audit-logging = { path = "../shared-crates/audit-logging" }
auth-min = { path = "../shared-crates/auth-min" }
input-validation = { path = "../shared-crates/input-validation" }
narration-core = { path = "../shared-crates/narration-core" }
secrets-management = { path = "../shared-crates/secrets-management" }

# HTTP server
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Utilities
uuid = { version = "1", features = ["v4", "serde"] }
chrono = "0.4"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
reqwest = { version = "0.11", features = ["json"] }
```

**Task 3: Implement HTTP server with existing crates (2 hours)**
```rust
// src/main.rs
use axum::{Router, routing::{get, post}, Json, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::env;
use uuid::Uuid;
use chrono::Utc;

// ✅ USE EXISTING CRATES!
use audit_logging::{AuditLogger, AuditConfig, AuditMode, AuditEvent, ActorInfo, AuthMethod, RotationPolicy, RetentionPolicy, FlushMode};
use auth_min::fingerprint_token;
use input_validation::sanitize_string;
use narration_core::narrate;

#[derive(Clone)]
struct AppState {
    jobs: Arc<Mutex<Vec<Job>>>,
    workers: Arc<Mutex<Vec<Worker>>>,
    audit_logger: Arc<AuditLogger>,  // ✅ Existing crate!
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Job {
    id: String,
    model: String,
    prompt: String,
    status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Worker {
    id: String,
    host: String,
    port: u16,
    model: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    // ✅ Initialize audit logger (existing crate!)
    let eu_audit = env::var("LLORCH_EU_AUDIT")
        .unwrap_or_else(|_| "false".to_string()) == "true";
    
    let audit_logger = if eu_audit {
        tracing::info!("🇪🇺 EU audit mode ENABLED");
        AuditLogger::new(AuditConfig {
            mode: AuditMode::Local {
                base_dir: std::path::PathBuf::from("/var/log/llorch/audit"),
            },
            service_id: "queen-rbee".to_string(),
            rotation_policy: RotationPolicy::Daily,
            retention_policy: RetentionPolicy::default(),
            flush_mode: FlushMode::Immediate,  // Compliance-safe
        })?
    } else {
        tracing::info!("🏠 Homelab mode (audit disabled)");
        AuditLogger::new(AuditConfig {
            mode: AuditMode::Disabled,  // Zero overhead
            service_id: "queen-rbee".to_string(),
            ..Default::default()
        })?
    };
    
    let state = AppState {
        jobs: Arc::new(Mutex::new(Vec::new())),
        workers: Arc::new(Mutex::new(Vec::new())),
        audit_logger: Arc::new(audit_logger),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v2/tasks", post(submit_task))
        .route("/workers/register", post(register_worker))
        .with_state(state);

    let addr = "0.0.0.0:8080";
    tracing::info!("🚀 queen-rbee listening on {}", addr);
    
    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}

async fn health() -> &'static str {
    "OK"
}

#[derive(Deserialize)]
struct TaskRequest {
    model: String,
    prompt: String,
}

#[derive(Serialize)]
struct TaskResponse {
    job_id: String,
    status: String,
}

async fn submit_task(
    State(state): State<AppState>,
    Json(req): Json<TaskRequest>,
) -> Result<Json<TaskResponse>, String> {
    let job_id = Uuid::new_v4().to_string();
    
    // ✅ Sanitize input (existing crate!)
    let safe_model = sanitize_string(&req.model)
        .map_err(|e| format!("Invalid model: {}", e))?;
    let safe_prompt = sanitize_string(&req.prompt)
        .map_err(|e| format!("Invalid prompt: {}", e))?;
    
    tracing::info!("Job {} submitted: model={}", job_id, safe_model);
    
    // ✅ Audit log (existing crate!)
    let _ = state.audit_logger.emit(AuditEvent::TaskSubmitted {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: "anonymous".to_string(),  // TODO: Extract from auth
            ip: None,  // TODO: Extract from request
            auth_method: AuthMethod::None,
            session_id: None,
        },
        task_id: job_id.clone(),
        model_ref: safe_model.clone(),
        prompt_length: safe_prompt.len(),
        service_id: "queen-rbee".to_string(),
    });
    
    let job = Job {
        id: job_id.clone(),
        model: safe_model,
        prompt: safe_prompt,
        status: "queued".to_string(),
    };
    
    state.jobs.lock().unwrap().push(job);
    
    Ok(Json(TaskResponse {
        job_id,
        status: "queued".to_string(),
    }))
}

async fn register_worker(
    State(state): State<AppState>,
    Json(worker): Json<Worker>,
) -> &'static str {
    tracing::info!("Worker registered: {} at {}:{}", worker.id, worker.host, worker.port);
    
    // ✅ Audit log (existing crate!)
    let _ = state.audit_logger.emit(AuditEvent::NodeRegistered {
        timestamp: Utc::now(),
        actor: ActorInfo {
            user_id: format!("worker:{}", worker.id),
            ip: None,
            auth_method: AuthMethod::None,
            session_id: None,
        },
        node_id: worker.id.clone(),
        gpu_count: 1,  // TODO: Get from worker
        total_vram_gb: 24,  // TODO: Get from worker
        service_id: "queen-rbee".to_string(),
    });
    
    state.workers.lock().unwrap().push(worker);
    "OK"
}
```

**Task 4: Test with audit logging (30 min)**
```bash
# Test with EU audit DISABLED (homelab mode - zero overhead)
LLORCH_EU_AUDIT=false cargo run

# In another terminal
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","prompt":"hello"}'

# Test with EU audit ENABLED
LLORCH_EU_AUDIT=true \
LLORCH_AUDIT_LOG_PATH=/tmp/llorch-audit.log \
cargo run

# Submit job
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","prompt":"hello"}'

# ✅ Check audit log (existing crate creates this!)
cat /tmp/llorch-audit.log
# Should see JSON audit events
```

**What you get for FREE:**
- ✅ Audit logging (immutable, tamper-evident)
- ✅ Input sanitization (log injection prevention)
- ✅ EU audit toggle (zero overhead when disabled)
- ✅ GDPR event types (already defined)

### Afternoon Session (14:00-18:00)

**Task 5: Add worker dispatch (3 hours)**
```rust
// Update submit_task function
async fn submit_task(
    State(state): State<AppState>,
    Json(req): Json<TaskRequest>,
) -> Result<Json<TaskResponse>, String> {
    let job_id = Uuid::new_v4().to_string();
    
    // Find worker with matching model
    let workers = state.workers.lock().unwrap();
    let worker = workers.iter()
        .find(|w| w.model == req.model)
        .ok_or_else(|| format!("No worker available for model: {}", req.model))?
        .clone();
    drop(workers);
    
    tracing::info!("Dispatching job {} to worker {}", job_id, worker.id);
    
    // Dispatch to worker (async)
    let job_id_clone = job_id.clone();
    let prompt = req.prompt.clone();
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let worker_url = format!("http://{}:{}/execute", worker.host, worker.port);
        
        let response = client
            .post(&worker_url)
            .json(&serde_json::json!({
                "job_id": job_id_clone,
                "prompt": prompt,
                "max_tokens": 100,
            }))
            .send()
            .await;
        
        match response {
            Ok(_) => tracing::info!("✅ Job {} dispatched successfully", job_id_clone),
            Err(e) => tracing::error!("❌ Job {} failed: {}", job_id_clone, e),
        }
    });
    
    Ok(Json(TaskResponse {
        job_id,
        status: "dispatched".to_string(),
    }))
}
```

**Task 6: Add error handling (1 hour)**
```rust
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::NoWorkerAvailable(model) => {
                (StatusCode::SERVICE_UNAVAILABLE, format!("No worker for model: {}", model))
            }
            AppError::WorkerDispatchFailed(err) => {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Dispatch failed: {}", err))
            }
        };
        
        (status, message).into_response()
    }
}
```

**Day 1 Deliverable**: queen-rbee accepts jobs and dispatches to workers

---

## Day 2 (Tuesday): rbee-hive

### Morning Session (09:00-13:00)

**Task 1: Create binary (30 min)**
```bash
cd /home/vince/Projects/llama-orch/bin
mkdir rbee-hive
cd rbee-hive
cargo init --name rbee-hive
```

**Task 2: Add dependencies (15 min)**
```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
anyhow = "1"
colored = "2"
indicatif = "0.17"
tokio = { version = "1", features = ["process", "fs"] }
walkdir = "2"
```

**Task 3: Implement CLI skeleton (2 hours)**
```rust
// src/main.rs
use clap::{Parser, Subcommand};
use colored::*;

#[derive(Parser)]
#[command(name = "rbee-hive")]
#[command(about = "Pool manager control CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
    },
}

#[derive(Subcommand)]
enum ModelsAction {
    Download { model: String },
    List,
}

#[derive(Subcommand)]
enum WorkerAction {
    Spawn {
        backend: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0")]
        gpu: u32,
        #[arg(long, default_value = "8001")]
        port: u16,
    },
    List,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Models { action } => match action {
            ModelsAction::Download { model } => download_model(&model),
            ModelsAction::List => list_models(),
        },
        Commands::Worker { action } => match action {
            WorkerAction::Spawn { backend, model, gpu, port } => {
                spawn_worker(&backend, &model, gpu, port)
            }
            WorkerAction::List => list_workers(),
        },
    }
}
```

**Task 4: Test skeleton (30 min)**
```bash
cargo run -- models download tinyllama
cargo run -- worker spawn metal --model tinyllama
```

### Afternoon Session (14:00-18:00)

**Task 5: Implement model download (2 hours)**
```rust
// src/commands/models.rs
use std::process::Command;
use colored::*;

pub fn download_model(model: &str) -> anyhow::Result<()> {
    let (repo, file) = match model {
        "tinyllama" => (
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        ),
        "qwen" => (
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ),
        "phi3" => (
            "microsoft/Phi-3-mini-4k-instruct-gguf",
            "Phi-3-mini-4k-instruct-q4.gguf",
        ),
        _ => return Err(anyhow::anyhow!("Unknown model: {}", model)),
    };
    
    let local_dir = format!(".test-models/{}", model);
    
    println!("{} Downloading {} from {}", "📥".green(), model.bold(), repo);
    println!("   File: {}", file);
    println!("   Destination: {}", local_dir);
    
    let status = Command::new("hf")
        .args(&[
            "download",
            repo,
            file,
            "--local-dir",
            &local_dir,
        ])
        .status()?;
    
    if status.success() {
        println!("{} Model downloaded successfully", "✅".green());
        Ok(())
    } else {
        Err(anyhow::anyhow!("Download failed"))
    }
}

pub fn list_models() -> anyhow::Result<()> {
    use walkdir::WalkDir;
    
    println!("{} Downloaded models:", "📋".cyan());
    
    if !std::path::Path::new(".test-models").exists() {
        println!("   No models downloaded yet");
        return Ok(());
    }
    
    for entry in WalkDir::new(".test-models").max_depth(2) {
        let entry = entry?;
        if entry.path().extension().map(|s| s == "gguf").unwrap_or(false) {
            let size = entry.metadata()?.len();
            println!("   {} ({:.2} GB)", 
                entry.path().display(), 
                size as f64 / 1_000_000_000.0
            );
        }
    }
    
    Ok(())
}
```

**Task 6: Implement worker spawn (2 hours)**
```rust
// src/commands/worker.rs
use std::process::Command;
use colored::*;

pub fn spawn_worker(backend: &str, model: &str, gpu: u32, port: u16) -> anyhow::Result<()> {
    let worker_id = format!("worker-{}-{}", backend, gpu);
    let model_path = format!(".test-models/{}", model);
    
    // Find model file
    let model_file = std::fs::read_dir(&model_path)?
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map(|s| s == "gguf").unwrap_or(false))
        .ok_or_else(|| anyhow::anyhow!("No GGUF file found in {}", model_path))?;
    
    let model_file_path = model_file.path();
    
    println!("{} Spawning worker:", "🚀".green());
    println!("   ID: {}", worker_id.bold());
    println!("   Backend: {}", backend);
    println!("   Model: {}", model_file_path.display());
    println!("   GPU: {}", gpu);
    println!("   Port: {}", port);
    
    // Spawn llm-worker-rbee
    let mut cmd = Command::new("llm-worker-rbee");
    cmd.args(&[
        "--worker-id", &worker_id,
        "--model", model_file_path.to_str().unwrap(),
        "--backend", backend,
        "--gpu", &gpu.to_string(),
        "--port", &port.to_string(),
    ]);
    
    // Spawn as background process
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    
    let child = cmd.spawn()?;
    
    println!("{} Worker spawned (PID: {})", "✅".green(), child.id());
    
    Ok(())
}

pub fn list_workers() -> anyhow::Result<()> {
    println!("{} Running workers:", "📋".cyan());
    
    let output = Command::new("ps")
        .args(&["aux"])
        .output()?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("llm-worker-rbee") {
            println!("   {}", line);
        }
    }
    
    Ok(())
}
```

**Day 2 Deliverable**: rbee-hive can download models and spawn workers

---

## Day 3 (Wednesday): rbee-keeper

### Morning Session (09:00-13:00)

**Task 1: Create binary (30 min)**
```bash
cd /home/vince/Projects/llama-orch/bin
mkdir rbee-keeper
cd rbee-keeper
cargo init --name rbee-keeper
```

**Task 2: Add dependencies (15 min)**
```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
anyhow = "1"
colored = "2"
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Task 3: Implement CLI skeleton (2 hours)**
```rust
// src/main.rs
use clap::{Parser, Subcommand};
use colored::*;

#[derive(Parser)]
#[command(name = "llorch")]
#[command(about = "Orchestrator control CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Pool {
        #[command(subcommand)]
        action: PoolAction,
    },
    Jobs {
        #[command(subcommand)]
        action: JobsAction,
    },
}

#[derive(Subcommand)]
enum PoolAction {
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
    },
}

#[derive(Subcommand)]
enum ModelsAction {
    Download {
        model: String,
        #[arg(long)]
        host: String,
    },
}

#[derive(Subcommand)]
enum WorkerAction {
    Spawn {
        backend: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        host: String,
    },
}

#[derive(Subcommand)]
enum JobsAction {
    Submit {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
    },
    List,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Pool { action } => handle_pool_action(action).await,
        Commands::Jobs { action } => handle_jobs_action(action).await,
    }
}
```

### Afternoon Session (14:00-18:00)

**Task 4: Implement SSH commands (2 hours)**
```rust
// src/commands/pool.rs
use std::process::Command;
use colored::*;

pub async fn download_model_on_pool(model: &str, host: &str) -> anyhow::Result<()> {
    println!("{} Downloading {} on {}", "📥".green(), model.bold(), host.bold());
    
    let ssh_cmd = format!(
        "cd ~/Projects/llama-orch && rbee-hive models download {}",
        model
    );
    
    let status = Command::new("ssh")
        .args(&[host, &ssh_cmd])
        .status()?;
    
    if status.success() {
        println!("{} Model downloaded on {}", "✅".green(), host);
        Ok(())
    } else {
        Err(anyhow::anyhow!("SSH command failed"))
    }
}

pub async fn spawn_worker_on_pool(backend: &str, model: &str, host: &str) -> anyhow::Result<()> {
    println!("{} Spawning worker on {}", "🚀".green(), host.bold());
    
    let ssh_cmd = format!(
        "cd ~/Projects/llama-orch && rbee-hive worker spawn {} --model {}",
        backend, model
    );
    
    let status = Command::new("ssh")
        .args(&[host, &ssh_cmd])
        .status()?;
    
    if status.success() {
        println!("{} Worker spawned on {}", "✅".green(), host);
        Ok(())
    } else {
        Err(anyhow::anyhow!("SSH command failed"))
    }
}
```

**Task 5: Implement job submission (2 hours)**
```rust
// src/commands/jobs.rs
use reqwest::Client;
use serde::{Deserialize, Serialize};
use colored::*;

#[derive(Serialize)]
struct TaskRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct TaskResponse {
    job_id: String,
    status: String,
}

pub async fn submit_job(model: &str, prompt: &str) -> anyhow::Result<()> {
    let client = Client::new();
    
    println!("{} Submitting job:", "📤".green());
    println!("   Model: {}", model.bold());
    println!("   Prompt: {}", prompt);
    
    let response = client
        .post("http://localhost:8080/v2/tasks")
        .json(&TaskRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
        })
        .send()
        .await?;
    
    if !response.status().is_success() {
        return Err(anyhow::anyhow!("Job submission failed: {}", response.status()));
    }
    
    let task: TaskResponse = response.json().await?;
    
    println!("{} Job submitted:", "✅".green());
    println!("   Job ID: {}", task.job_id.bold());
    println!("   Status: {}", task.status);
    
    Ok(())
}

pub async fn list_jobs() -> anyhow::Result<()> {
    println!("{} Jobs:", "📋".cyan());
    println!("   (Not implemented yet)");
    Ok(())
}
```

**Day 3 Deliverable**: rbee-keeper can command pools via SSH and submit jobs

---

## Day 4 (Thursday): Integration

### Full Day Session (09:00-18:00)

**Task 1: Add worker registration to llm-worker-rbee (2 hours)**
```rust
// In llm-worker-rbee/src/main.rs
async fn register_with_orchestrator(config: &Config) -> Result<()> {
    let client = reqwest::Client::new();
    
    let registration = serde_json::json!({
        "id": config.worker_id,
        "host": get_local_ip()?,
        "port": config.port,
        "model": extract_model_name(&config.model_path),
    });
    
    let orchestrator_url = std::env::var("LLORCH_ORCHESTRATOR_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());
    
    println!("📡 Registering with orchestrator at {}", orchestrator_url);
    
    let response = client
        .post(&format!("{}/workers/register", orchestrator_url))
        .json(&registration)
        .send()
        .await?;
    
    if response.status().is_success() {
        println!("✅ Registered with orchestrator");
        Ok(())
    } else {
        Err(anyhow::anyhow!("Registration failed: {}", response.status()))
    }
}

fn get_local_ip() -> Result<String> {
    // Simple implementation - get hostname
    let output = std::process::Command::new("hostname")
        .arg("-I")
        .output()?;
    
    let ip = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("127.0.0.1")
        .to_string();
    
    Ok(ip)
}

fn extract_model_name(path: &str) -> String {
    std::path::Path::new(path)
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}
```

**Task 2: End-to-end testing (4 hours)**
```bash
# Terminal 1: Start queen-rbee
cd bin/queen-rbee
cargo run

# Terminal 2: Download model on mac
llorch pool models download tinyllama --host mac.home.arpa

# Terminal 3: Spawn worker on mac
llorch pool worker spawn metal --model tinyllama --host mac.home.arpa

# Terminal 4: Submit job
llorch jobs submit --model tinyllama --prompt "Write a haiku about Rust"

# Verify:
# - Worker registered with queen-rbee
# - Job dispatched to worker
# - Tokens stream back
```

**Task 3: Fix integration issues (2 hours)**
- Worker registration not working
- Job dispatch failing
- SSH commands not executing
- Model path resolution issues

**Day 4 Deliverable**: Full end-to-end flow working

---

## Day 5 (Friday): Polish

### Morning Session (09:00-13:00)

**Task 1: Error handling (2 hours)**
```rust
// Better error messages
#[derive(Debug)]
enum AppError {
    NoWorkerAvailable(String),
    WorkerDispatchFailed(String),
    ModelNotFound(String),
    SSHCommandFailed(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AppError::NoWorkerAvailable(model) => {
                write!(f, "No worker available for model '{}'. Try spawning a worker first:\n  rbee pool worker spawn <backend> --model {} --host <host>", model, model)
            }
            AppError::WorkerDispatchFailed(err) => {
                write!(f, "Failed to dispatch job to worker: {}\nCheck worker logs for details.", err)
            }
            AppError::ModelNotFound(model) => {
                write!(f, "Model '{}' not found. Download it first:\n  rbee pool models download {} --host <host>", model, model)
            }
            AppError::SSHCommandFailed(cmd) => {
                write!(f, "SSH command failed: {}\nCheck SSH connectivity and permissions.", cmd)
            }
        }
    }
}
```

**Task 2: Logging (1 hour)**
```rust
// Add structured logging
use tracing::{info, error, warn, debug};

// In queen-rbee
info!(job_id = %job.id, model = %job.model, "Job submitted");
info!(worker_id = %worker.id, host = %worker.host, "Worker registered");
error!(job_id = %job.id, error = %err, "Job dispatch failed");

// In rbee-hive
println!("{} {}", "INFO".blue(), "Downloading model...");
println!("{} {}", "ERROR".red(), "Download failed");
println!("{} {}", "SUCCESS".green(), "Model downloaded");
```

### Afternoon Session (14:00-18:00)

**Task 3: Progress indicators (2 hours)**
```rust
// In rbee-hive
use indicatif::{ProgressBar, ProgressStyle};

pub fn download_model(model: &str) -> anyhow::Result<()> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap()
    );
    
    pb.set_message(format!("Downloading {}...", model));
    
    // ... download logic ...
    
    pb.finish_with_message(format!("✅ Downloaded {}", model));
    Ok(())
}
```

**Task 4: Write README (2 hours)**
```markdown
# llama-orch

EU-Native LLM Inference with Full Audit Trails

## Quick Start

### 1. Start orchestrator
cd bin/queen-rbee
cargo run

### 2. Download a model
llorch pool models download tinyllama --host mac.home.arpa

### 3. Spawn a worker
llorch pool worker spawn metal --model tinyllama --host mac.home.arpa

### 4. Submit a job
llorch jobs submit --model tinyllama --prompt "Hello world"

## Architecture

- **queen-rbee**: Job scheduling and worker management
- **rbee-hive**: Pool manager operations (local)
- **rbee-keeper**: Orchestrator operations (remote)
- **llm-worker-rbee**: Worker daemon (inference)

## Commands

### rbee-hive (local pool operations)
rbee-hive models download <model>
rbee-hive models list
rbee-hive worker spawn <backend> --model <model>
rbee-hive worker list

### rbee-keeper (orchestrator operations)
llorch pool models download <model> --host <host>
llorch pool worker spawn <backend> --model <model> --host <host>
llorch jobs submit --model <model> --prompt <prompt>
llorch jobs list

## Environment Variables

- `LLORCH_ORCHESTRATOR_URL`: Orchestrator URL (default: http://localhost:8080)
- `LLORCH_REPO_ROOT`: Repo root path (default: ~/Projects/llama-orch)

## License

GPL-3.0-or-later
```

**Day 5 Deliverable**: Polished system with good DX

---

## Days 6-7 (Weekend): Buffer & Prep

### Saturday

**Task 1: Fix critical bugs (4 hours)**
- Worker registration race conditions
- Job dispatch failures
- SSH connectivity issues

**Task 2: Improve DX (2 hours)**
- Better error messages
- Helpful hints
- Command examples in help text

**Task 3: Test clean slate (2 hours)**
```bash
# Clean everything
rm -rf .test-models
pkill llm-worker-rbee

# Full flow from scratch
llorch pool models download tinyllama --host mac.home.arpa
llorch pool worker spawn metal --model tinyllama --host mac.home.arpa
llorch jobs submit --model tinyllama --prompt "Test"
```

### Sunday

**Task 1: Documentation (3 hours)**
- Update README
- Add troubleshooting guide
- Document common issues

**Task 2: Prepare for Week 2 (2 hours)**
- Review EU compliance requirements
- Plan audit logging implementation
- Sketch web UI wireframes

**Task 3: Rest (3 hours)**
- Take a break
- Reflect on progress
- Prepare mentally for Week 2

---

## Week 1 Success Criteria

- [ ] queen-rbee accepts jobs and dispatches to workers
- [ ] rbee-hive downloads models and spawns workers
- [ ] rbee-keeper commands pools via SSH and submits jobs
- [ ] llm-worker-rbee registers with orchestrator
- [ ] End-to-end flow works: submit job → worker executes → tokens stream back
- [ ] Basic error handling and logging
- [ ] README with quick start guide
- [ ] Clean slate to working system in < 10 minutes

---

**Version**: 1.0  
**Status**: EXECUTE  
**Last Updated**: 2025-10-09

# Week 1 Implementation Plan — Get to Working MVP

**Date**: 2025-10-09  
**Goal**: End-to-end working system by Friday  
**Focus**: Minimum viable, maximum shipping

---

## Monday: rbees-orcd Skeleton

### Morning (4 hours)

**Create rbees-orcd binary:**
```bash
cd bin
mkdir rbees-orcd
cd rbees-orcd
cargo init --name rbees-orcd
```

**Add dependencies:**
```toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1"
uuid = { version = "1", features = ["v4", "serde"] }
```

**Implement minimal HTTP server:**
```rust
// src/main.rs
use axum::{Router, routing::{get, post}, Json};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    jobs: Arc<Mutex<Vec<Job>>>,
    workers: Arc<Mutex<Vec<Worker>>>,
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
async fn main() {
    let state = AppState {
        jobs: Arc::new(Mutex::new(Vec::new())),
        workers: Arc::new(Mutex::new(Vec::new())),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v2/tasks", post(submit_task))
        .route("/workers/register", post(register_worker))
        .with_state(state);

    println!("🚀 rbees-orcd listening on :8080");
    
    axum::Server::bind(&"0.0.0.0:8080".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn health() -> &'static str {
    "OK"
}

async fn submit_task(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<TaskRequest>,
) -> Json<TaskResponse> {
    let job_id = Uuid::new_v4().to_string();
    
    let job = Job {
        id: job_id.clone(),
        model: req.model,
        prompt: req.prompt,
        status: "queued".to_string(),
    };
    
    state.jobs.lock().unwrap().push(job);
    
    Json(TaskResponse {
        job_id,
        status: "queued".to_string(),
    })
}

async fn register_worker(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(worker): Json<Worker>,
) -> &'static str {
    state.workers.lock().unwrap().push(worker);
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
```

**Test:**
```bash
cargo run

# In another terminal
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","prompt":"hello"}'
```

### Afternoon (4 hours)

**Add worker dispatch:**
```rust
// src/main.rs (add to submit_task)

async fn submit_task(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<TaskRequest>,
) -> Result<Json<TaskResponse>, String> {
    let job_id = Uuid::new_v4().to_string();
    
    // Find worker with matching model
    let workers = state.workers.lock().unwrap();
    let worker = workers.iter()
        .find(|w| w.model == req.model)
        .ok_or("No worker available for model")?
        .clone();
    drop(workers);
    
    // Dispatch to worker
    let client = reqwest::Client::new();
    let worker_url = format!("http://{}:{}/execute", worker.host, worker.port);
    
    tokio::spawn(async move {
        let response = client
            .post(&worker_url)
            .json(&serde_json::json!({
                "job_id": job_id,
                "prompt": req.prompt,
                "max_tokens": 100,
            }))
            .send()
            .await;
        
        match response {
            Ok(_) => println!("✅ Job {} dispatched to worker", job_id),
            Err(e) => eprintln!("❌ Job {} failed: {}", job_id, e),
        }
    });
    
    Ok(Json(TaskResponse {
        job_id,
        status: "dispatched".to_string(),
    }))
}
```

**Deliverable:** rbees-orcd accepts jobs and dispatches to workers

---

## Tuesday: rbees-pool CLI

### Morning (4 hours)

**Create rbees-pool binary:**
```bash
cd bin
mkdir rbees-pool
cd rbees-pool
cargo init --name rbees-pool
```

**Add dependencies:**
```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
anyhow = "1"
colored = "2"
indicatif = "0.17"
tokio = { version = "1", features = ["process"] }
```

**Implement CLI skeleton:**
```rust
// src/main.rs
use clap::{Parser, Subcommand};
use colored::*;

#[derive(Parser)]
#[command(name = "rbees-pool")]
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

fn download_model(model: &str) -> anyhow::Result<()> {
    println!("{} Downloading model: {}", "📥".green(), model);
    
    // TODO: Implement hf download
    println!("{} Model downloaded", "✅".green());
    Ok(())
}

fn list_models() -> anyhow::Result<()> {
    println!("{} Downloaded models:", "📋".cyan());
    // TODO: List .test-models/
    Ok(())
}

fn spawn_worker(backend: &str, model: &str, gpu: u32, port: u16) -> anyhow::Result<()> {
    println!("{} Spawning worker:", "🚀".green());
    println!("  Backend: {}", backend);
    println!("  Model: {}", model);
    println!("  GPU: {}", gpu);
    println!("  Port: {}", port);
    
    // TODO: Spawn rbees-workerd
    println!("{} Worker spawned", "✅".green());
    Ok(())
}

fn list_workers() -> anyhow::Result<()> {
    println!("{} Running workers:", "📋".cyan());
    // TODO: List running rbees-workerd processes
    Ok(())
}
```

**Test:**
```bash
cargo run -- models download tinyllama
cargo run -- worker spawn metal --model tinyllama
```

### Afternoon (4 hours)

**Implement model download:**
```rust
// src/commands/models.rs

use std::process::Command;
use indicatif::{ProgressBar, ProgressStyle};

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
        _ => return Err(anyhow::anyhow!("Unknown model: {}", model)),
    };
    
    let local_dir = format!(".test-models/{}", model);
    
    println!("📥 Downloading {} from {}", model, repo);
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
        println!("✅ Model downloaded successfully");
        Ok(())
    } else {
        Err(anyhow::anyhow!("Download failed"))
    }
}
```

**Implement worker spawn:**
```rust
// src/commands/worker.rs

use std::process::Command;

pub fn spawn_worker(backend: &str, model: &str, gpu: u32, port: u16) -> anyhow::Result<()> {
    let worker_id = format!("worker-{}-{}", backend, gpu);
    let model_path = format!(".test-models/{}", model);
    
    // Find model file
    let model_file = std::fs::read_dir(&model_path)?
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map(|s| s == "gguf").unwrap_or(false))
        .ok_or_else(|| anyhow::anyhow!("No GGUF file found in {}", model_path))?;
    
    let model_file_path = model_file.path();
    
    println!("🚀 Spawning worker:");
    println!("   ID: {}", worker_id);
    println!("   Backend: {}", backend);
    println!("   Model: {}", model_file_path.display());
    println!("   GPU: {}", gpu);
    println!("   Port: {}", port);
    
    // Spawn rbees-workerd
    let mut cmd = Command::new("rbees-workerd");
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
                // Detach from parent
                libc::setsid();
                Ok(())
            });
        }
    }
    
    let child = cmd.spawn()?;
    
    println!("✅ Worker spawned (PID: {})", child.id());
    println!("   Waiting for worker to be ready...");
    
    // TODO: Wait for worker to register with orchestrator
    
    Ok(())
}
```

**Deliverable:** rbees-pool can download models and spawn workers

---

## Wednesday: rbees-ctl CLI

### Morning (4 hours)

**Create rbees-ctl binary:**
```bash
cd bin
mkdir rbees-ctl
cd rbees-ctl
cargo init --name rbees-ctl
```

**Add dependencies:**
```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
anyhow = "1"
colored = "2"
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
```

**Implement CLI skeleton:**
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

async fn handle_pool_action(action: PoolAction) -> anyhow::Result<()> {
    match action {
        PoolAction::Models { action } => match action {
            ModelsAction::Download { model, host } => {
                download_model_on_pool(&model, &host).await
            }
        },
        PoolAction::Worker { action } => match action {
            WorkerAction::Spawn { backend, model, host } => {
                spawn_worker_on_pool(&backend, &model, &host).await
            }
        },
    }
}

async fn handle_jobs_action(action: JobsAction) -> anyhow::Result<()> {
    match action {
        JobsAction::Submit { model, prompt } => submit_job(&model, &prompt).await,
        JobsAction::List => list_jobs().await,
    }
}
```

### Afternoon (4 hours)

**Implement SSH commands:**
```rust
// src/commands/pool.rs

use std::process::Command;

pub async fn download_model_on_pool(model: &str, host: &str) -> anyhow::Result<()> {
    println!("📥 Downloading {} on {}", model, host);
    
    let ssh_cmd = format!(
        "cd ~/Projects/llama-orch && rbees-pool models download {}",
        model
    );
    
    let status = Command::new("ssh")
        .args(&[host, &ssh_cmd])
        .status()?;
    
    if status.success() {
        println!("✅ Model downloaded on {}", host);
        Ok(())
    } else {
        Err(anyhow::anyhow!("SSH command failed"))
    }
}

pub async fn spawn_worker_on_pool(backend: &str, model: &str, host: &str) -> anyhow::Result<()> {
    println!("🚀 Spawning worker on {}", host);
    
    let ssh_cmd = format!(
        "cd ~/Projects/llama-orch && rbees-pool worker spawn {} --model {}",
        backend, model
    );
    
    let status = Command::new("ssh")
        .args(&[host, &ssh_cmd])
        .status()?;
    
    if status.success() {
        println!("✅ Worker spawned on {}", host);
        Ok(())
    } else {
        Err(anyhow::anyhow!("SSH command failed"))
    }
}
```

**Implement job submission:**
```rust
// src/commands/jobs.rs

use reqwest::Client;
use serde::{Deserialize, Serialize};

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
    
    println!("📤 Submitting job:");
    println!("   Model: {}", model);
    println!("   Prompt: {}", prompt);
    
    let response = client
        .post("http://localhost:8080/v2/tasks")
        .json(&TaskRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
        })
        .send()
        .await?;
    
    let task: TaskResponse = response.json().await?;
    
    println!("✅ Job submitted:");
    println!("   Job ID: {}", task.job_id);
    println!("   Status: {}", task.status);
    
    Ok(())
}

pub async fn list_jobs() -> anyhow::Result<()> {
    println!("📋 Jobs:");
    // TODO: Implement
    Ok(())
}
```

**Deliverable:** rbees-ctl can command pools via SSH and submit jobs

---

## Thursday: Integration & Testing

### Morning (4 hours)

**End-to-end test:**
```bash
# Terminal 1: Start rbees-orcd
cd bin/rbees-orcd
cargo run

# Terminal 2: Download model on mac
llorch pool models download tinyllama --host mac.home.arpa

# Terminal 3: Spawn worker on mac
llorch pool worker spawn metal --model tinyllama --host mac.home.arpa

# Terminal 4: Submit job
llorch jobs submit --model tinyllama --prompt "Write a haiku about Rust"
```

**Fix issues:**
- Worker registration not working
- Job dispatch failing
- SSH commands not executing

### Afternoon (4 hours)

**Add worker registration to rbees-workerd:**
```rust
// In rbees-workerd (existing binary)
// Add registration callback after model loads

async fn register_with_orchestrator(config: &Config) -> Result<()> {
    let client = reqwest::Client::new();
    
    let registration = serde_json::json!({
        "id": config.worker_id,
        "host": get_local_ip()?,
        "port": config.port,
        "model": config.model_path,
    });
    
    let orchestrator_url = env::var("LLORCH_ORCHESTRATOR_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_string());
    
    client
        .post(&format!("{}/workers/register", orchestrator_url))
        .json(&registration)
        .send()
        .await?;
    
    println!("✅ Registered with orchestrator");
    Ok(())
}
```

**Deliverable:** Full end-to-end flow working

---

## Friday: Polish & Documentation

### Morning (4 hours)

**Add error handling:**
- Better error messages
- Validation
- Retries

**Add logging:**
```rust
// In rbees-orcd
use tracing::{info, error};

info!("Job {} submitted", job_id);
error!("Worker {} failed: {}", worker_id, err);
```

**Add progress indicators:**
```rust
// In rbees-pool
use indicatif::ProgressBar;

let pb = ProgressBar::new_spinner();
pb.set_message("Downloading model...");
// ... download ...
pb.finish_with_message("✅ Downloaded");
```

### Afternoon (4 hours)

**Write README:**
```markdown
# llama-orch

EU-Native LLM Inference with Full Audit Trails

## Quick Start

### 1. Download a model
llorch pool models download tinyllama --host mac.home.arpa

### 2. Spawn a worker
llorch pool worker spawn metal --model tinyllama --host mac.home.arpa

### 3. Submit a job
llorch jobs submit --model tinyllama --prompt "Hello world"

## Architecture

- rbees-orcd: Job scheduling and worker management
- rbees-pool: Pool manager operations (local)
- rbees-ctl: Orchestrator operations (remote)
- rbees-workerd: Worker daemon (inference)
```

**Test everything one more time:**
```bash
# Clean slate
rm -rf .test-models

# Full flow
llorch pool models download tinyllama --host mac.home.arpa
llorch pool worker spawn metal --model tinyllama --host mac.home.arpa
llorch jobs submit --model tinyllama --prompt "Write a haiku"
```

**Deliverable:** Working MVP with documentation

---

## Success Criteria

By end of Friday:

- [ ] rbees-orcd accepts jobs and dispatches to workers
- [ ] rbees-pool downloads models and spawns workers
- [ ] rbees-ctl commands pools via SSH and submits jobs
- [ ] rbees-workerd registers with orchestrator
- [ ] End-to-end flow works: submit job → worker executes → tokens stream back
- [ ] Basic error handling and logging
- [ ] README with quick start guide

**Minimum viable, maximum shipping. Let's go! 🚀**

---

**Version**: 1.0  
**Last Updated**: 2025-10-09  
**Status**: Action Plan (START MONDAY)

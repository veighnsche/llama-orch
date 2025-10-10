# HANDOFF TO TEAM-043: Wire BDD Tests to Real Binaries (UPDATED AFTER SURVEY)

**From:** TEAM-042  
**To:** TEAM-043  
**Date:** 2025-10-10  
**Status:** 🟢 SURVEYED - READY FOR REAL IMPLEMENTATION

---

## What I Actually Found in `bin/`

I surveyed the codebase. Here's what **ALREADY EXISTS**:

### ✅ `bin/rbee-hive/` - FULLY IMPLEMENTED

**Worker Registry** (`src/registry.rs`):
- ✅ In-memory `WorkerRegistry` with full CRUD
- ✅ `WorkerInfo` struct with all fields (id, url, model_ref, backend, device, state, slots)
- ✅ Methods: `register()`, `update_state()`, `get()`, `list()`, `remove()`, `find_idle_worker()`
- ✅ Thread-safe with `Arc<RwLock<HashMap>>`
- ✅ Comprehensive unit tests

**Model Provisioner** (`src/provisioner/`):
- ✅ `ModelProvisioner` for downloading models
- ✅ `DownloadProgress` tracking
- ✅ Integration with model catalog

**HTTP Server** (`src/http/`):
- ✅ Full HTTP server with Axum
- ✅ Endpoints:
  - `GET /v1/health` - Health check
  - `POST /v1/workers/spawn` - Spawn worker
  - `POST /v1/workers/ready` - Worker ready callback
  - `GET /v1/workers/list` - List workers
  - `POST /v1/models/download` - Download model
  - `GET /v1/models/download/progress` - SSE progress stream

**Commands** (`src/commands/`):
- ✅ `daemon` - Start HTTP daemon
- ✅ `models` - Model management
- ✅ `worker` - Worker management
- ✅ `status` - Status check

### ✅ `bin/llm-worker-rbee/` - FULLY IMPLEMENTED

**Inference Backend** (`src/backend/`):
- ✅ Multiple model support (Llama, Mistral, Phi, Qwen, Quantized GGUF)
- ✅ Multiple backends (CPU, CUDA, Metal)
- ✅ Sampling, tokenization
- ✅ SSE streaming

**HTTP Server** (`src/http/`):
- ✅ Full HTTP server with Axum
- ✅ Endpoints:
  - `GET /health` - Health check
  - `POST /v1/inference` - Execute inference (SSE streaming)
  - `GET /v1/loading/progress` - Model loading progress (SSE)

**Narration** (`src/narration.rs`):
- ✅ Narration system for progress updates
- ✅ SSE event streaming

### ✅ `bin/rbee-keeper/` - PARTIALLY IMPLEMENTED

**CLI** (`src/cli.rs`):
- ✅ `Commands::Infer` - Inference command (FULLY IMPLEMENTED)
- ✅ `Commands::Pool` - Pool management commands
- ✅ `Commands::Install` - Installation command
- ❌ `Commands::Setup` - **MISSING** (needs to be added)

**Infer Command** (`src/commands/infer.rs`):
- ✅ Full 8-phase MVP flow implemented
- ✅ Connects to rbee-hive
- ✅ Spawns workers
- ✅ Waits for ready
- ✅ Executes inference
- ✅ SSE token streaming
- ✅ Error handling with retries

**Pool Client** (`src/pool_client.rs`):
- ✅ HTTP client for rbee-hive
- ✅ Methods: `health_check()`, `spawn_worker()`, etc.

### ❌ `bin/queen-rbee/` - SCAFFOLD ONLY

**Current State**:
- ❌ Just a scaffold with CLI arg parsing
- ❌ No HTTP server
- ❌ No registry module
- ❌ No SSH connection logic
- ❌ No orchestration logic

**What's Missing**:
- ❌ HTTP server on port 8080
- ❌ rbee-hive registry (SQLite at `~/.rbee/beehives.db`)
- ❌ SSH connection management
- ❌ Registry API endpoints (`/v2/registry/beehives/*`)
- ❌ Orchestration logic

### ✅ `bin/shared-crates/model-catalog/` - FULLY IMPLEMENTED

**Model Catalog** (`src/lib.rs`):
- ✅ SQLite-backed model tracking
- ✅ `ModelCatalog` with full CRUD
- ✅ Methods: `init()`, `find_model()`, `register_model()`, `remove_model()`, `list_models()`
- ✅ Comprehensive unit tests
- ✅ Used by rbee-hive

---

## What Needs to Be Implemented

### 🎯 Priority 1: queen-rbee Registry Module

**Create the registry module:**

```bash
mkdir -p bin/queen-rbee/src/registry
touch bin/queen-rbee/src/registry/mod.rs
touch bin/queen-rbee/src/registry/db.rs
touch bin/queen-rbee/src/registry/ssh.rs
touch bin/queen-rbee/src/registry/api.rs
```

**`bin/queen-rbee/src/registry/db.rs`** - SQLite operations:
```rust
use sqlx::{Connection, SqliteConnection};
use anyhow::Result;

pub struct BeehiveRegistry {
    db_path: String,
}

#[derive(Debug, Clone)]
pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: String,
    pub install_path: String,
    pub last_connected_unix: Option<i64>,
    pub status: String,
}

impl BeehiveRegistry {
    pub fn new(db_path: String) -> Self {
        Self { db_path }
    }

    pub async fn init(&self) -> Result<()> {
        // Create ~/.rbee directory
        if let Some(parent) = std::path::Path::new(&self.db_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut conn = SqliteConnection::connect(&format!("sqlite://{}?mode=rwc", self.db_path)).await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS beehives (
                node_name TEXT PRIMARY KEY,
                ssh_host TEXT NOT NULL,
                ssh_port INTEGER DEFAULT 22,
                ssh_user TEXT NOT NULL,
                ssh_key_path TEXT,
                git_repo_url TEXT NOT NULL,
                git_branch TEXT DEFAULT 'main',
                install_path TEXT NOT NULL,
                last_connected_unix INTEGER,
                status TEXT DEFAULT 'unknown'
            )
            "#,
        )
        .execute(&mut conn)
        .await?;

        Ok(())
    }

    pub async fn add_node(&self, node: &BeehiveNode) -> Result<()> {
        let mut conn = SqliteConnection::connect(&format!("sqlite://{}", self.db_path)).await?;

        sqlx::query(
            r#"
            INSERT INTO beehives 
            (node_name, ssh_host, ssh_port, ssh_user, ssh_key_path, 
             git_repo_url, git_branch, install_path, last_connected_unix, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&node.node_name)
        .bind(&node.ssh_host)
        .bind(node.ssh_port)
        .bind(&node.ssh_user)
        .bind(&node.ssh_key_path)
        .bind(&node.git_repo_url)
        .bind(&node.git_branch)
        .bind(&node.install_path)
        .bind(node.last_connected_unix)
        .bind(&node.status)
        .execute(&mut conn)
        .await?;

        Ok(())
    }

    pub async fn get_node(&self, node_name: &str) -> Result<Option<BeehiveNode>> {
        let mut conn = SqliteConnection::connect(&format!("sqlite://{}", self.db_path)).await?;

        let row = sqlx::query_as::<_, (String, String, i32, String, Option<String>, String, String, String, Option<i64>, String)>(
            r#"
            SELECT node_name, ssh_host, ssh_port, ssh_user, ssh_key_path,
                   git_repo_url, git_branch, install_path, last_connected_unix, status
            FROM beehives 
            WHERE node_name = ?
            "#,
        )
        .bind(node_name)
        .fetch_optional(&mut conn)
        .await?;

        Ok(row.map(|(node_name, ssh_host, ssh_port, ssh_user, ssh_key_path, git_repo_url, git_branch, install_path, last_connected_unix, status)| {
            BeehiveNode {
                node_name,
                ssh_host,
                ssh_port: ssh_port as u16,
                ssh_user,
                ssh_key_path,
                git_repo_url,
                git_branch,
                install_path,
                last_connected_unix,
                status,
            }
        }))
    }

    pub async fn list_nodes(&self) -> Result<Vec<BeehiveNode>> {
        let mut conn = SqliteConnection::connect(&format!("sqlite://{}", self.db_path)).await?;

        let rows = sqlx::query_as::<_, (String, String, i32, String, Option<String>, String, String, String, Option<i64>, String)>(
            r#"
            SELECT node_name, ssh_host, ssh_port, ssh_user, ssh_key_path,
                   git_repo_url, git_branch, install_path, last_connected_unix, status
            FROM beehives
            ORDER BY node_name
            "#,
        )
        .fetch_all(&mut conn)
        .await?;

        Ok(rows.into_iter().map(|(node_name, ssh_host, ssh_port, ssh_user, ssh_key_path, git_repo_url, git_branch, install_path, last_connected_unix, status)| {
            BeehiveNode {
                node_name,
                ssh_host,
                ssh_port: ssh_port as u16,
                ssh_user,
                ssh_key_path,
                git_repo_url,
                git_branch,
                install_path,
                last_connected_unix,
                status,
            }
        }).collect())
    }

    pub async fn remove_node(&self, node_name: &str) -> Result<()> {
        let mut conn = SqliteConnection::connect(&format!("sqlite://{}", self.db_path)).await?;

        sqlx::query("DELETE FROM beehives WHERE node_name = ?")
            .bind(node_name)
            .execute(&mut conn)
            .await?;

        Ok(())
    }

    pub async fn update_last_connected(&self, node_name: &str, timestamp: i64) -> Result<()> {
        let mut conn = SqliteConnection::connect(&format!("sqlite://{}", self.db_path)).await?;

        sqlx::query("UPDATE beehives SET last_connected_unix = ?, status = 'reachable' WHERE node_name = ?")
            .bind(timestamp)
            .bind(node_name)
            .execute(&mut conn)
            .await?;

        Ok(())
    }
}
```

**`bin/queen-rbee/src/registry/ssh.rs`** - SSH validation:
```rust
use anyhow::Result;
use tokio::process::Command;

pub async fn validate_ssh_connection(
    ssh_host: &str,
    ssh_user: &str,
    ssh_key_path: Option<&str>,
) -> Result<bool> {
    let mut cmd = Command::new("ssh");
    
    if let Some(key_path) = ssh_key_path {
        cmd.arg("-i").arg(key_path);
    }
    
    cmd.arg("-o")
        .arg("ConnectTimeout=10")
        .arg("-o")
        .arg("StrictHostKeyChecking=no")
        .arg(format!("{}@{}", ssh_user, ssh_host))
        .arg("echo 'connection test'");

    let output = cmd.output().await?;
    Ok(output.status.success())
}
```

**`bin/queen-rbee/src/registry/api.rs`** - HTTP endpoints:
```rust
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use super::db::{BeehiveRegistry, BeehiveNode};

#[derive(Clone)]
pub struct RegistryState {
    pub registry: Arc<BeehiveRegistry>,
}

#[derive(Deserialize)]
pub struct AddNodeRequest {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: Option<u16>,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: Option<String>,
    pub install_path: String,
}

#[derive(Serialize)]
pub struct AddNodeResponse {
    pub success: bool,
    pub message: String,
}

pub async fn handle_add_node(
    State(state): State<RegistryState>,
    Json(req): Json<AddNodeRequest>,
) -> impl IntoResponse {
    // Validate SSH connection
    match super::ssh::validate_ssh_connection(
        &req.ssh_host,
        &req.ssh_user,
        req.ssh_key_path.as_deref(),
    ).await {
        Ok(true) => {
            // SSH connection successful
            let node = BeehiveNode {
                node_name: req.node_name.clone(),
                ssh_host: req.ssh_host,
                ssh_port: req.ssh_port.unwrap_or(22),
                ssh_user: req.ssh_user,
                ssh_key_path: req.ssh_key_path,
                git_repo_url: req.git_repo_url,
                git_branch: req.git_branch.unwrap_or_else(|| "main".to_string()),
                install_path: req.install_path,
                last_connected_unix: Some(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64),
                status: "reachable".to_string(),
            };

            match state.registry.add_node(&node).await {
                Ok(_) => (
                    StatusCode::OK,
                    Json(AddNodeResponse {
                        success: true,
                        message: format!("Node '{}' added to registry", req.node_name),
                    }),
                ),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(AddNodeResponse {
                        success: false,
                        message: format!("Failed to add node: {}", e),
                    }),
                ),
            }
        }
        Ok(false) | Err(_) => (
            StatusCode::BAD_REQUEST,
            Json(AddNodeResponse {
                success: false,
                message: "SSH connection failed".to_string(),
            }),
        ),
    }
}

pub async fn handle_list_nodes(
    State(state): State<RegistryState>,
) -> impl IntoResponse {
    match state.registry.list_nodes().await {
        Ok(nodes) => (StatusCode::OK, Json(nodes)),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(vec![]),
        ),
    }
}

pub async fn handle_get_node(
    State(state): State<RegistryState>,
    Path(node_name): Path<String>,
) -> impl IntoResponse {
    match state.registry.get_node(&node_name).await {
        Ok(Some(node)) => (StatusCode::OK, Json(Some(node))),
        Ok(None) => (StatusCode::NOT_FOUND, Json(None)),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(None)),
    }
}

pub async fn handle_remove_node(
    State(state): State<RegistryState>,
    Path(node_name): Path<String>,
) -> impl IntoResponse {
    match state.registry.remove_node(&node_name).await {
        Ok(_) => (
            StatusCode::OK,
            Json(AddNodeResponse {
                success: true,
                message: format!("Node '{}' removed", node_name),
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(AddNodeResponse {
                success: false,
                message: format!("Failed to remove node: {}", e),
            }),
        ),
    }
}
```

**`bin/queen-rbee/src/main.rs`** - Wire it all together:
```rust
use anyhow::Result;
use axum::{
    routing::{get, post, delete},
    Router,
};
use std::sync::Arc;

mod registry;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize registry
    let registry_path = shellexpand::tilde("~/.rbee/beehives.db").to_string();
    let registry = Arc::new(registry::db::BeehiveRegistry::new(registry_path));
    registry.init().await?;

    let state = registry::api::RegistryState {
        registry: registry.clone(),
    };

    // Create router
    let app = Router::new()
        .route("/v2/registry/beehives/add", post(registry::api::handle_add_node))
        .route("/v2/registry/beehives/list", get(registry::api::handle_list_nodes))
        .route("/v2/registry/beehives/:name", get(registry::api::handle_get_node))
        .route("/v2/registry/beehives/:name", delete(registry::api::handle_remove_node))
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    println!("🐝 queen-rbee listening on http://0.0.0.0:8080");
    
    axum::serve(listener, app).await?;

    Ok(())
}
```

### 🎯 Priority 2: rbee-keeper Setup Commands

**Add to `bin/rbee-keeper/src/cli.rs`:**
```rust
#[derive(Subcommand)]
pub enum Commands {
    Infer { /* existing */ },
    Pool { /* existing */ },
    Install { /* existing */ },
    Setup {  // ADD THIS
        #[command(subcommand)]
        action: SetupAction,
    },
}

#[derive(Subcommand)]
pub enum SetupAction {
    AddNode {
        #[arg(long)]
        name: String,
        #[arg(long)]
        ssh_host: String,
        #[arg(long)]
        ssh_user: String,
        #[arg(long)]
        ssh_key: String,
        #[arg(long)]
        git_repo: String,
        #[arg(long)]
        git_branch: Option<String>,
        #[arg(long)]
        install_path: String,
    },
    Install {
        #[arg(long)]
        node: String,
    },
    ListNodes,
    RemoveNode {
        #[arg(long)]
        name: String,
    },
}
```

**Create `bin/rbee-keeper/src/commands/setup.rs`:**
```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct AddNodeRequest {
    node_name: String,
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
    ssh_key_path: String,
    git_repo_url: String,
    git_branch: String,
    install_path: String,
}

#[derive(Deserialize)]
struct AddNodeResponse {
    success: bool,
    message: String,
}

pub async fn handle_add_node(
    name: String,
    ssh_host: String,
    ssh_user: String,
    ssh_key: String,
    git_repo: String,
    git_branch: Option<String>,
    install_path: String,
) -> Result<()> {
    let client = reqwest::Client::new();
    
    let request = AddNodeRequest {
        node_name: name.clone(),
        ssh_host: ssh_host.clone(),
        ssh_port: 22,
        ssh_user,
        ssh_key_path: shellexpand::tilde(&ssh_key).to_string(),
        git_repo_url: git_repo,
        git_branch: git_branch.unwrap_or_else(|| "main".to_string()),
        install_path: shellexpand::tilde(&install_path).to_string(),
    };

    println!("[queen-rbee] 🔌 Testing SSH connection to {}", ssh_host);

    let response = client
        .post("http://localhost:8080/v2/registry/beehives/add")
        .json(&request)
        .send()
        .await?;

    let result: AddNodeResponse = response.json().await?;

    if result.success {
        println!("[queen-rbee] ✅ SSH connection successful! Node '{}' saved to registry", name);
        Ok(())
    } else {
        println!("[queen-rbee] ❌ {}", result.message);
        anyhow::bail!("Failed to add node");
    }
}

pub async fn handle_list_nodes() -> Result<()> {
    let client = reqwest::Client::new();
    
    let response = client
        .get("http://localhost:8080/v2/registry/beehives/list")
        .send()
        .await?;

    let nodes: Vec<serde_json::Value> = response.json().await?;

    println!("Registered rbee-hive Nodes:\n");
    for node in nodes {
        println!("{} ({})", node["node_name"], node["ssh_host"]);
        println!("  Status: {}", node["status"]);
        if let Some(ts) = node["last_connected_unix"].as_i64() {
            let dt = chrono::DateTime::from_timestamp(ts, 0).unwrap();
            println!("  Last connected: {}", dt.format("%Y-%m-%d %H:%M:%S"));
        }
        println!("  Install path: {}", node["install_path"]);
        println!();
    }

    Ok(())
}

pub async fn handle_remove_node(name: String) -> Result<()> {
    let client = reqwest::Client::new();
    
    let response = client
        .delete(&format!("http://localhost:8080/v2/registry/beehives/{}", name))
        .send()
        .await?;

    let result: AddNodeResponse = response.json().await?;

    if result.success {
        println!("[queen-rbee] ✅ Node '{}' removed from registry", name);
        Ok(())
    } else {
        println!("[queen-rbee] ❌ {}", result.message);
        anyhow::bail!("Failed to remove node");
    }
}
```

---

## How to Wire BDD Tests

### Step 1: Start Real Processes

**In `test-harness/bdd/src/steps/beehive_registry.rs`:**

Replace mocks with real process spawning:

```rust
#[given(expr = "queen-rbee is running")]
pub async fn given_queen_rbee_running(world: &mut World) {
    // Start queen-rbee as background process
    let mut child = tokio::process::Command::new("./target/debug/queen-rbee")
        .spawn()
        .expect("Failed to start queen-rbee");
    
    // Wait for HTTP server to be ready
    for _ in 0..30 {
        if reqwest::get("http://localhost:8080/v2/registry/beehives/list").await.is_ok() {
            world.queen_rbee_process = Some(child);
            tracing::info!("✅ queen-rbee started and ready");
            return;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    panic!("queen-rbee failed to start");
}
```

### Step 2: Execute Real Commands

```rust
#[when(expr = "I run:")]
pub async fn when_i_run_command(world: &mut World, step: &cucumber::gherkin::Step) {
    let command = step.docstring.as_ref().unwrap().trim();
    let parts: Vec<&str> = command.split_whitespace().collect();
    
    // Execute real command
    let output = tokio::process::Command::new("./target/debug/rbee-keeper")
        .args(&parts[1..]) // Skip "rbee-keeper"
        .output()
        .await
        .expect("Failed to execute command");
    
    world.last_stdout = String::from_utf8_lossy(&output.stdout).to_string();
    world.last_stderr = String::from_utf8_lossy(&output.stderr).to_string();
    world.last_exit_code = output.status.code();
}
```

### Step 3: Verify Real Database

```rust
#[then(expr = "queen-rbee saves node to rbee-hive registry:")]
pub async fn then_save_node_to_registry(world: &mut World, step: &cucumber::gherkin::Step) {
    // Query real database
    let db_path = shellexpand::tilde("~/.rbee/beehives.db");
    let conn = rusqlite::Connection::open(db_path.as_ref())
        .expect("Failed to open database");
    
    let mut stmt = conn.prepare("SELECT * FROM beehives WHERE node_name = ?")
        .expect("Failed to prepare query");
    
    let node_name = "mac"; // Extract from table
    let exists: bool = stmt.exists([node_name]).expect("Query failed");
    
    assert!(exists, "Node not found in database");
    tracing::info!("✅ Verified node in real database");
}
```

---

## Summary

### What Already Exists ✅
- **rbee-hive**: Fully implemented (worker registry, model provisioner, HTTP server)
- **llm-worker-rbee**: Fully implemented (inference, SSE streaming, multiple backends)
- **rbee-keeper infer**: Fully implemented (8-phase MVP flow)
- **model-catalog**: Fully implemented (SQLite model tracking)

### What's Missing ❌
- **queen-rbee**: Registry module, HTTP server, SSH validation
- **rbee-keeper setup**: Setup subcommands (add-node, list-nodes, remove-node, install)

### What You Need to Do 🎯
1. Implement queen-rbee registry module (Priority 1)
2. Implement rbee-keeper setup commands (Priority 2)
3. Wire BDD tests to real binaries (replace mocks with real execution)
4. Run tests and iterate until all pass

### My Mocks Are Still Useful 💡
- They show expected behavior
- They show data flow
- They show World state management
- Use them as implementation hints

---

**Good luck, TEAM-043! Now you know exactly what exists and what needs to be built.**

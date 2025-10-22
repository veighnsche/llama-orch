# TEAM-190 Implementation Guide - Part 1: Architecture & Patterns

**Based on TEAM-189 Implementation Experience**

## Table of Contents

1. System Architecture
2. The Four-Layer Pattern
3. Operation Flow
4. Common Patterns

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│ LAYER 1: CLI (rbee-keeper)                             │
│ - Parses user commands                                 │
│ - Creates Operation enum                               │
│ - Submits to queen-rbee via HTTP                       │
│ - Streams results via SSE                              │
│ - Shows final status (✅ Complete / ❌ Failed)          │
└─────────────────────────────────────────────────────────┘
                           ↓ HTTP POST + SSE
┌─────────────────────────────────────────────────────────┐
│ LAYER 2: Operations (rbee-operations)                  │
│ - Defines Operation enum variants                      │
│ - Serialization/deserialization                        │
│ - Operation name mapping                               │
└─────────────────────────────────────────────────────────┘
                           ↓ JSON
┌─────────────────────────────────────────────────────────┐
│ LAYER 3: Router (queen-rbee/job_router.rs)            │
│ - Pattern matches Operation variants                   │
│ - Executes business logic                             │
│ - Emits narration events                              │
│ - Returns success/error                                │
└─────────────────────────────────────────────────────────┘
                           ↓ Process spawning, HTTP calls
┌─────────────────────────────────────────────────────────┐
│ LAYER 4: Resources                                     │
│ - Database (HiveCatalog)                               │
│ - Daemons (rbee-hive)                                  │
│ - SSH connections                                      │
│ - File system                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 2. The Four-Layer Pattern

### Layer 1: CLI Command Definition

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
#[derive(Subcommand)]
pub enum HiveAction {
    /// Check hive status
    /// TEAM-XXX: Description of what this does
    Status {
        /// Hive ID (defaults to localhost)
        #[arg(default_value = "localhost")]
        id: String,
    },
}
```

**Key Points:**
- Use doc comments (`///`) for help text
- Add TEAM-XXX comment for tracking
- Use `default_value` for common defaults
- Keep parameter names consistent (`id`, not `hive_id`)

### Layer 2: Operation Definition

**File:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    /// TEAM-XXX: Brief description
    HiveStatus {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },
}
```

**Add to name() method:**
```rust
Operation::HiveStatus { .. } => constants::OP_HIVE_STATUS,
```

**Add to hive_id() extraction:**
```rust
Operation::HiveStatus { hive_id } => Some(hive_id),
```

**Add constant:**
```rust
pub mod constants {
    pub const OP_HIVE_STATUS: &str = "hive_status"; // TEAM-XXX
}
```

### Layer 3: CLI Routing

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
Commands::Hive { action } => {
    let operation = match action {
        HiveAction::Status { id } => Operation::HiveStatus { hive_id: id },
    };
    submit_and_stream_job(&client, &queen_url, operation).await
}
```

### Layer 4: Server Implementation

**File:** `bin/10_queen_rbee/src/job_router.rs`

```rust
pub async fn route_operation(operation: Operation, state: RouterState) -> Result<()> {
    match operation {
        Operation::HiveStatus { hive_id } => {
            // TEAM-XXX: Implementation with detailed comments
            
            // Step 1: Narrate what you're doing
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_status", &hive_id)
                .human(format!("🔍 Checking status for '{}'", hive_id))
                .emit();
            
            // Step 2: Do the work
            // ...
            
            // Step 3: Narrate the result
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_status_result", &hive_id)
                .human(format!("✅ Hive is {}", status))
                .emit();
        }
    }
    Ok(())
}
```

---

## 3. Operation Flow

### Complete Flow Example (from TEAM-189)

```
User runs: ./rbee hive status

1. CLI parses command → HiveAction::Status { id: "localhost" }
2. CLI creates Operation::HiveStatus { hive_id: "localhost" }
3. CLI POSTs to queen-rbee /v1/jobs with JSON
4. Queen creates job, returns job_id
5. CLI opens SSE stream /v1/jobs/{job_id}/stream
6. Queen executes route_operation()
   - Emits narration: "🔍 Checking status..."
   - Calls health endpoint
   - Emits narration: "✅ Hive is running"
7. CLI receives narration via SSE
8. CLI prints each narration line
9. Queen sends [DONE] marker
10. CLI shows "✅ Complete" or "❌ Failed"
```

---

## 4. Common Patterns

### Pattern 1: Pre-flight Checks

```rust
// Check if resource exists
let resource = state.catalog.get(&id).await?;
if resource.is_none() {
    Narration::new(ACTOR_QUEEN_ROUTER, "not_found", &id)
        .human(format!(
            "❌ Resource '{}' not found.\n\
             \n\
             To create it:\n\
             \n\
               ./rbee resource create --id {}",
            id, id
        ))
        .emit();
    return Err(anyhow::anyhow!("Resource '{}' not found", id));
}
```

### Pattern 2: Health Checks

```rust
let health_url = format!("http://{}:{}/health", host, port);
let client = reqwest::Client::builder()
    .timeout(tokio::time::Duration::from_secs(2))
    .build()?;

if let Ok(response) = client.get(&health_url).send().await {
    if response.status().is_success() {
        // Running
    }
} else {
    // Not running
}
```

### Pattern 3: Process Management

```rust
// Spawn daemon
use daemon_lifecycle::DaemonManager;
let manager = DaemonManager::new(
    std::path::PathBuf::from(&binary_path),
    vec!["--port".to_string(), port.to_string()],
);
let _child = manager.spawn().await?;

// Stop with SIGTERM
tokio::process::Command::new("pkill")
    .args(&["-TERM", binary_name])
    .output()
    .await?;

// Wait for shutdown
for attempt in 1..=5 {
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    if health_check_fails {
        break;
    }
    if attempt == 5 {
        // Force kill
        tokio::process::Command::new("pkill")
            .args(&["-KILL", binary_name])
            .output()
            .await?;
    }
}
```

### Pattern 4: Error Messages

**Good:**
```rust
"❌ Cannot uninstall hive 'localhost' while it's running.\n\
 \n\
 Please stop the hive first:\n\
 \n\
   ./rbee hive stop"
```

**Bad:**
```rust
"Error: hive is running"
```

**Rules:**
- Start with ❌ for errors, ⚠️ for warnings
- Blank line after problem statement
- Provide exact command to fix
- Use actual command syntax

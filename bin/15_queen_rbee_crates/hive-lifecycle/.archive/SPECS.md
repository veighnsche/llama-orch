# Hive Lifecycle - Complete Specifications

## Purpose

**Lifecycle management for rbee-hive instances** - starting, stopping, and checking status of hives on localhost or remote machines via SSH.

Queen orchestrates hive spawning - rbee-keeper does NOT.

## Core Responsibilities

### 1. Hive Orchestration
- Decide where to spawn hives (localhost vs remote)
- Allocate ports
- Manage hive processes

### 2. Catalog Integration
- Register hives in catalog before spawning
- Track hive metadata (host, port, SSH credentials)

### 3. Process Management
- Spawn hive processes (localhost)
- SSH-based spawning (remote) - **FUTURE**
- Process lifecycle tracking

### 4. Fire-and-Forget Pattern
- Spawn hive and return immediately
- Hive sends heartbeat when ready (callback)
- No blocking on hive startup

## Current Implementation

### ✅ Localhost Support
- Start hives on localhost
- Fixed port: 8600
- Direct process spawning
- Catalog registration

### 🚧 Future: Remote Support
- Start hives on remote machines via SSH
- Dynamic port allocation
- SSH tunneling
- Remote catalog management

## Data Model

### HiveStartRequest
```rust
pub struct HiveStartRequest {
    /// URL of the queen-rbee instance
    /// Example: "http://localhost:7800"
    pub queen_url: String,
}
```

### HiveStartResponse
```rust
pub struct HiveStartResponse {
    /// Full URL to access the hive
    /// Example: "http://localhost:8600"
    pub hive_url: String,
    
    /// Hive identifier (typically hostname)
    /// Example: "localhost"
    pub hive_id: String,
    
    /// Port the hive is listening on
    /// Example: 8600
    pub port: u16,
}
```

## Public API

### execute_hive_start()

```rust
pub async fn execute_hive_start(
    catalog: Arc<HiveCatalog>,
    request: HiveStartRequest,
) -> Result<HiveStartResponse>
```

**Purpose**: Start a new hive instance

**Pattern**: Command Pattern (fire-and-forget)

**Flow**:
1. Queen decides where to spawn (localhost for now)
2. Queen decides port (8600)
3. Queen adds hive to catalog (status: Unknown)
4. Queen spawns hive process (fire and forget)
5. Hive starts up asynchronously
6. **Hive sends heartbeat → Queen receives → Updates catalog → Triggers device detection**

**IMPORTANT**: This function does NOT wait for the hive to be ready!

**Parameters**:
- `catalog`: Hive catalog for persistence
- `request`: Hive start request (contains queen URL)

**Returns**:
- `Ok(HiveStartResponse)` - Hive spawn initiated (NOT necessarily running yet)
- `Err` - Failed to spawn hive

**Example**:
```rust
let catalog = Arc::new(HiveCatalog::new(Path::new("./data/hives.db")).await?);

let request = HiveStartRequest {
    queen_url: "http://localhost:7800".to_string(),
};

let response = execute_hive_start(catalog, request).await?;

println!("Hive spawn initiated: {}", response.hive_url);
println!("Hive ID: {}", response.hive_id);
println!("Waiting for heartbeat...");
// Hive will send heartbeat when ready
```

## Hive Lifecycle Flow

### Detailed Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        QUEEN (Orchestrator)                       │
└──────────────────────────────────────────────────────────────────┘
                               │
                               │ 1. User triggers hive start
                               ▼
                    ┌──────────────────────┐
                    │   execute_hive_start │
                    └──────────────────────┘
                               │
      ┌────────────────────────┼────────────────────────┐
      │                        │                        │
      ▼                        ▼                        ▼
┌───────────┐          ┌─────────────┐         ┌──────────────┐
│ DECIDE    │          │ DECIDE      │         │ ADD TO       │
│ Location  │          │ Port        │         │ CATALOG      │
│ localhost │          │ 8600        │         │ status:      │
│           │          │             │         │ Unknown      │
└───────────┘          └─────────────┘         └──────────────┘
                               │
                               │ 2. Spawn hive process
                               ▼
                    ┌──────────────────────┐
                    │   spawn_hive()       │
                    │   Fire & Forget!     │
                    └──────────────────────┘
                               │
                               │ 3. Return immediately
                               ▼
                    ┌──────────────────────┐
                    │  HiveStartResponse   │
                    │  (NOT running yet)   │
                    └──────────────────────┘
                               │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                               │
                               │ 4. Hive starts asynchronously
                               ▼
        ┌──────────────────────────────────────────────┐
        │              RBEE-HIVE (Spawned)              │
        └──────────────────────────────────────────────┘
                               │
                               │ 5. Hive initializes
                               │    - Load models
                               │    - Detect devices
                               │    - Start HTTP server
                               ▼
                    ┌──────────────────────┐
                    │   Hive ready!        │
                    │   Send heartbeat     │
                    └──────────────────────┘
                               │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                               │
                               │ 6. Heartbeat received
                               ▼
        ┌──────────────────────────────────────────────┐
        │           QUEEN (Heartbeat Handler)          │
        └──────────────────────────────────────────────┘
                               │
      ┌────────────────────────┼────────────────────────┐
      │                        │                        │
      ▼                        ▼                        ▼
┌───────────┐          ┌─────────────┐         ┌──────────────┐
│ UPDATE    │          │ UPDATE      │         │ TRIGGER      │
│ CATALOG   │          │ REGISTRY    │         │ DEVICE       │
│ status:   │          │ workers,    │         │ DETECTION    │
│ Online    │          │ resources   │         │              │
└───────────┘          └─────────────┘         └──────────────┘

CALLBACK MECHANISM: Heartbeat acts as the ready signal!
```

### State Transitions

```
Unknown → Online → Offline
   ↑        ↓         ↓
   └────────┴─────────┘
```

**Unknown**
- Initial state after `execute_hive_start()`
- Hive registered in catalog but not confirmed running

**Online**
- Hive sent heartbeat
- Confirmed running and responding

**Offline**
- No heartbeat received in threshold time
- Health check failed

## Integration Points

### With hive-catalog

```rust
// 1. Add hive before spawning
let hive = HiveRecord {
    id: "localhost".to_string(),
    host: "localhost".to_string(),
    port: 8600,
    status: HiveStatus::Unknown,  // Will be updated on heartbeat
    // ...
};
catalog.add_hive(hive).await?;

// 2. Spawn hive
spawn_hive(8600, &queen_url).await?;

// 3. Later: Heartbeat updates status
catalog.update_hive_status("localhost", HiveStatus::Online).await?;
```

### With hive-registry

```rust
// On heartbeat received
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<...> {
    // Update catalog (persistent)
    state.hive_catalog
        .update_heartbeat(&payload.hive_id, timestamp_ms)
        .await?;
    
    // Update registry (in-memory)
    state.hive_registry
        .update_hive_state(&payload.hive_id, payload);
    
    Ok(...)
}
```

### With health check

```rust
// Periodic health check
async fn health_check_task(catalog: Arc<HiveCatalog>) {
    loop {
        // Find stale hives (no heartbeat in 60s)
        let stale = catalog.find_stale_hives(60_000).await.unwrap();
        
        for hive in stale {
            // Mark as offline
            catalog.update_hive_status(&hive.id, HiveStatus::Offline).await.unwrap();
        }
        
        tokio::time::sleep(Duration::from_secs(30)).await;
    }
}
```

## Current vs Future

### Current Implementation (Localhost)

```rust
// Localhost spawning
let _child = Command::new("target/debug/rbee-hive")
    .arg("--port").arg("8600")
    .arg("--queen-url").arg(queen_url)
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .spawn()?;
```

**Limitations**:
- Fixed localhost
- Fixed port (8600)
- Single hive per machine
- No remote support

### Future Implementation (SSH Remote)

```rust
// Future: SSH remote spawning
pub async fn execute_hive_start_remote(
    catalog: Arc<HiveCatalog>,
    ssh_client: Arc<SshClient>,
    request: HiveStartRequestRemote,
) -> Result<HiveStartResponse> {
    // 1. Connect to remote machine via SSH
    let session = ssh_client.connect(&request.ssh_host).await?;
    
    // 2. Allocate port (dynamic)
    let port = allocate_port(&session).await?;
    
    // 3. Upload rbee-hive binary (if needed)
    upload_binary(&session).await?;
    
    // 4. Spawn hive on remote machine
    session.execute(&format!(
        "nohup ./rbee-hive --port {} --queen-url {} > /dev/null 2>&1 &",
        port, request.queen_url
    )).await?;
    
    // 5. Add to catalog with SSH info
    let hive = HiveRecord {
        id: request.ssh_host.clone(),
        host: request.ssh_host.clone(),
        port,
        ssh_host: Some(request.ssh_host.clone()),
        ssh_port: Some(22),
        ssh_user: Some(request.ssh_user.clone()),
        status: HiveStatus::Unknown,
        // ...
    };
    catalog.add_hive(hive).await?;
    
    Ok(HiveStartResponse {
        hive_url: format!("http://{}:{}", request.ssh_host, port),
        hive_id: request.ssh_host,
        port,
    })
}
```

## Command Pattern

The hive-lifecycle follows the Command Pattern (see `/bin/CRATE_INTERFACE_STANDARD.md`):

### Request/Response Structure
```rust
// Request type (input)
pub struct HiveStartRequest {
    pub queen_url: String,
}

// Response type (output)
pub struct HiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
}

// Entrypoint function
pub async fn execute_hive_start(
    catalog: Arc<HiveCatalog>,
    request: HiveStartRequest,
) -> Result<HiveStartResponse>
```

### Benefits
- Clear interface boundaries
- Easy to test
- Type-safe
- Self-documenting

## Usage Examples

### Example 1: Basic Hive Start
```rust
use queen_rbee_hive_lifecycle::{execute_hive_start, HiveStartRequest};
use queen_rbee_hive_catalog::HiveCatalog;
use std::sync::Arc;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize catalog
    let catalog = Arc::new(
        HiveCatalog::new(Path::new("./data/hives.db")).await?
    );
    
    // Create request
    let request = HiveStartRequest {
        queen_url: "http://localhost:7800".to_string(),
    };
    
    // Start hive (fire and forget)
    let response = execute_hive_start(catalog.clone(), request).await?;
    
    println!("✅ Hive spawn initiated");
    println!("   URL: {}", response.hive_url);
    println!("   ID: {}", response.hive_id);
    println!("   Port: {}", response.port);
    println!("   Waiting for heartbeat...");
    
    // Hive will send heartbeat when ready
    // This triggers:
    // 1. Catalog update (status → Online)
    // 2. Registry update (workers, resources)
    // 3. Device detection
    
    Ok(())
}
```

### Example 2: HTTP Handler Integration
```rust
use axum::{extract::State, Json};
use queen_rbee_hive_lifecycle::{execute_hive_start, HiveStartRequest};

#[derive(Clone)]
struct AppState {
    catalog: Arc<HiveCatalog>,
}

async fn handle_start_hive(
    State(state): State<AppState>,
    Json(req): Json<StartHiveRequest>,
) -> Result<Json<HiveStartResponse>, AppError> {
    let request = HiveStartRequest {
        queen_url: req.queen_url,
    };
    
    let response = execute_hive_start(state.catalog, request).await?;
    
    Ok(Json(response))
}
```

### Example 3: Monitoring Hive Startup
```rust
// Start hive
let response = execute_hive_start(catalog.clone(), request).await?;
println!("Hive spawn initiated: {}", response.hive_id);

// Poll catalog until hive is online
loop {
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    if let Some(hive) = catalog.get_hive(&response.hive_id).await? {
        match hive.status {
            HiveStatus::Online => {
                println!("✅ Hive is online!");
                break;
            }
            HiveStatus::Unknown => {
                println!("⏳ Waiting for hive...");
            }
            HiveStatus::Offline => {
                println!("❌ Hive failed to start");
                return Err(anyhow!("Hive startup failed"));
            }
        }
    }
}
```

## Error Handling

### Possible Errors

1. **Catalog Errors**
   - Database connection failed
   - Failed to add hive record
   - Constraint violations (duplicate hive ID)

2. **Process Spawn Errors**
   - Binary not found (`target/debug/rbee-hive`)
   - Permission denied
   - Port already in use
   - System resource limits

3. **Future: SSH Errors**
   - SSH connection failed
   - Authentication failed
   - Remote command execution failed
   - Network timeout

### Error Example
```rust
match execute_hive_start(catalog, request).await {
    Ok(response) => {
        println!("Hive spawned: {}", response.hive_url);
    }
    Err(e) => {
        eprintln!("Failed to start hive: {}", e);
        
        // Common errors:
        // - "Failed to spawn rbee-hive: No such file or directory"
        // - "Failed to add hive to catalog: UNIQUE constraint failed"
        // - "Address already in use"
    }
}
```

## Testing

### Unit Tests
```rust
#[tokio::test]
async fn test_hive_start_localhost() {
    let catalog = Arc::new(HiveCatalog::new_in_memory().await.unwrap());
    
    let request = HiveStartRequest {
        queen_url: "http://localhost:7800".to_string(),
    };
    
    let response = execute_hive_start(catalog.clone(), request).await.unwrap();
    
    assert_eq!(response.hive_id, "localhost");
    assert_eq!(response.port, 8600);
    
    // Check catalog
    let hive = catalog.get_hive("localhost").await.unwrap().unwrap();
    assert_eq!(hive.status, HiveStatus::Unknown);
}
```

### BDD Tests
See `/bin/15_queen_rbee_crates/hive-lifecycle/bdd/` for behavior-driven tests.

## Architecture Decisions

### Why Fire-and-Forget?

**Alternative**: Wait for hive to be ready
```rust
// BAD: Blocking approach
let response = execute_hive_start(catalog, request).await?;
wait_for_hive_ready(&response.hive_id).await?; // Blocks!
```

**Problems**:
- HTTP request times out
- Locks resources while waiting
- Complex timeout handling
- Can't start multiple hives in parallel

**Current**: Fire-and-forget with callback
```rust
// GOOD: Non-blocking approach
let response = execute_hive_start(catalog, request).await?;
// Returns immediately!
// Hive sends heartbeat when ready (callback)
```

**Benefits**:
- Instant response
- Non-blocking
- Natural parallelism
- Event-driven (heartbeat is the callback)

### Why Catalog Before Spawn?

**Reason**: If spawn fails, we know about it in catalog

**Flow**:
1. Add to catalog (status: Unknown)
2. Attempt spawn
3. If spawn fails → Catalog shows Unknown (can retry)
4. If spawn succeeds → Hive sends heartbeat → Catalog updated to Online

**Alternative** (spawn then add):
1. Spawn hive
2. Hive starts sending heartbeats
3. Add to catalog
4. **Problem**: Heartbeats arrive before catalog entry exists!

## Future Enhancements

### 1. Stop Hive
```rust
pub async fn execute_hive_stop(
    catalog: Arc<HiveCatalog>,
    registry: Arc<HiveRegistry>,
    hive_id: &str,
) -> Result<()>
```

**Flow**:
1. Get hive from catalog
2. Send shutdown signal to hive
3. Wait for graceful shutdown
4. Update catalog (status: Offline)
5. Remove from registry

### 2. Hive Status Query
```rust
pub async fn get_hive_status(
    catalog: Arc<HiveCatalog>,
    registry: Arc<HiveRegistry>,
    hive_id: &str,
) -> Result<HiveStatusResponse>

pub struct HiveStatusResponse {
    pub status: HiveStatus,
    pub uptime_ms: i64,
    pub worker_count: usize,
    pub last_heartbeat_ms: Option<i64>,
}
```

### 3. Restart Hive
```rust
pub async fn execute_hive_restart(
    catalog: Arc<HiveCatalog>,
    hive_id: &str,
) -> Result<HiveStartResponse>
```

### 4. Multi-Hive Start
```rust
pub async fn execute_hive_start_multi(
    catalog: Arc<HiveCatalog>,
    requests: Vec<HiveStartRequest>,
) -> Result<Vec<HiveStartResponse>>
```

### 5. Dynamic Port Allocation
```rust
pub async fn allocate_port(catalog: Arc<HiveCatalog>) -> Result<u16> {
    // Find unused port
    // Check catalog for existing ports
    // Return available port
}
```

### 6. SSH Remote Support
```rust
pub async fn execute_hive_start_ssh(
    catalog: Arc<HiveCatalog>,
    ssh_client: Arc<SshClient>,
    request: HiveStartRequestSsh,
) -> Result<HiveStartResponse>
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| `execute_hive_start()` | ~50ms | Catalog write + spawn |
| Hive startup time | ~2-5s | Loading models, devices |
| First heartbeat | ~3-6s | After startup |

## Dependencies

```toml
[dependencies]
anyhow = "1.0"
observability-narration-core = { path = "../../99_shared_crates/narration-core" }
queen-rbee-hive-catalog = { path = "../hive-catalog" }
tokio = { version = "1", features = ["full"] }
chrono = "0.4"

# Future
queen-rbee-ssh-client = { path = "../ssh-client" }  # For remote spawning
```

## Success Criteria

✅ Localhost hive spawning  
✅ Catalog integration  
✅ Fire-and-forget pattern  
✅ Heartbeat callback  
⏳ Remote SSH spawning (future)  
⏳ Stop hive (future)  
⏳ Status query (future)

## Related Crates

- **hive-catalog**: Persistent hive configuration
- **hive-registry**: Runtime hive state
- **ssh-client**: SSH operations (future)
- **health**: Health checking
- **preflight**: Pre-flight checks before spawn

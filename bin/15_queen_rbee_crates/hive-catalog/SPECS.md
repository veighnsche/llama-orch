# Hive Catalog - Complete Specifications

## Purpose

**Persistent storage (SQLite) for hive configuration and metadata.**

This is DIFFERENT from `hive-registry` (RAM - runtime state):
- **Catalog** = Persistent config (host, port, SSH, device capabilities) - **SURVIVES RESTARTS**
- **Registry** = Runtime state (workers, VRAM usage, last heartbeat) - **LOST ON RESTART**

## Core Responsibilities

### 1. Persistent Storage
- Store hive configuration that survives restarts
- SQLite database for reliability
- Schema versioning and migrations

### 2. Hive Metadata Management
- Connection info (host, port)
- SSH credentials for remote hives
- Device capabilities (CPU, GPUs, VRAM, RAM)
- Status tracking (Unknown, Online, Offline)
- Heartbeat timestamps

### 3. CRUD Operations
- **Create**: Add new hives
- **Read**: Get single hive or list all
- **Update**: Full or partial updates
- **Delete**: Remove hives

## Data Model

### HiveRecord
```rust
pub struct HiveRecord {
    /// Unique hive identifier (e.g., "localhost", "hive-prod-01")
    pub id: String,
    
    /// Hive host address (e.g., "127.0.0.1", "192.168.1.100")
    pub host: String,
    
    /// Hive HTTP port (e.g., 8600)
    pub port: u16,
    
    /// SSH host for remote hives (optional - only for network hives)
    pub ssh_host: Option<String>,
    
    /// SSH port (optional - only for network hives)
    pub ssh_port: Option<u16>,
    
    /// SSH username (optional - only for network hives)
    pub ssh_user: Option<String>,
    
    /// Current status
    pub status: HiveStatus,
    
    /// Last heartbeat timestamp (milliseconds since epoch)
    pub last_heartbeat_ms: Option<i64>,
    
    /// Device capabilities (CPU, GPUs)
    /// None = not yet detected
    pub devices: Option<DeviceCapabilities>,
    
    /// Creation timestamp (milliseconds since epoch)
    pub created_at_ms: i64,
    
    /// Last update timestamp (milliseconds since epoch)
    pub updated_at_ms: i64,
}
```

### HiveStatus
```rust
pub enum HiveStatus {
    /// Status unknown (freshly added, never seen)
    Unknown,
    
    /// Hive is online and responding
    Online,
    
    /// Hive is offline (not responding)
    Offline,
}
```

### DeviceCapabilities
```rust
pub struct DeviceCapabilities {
    /// CPU information
    pub cpu: Option<CpuDevice>,
    
    /// GPU list
    pub gpus: Vec<GpuDevice>,
}

pub struct CpuDevice {
    /// Number of CPU cores
    pub cores: u32,
    
    /// System RAM in GB
    pub ram_gb: u32,
}

pub struct GpuDevice {
    /// GPU index (0, 1, 2, ...)
    pub index: u32,
    
    /// GPU name (e.g., "NVIDIA RTX 4090")
    pub name: String,
    
    /// VRAM in GB
    pub vram_gb: u32,
}
```

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS hives (
    id                  TEXT PRIMARY KEY,
    host                TEXT NOT NULL,
    port                INTEGER NOT NULL,
    ssh_host            TEXT,
    ssh_port            INTEGER,
    ssh_user            TEXT,
    status              TEXT NOT NULL DEFAULT 'unknown',
    last_heartbeat_ms   INTEGER,
    devices_json        TEXT,        -- JSON serialized DeviceCapabilities
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hives_status ON hives(status);
CREATE INDEX IF NOT EXISTS idx_hives_heartbeat ON hives(last_heartbeat_ms);
```

## Public API Summary

**Total: 16 Public Methods**
- 1 Initialization
- 1 Create
- 2 Read
- 4 Update
- 1 Delete
- 8 Query operations

---

## Public API

### Initialization

#### new()
```rust
pub async fn new(db_path: &Path) -> Result<Self>
```
**Purpose**: Create catalog with SQLite database  
**Parameters**:
- `db_path`: Path to SQLite database file (created if missing)

**Example**:
```rust
let catalog = HiveCatalog::new(Path::new("./data/hives.db")).await?;
```

---

### CREATE

#### add_hive()
```rust
pub async fn add_hive(&self, hive: HiveRecord) -> Result<()>
```
**Purpose**: Add new hive to catalog  
**Use case**: Register new hive (local or remote)

**Example**:
```rust
let hive = HiveRecord {
    id: "localhost".to_string(),
    host: "127.0.0.1".to_string(),
    port: 8600,
    ssh_host: None,
    ssh_port: None,
    ssh_user: None,
    status: HiveStatus::Unknown,
    last_heartbeat_ms: None,
    devices: None,
    created_at_ms: chrono::Utc::now().timestamp_millis(),
    updated_at_ms: chrono::Utc::now().timestamp_millis(),
};

catalog.add_hive(hive).await?;
```

---

### READ

#### get_hive()
```rust
pub async fn get_hive(&self, id: &str) -> Result<Option<HiveRecord>>
```
**Purpose**: Get single hive by ID  
**Returns**: `Some(hive)` if found, `None` if not found

**Example**:
```rust
if let Some(hive) = catalog.get_hive("localhost").await? {
    println!("Hive: {}:{}", hive.host, hive.port);
}
```

#### list_hives()
```rust
pub async fn list_hives(&self) -> Result<Vec<HiveRecord>>
```
**Purpose**: Get all hives  
**Returns**: Vector of all hives (empty if none)

**Example**:
```rust
let hives = catalog.list_hives().await?;
for hive in hives {
    println!("Hive: {} - {:?}", hive.id, hive.status);
}
```

---

### UPDATE

#### update_hive()
```rust
pub async fn update_hive(&self, hive: HiveRecord) -> Result<()>
```
**Purpose**: Full update of hive record  
**Use case**: Updating multiple fields at once

**Example**:
```rust
let mut hive = catalog.get_hive("localhost").await?.unwrap();
hive.host = "192.168.1.100".to_string();
hive.port = 9000;
hive.status = HiveStatus::Online;
catalog.update_hive(hive).await?;
```

#### update_hive_status()
```rust
pub async fn update_hive_status(&self, id: &str, status: HiveStatus) -> Result<()>
```
**Purpose**: Update status only (lightweight)  
**Use case**: Status transitions after health checks

**Example**:
```rust
// Mark hive as online after successful health check
catalog.update_hive_status("localhost", HiveStatus::Online).await?;

// Mark as offline after failed health check
catalog.update_hive_status("localhost", HiveStatus::Offline).await?;
```

#### update_heartbeat()
```rust
pub async fn update_heartbeat(&self, id: &str, timestamp_ms: i64) -> Result<()>
```
**Purpose**: Update heartbeat timestamp only  
**Use case**: High-frequency heartbeat updates  
**Optimized**: Minimal database write

**Example**:
```rust
let now = chrono::Utc::now().timestamp_millis();
catalog.update_heartbeat("localhost", now).await?;
```

#### update_devices()
```rust
pub async fn update_devices(&self, id: &str, devices: DeviceCapabilities) -> Result<()>
```
**Purpose**: Update device capabilities only  
**Use case**: After device detection completes

**Example**:
```rust
let devices = DeviceCapabilities {
    cpu: Some(CpuDevice {
        cores: 16,
        ram_gb: 64,
    }),
    gpus: vec![
        GpuDevice {
            index: 0,
            name: "NVIDIA RTX 4090".to_string(),
            vram_gb: 24,
        },
    ],
};

catalog.update_devices("localhost", devices).await?;
```

---

### DELETE

#### remove_hive()
```rust
pub async fn remove_hive(&self, id: &str) -> Result<()>
```
**Purpose**: Remove hive from catalog  
**Use case**: Decommission hive

**Example**:
```rust
catalog.remove_hive("old-hive").await?;
```

---

### QUERY

#### find_hives_by_status()
```rust
pub async fn find_hives_by_status(&self, status: HiveStatus) -> Result<Vec<HiveRecord>>
```
**Purpose**: Find all hives with specific status  
**Use case**: Query by status

**Example**:
```rust
let online = catalog.find_hives_by_status(HiveStatus::Online).await?;
```

#### find_online_hives()
```rust
pub async fn find_online_hives(&self) -> Result<Vec<HiveRecord>>
```
**Purpose**: Convenience method for online hives  
**Use case**: Quick access to online hives

#### find_offline_hives()
```rust
pub async fn find_offline_hives(&self) -> Result<Vec<HiveRecord>>
```
**Purpose**: Convenience method for offline hives  
**Use case**: Quick access to offline hives

#### count_hives()
```rust
pub async fn count_hives(&self) -> Result<i64>
```
**Purpose**: Get total count of hives  
**Use case**: Statistics, monitoring

**Example**:
```rust
let total = catalog.count_hives().await?;
println!("Total hives: {}", total);
```

#### count_by_status()
```rust
pub async fn count_by_status(&self, status: HiveStatus) -> Result<i64>
```
**Purpose**: Count hives by status  
**Use case**: Statistics, monitoring

**Example**:
```rust
let online = catalog.count_by_status(HiveStatus::Online).await?;
let offline = catalog.count_by_status(HiveStatus::Offline).await?;
```

#### hive_exists()
```rust
pub async fn hive_exists(&self, id: &str) -> Result<bool>
```
**Purpose**: Check if hive exists  
**Use case**: Validation before operations

**Example**:
```rust
if catalog.hive_exists("my-hive").await? {
    // Safe to proceed
}
```

#### find_stale_hives()
```rust
pub async fn find_stale_hives(&self, max_age_ms: i64) -> Result<Vec<HiveRecord>>
```
**Purpose**: Find hives with old heartbeats  
**Use case**: Detect offline hives

**Example**:
```rust
// Find hives with no heartbeat in 60 seconds
let stale = catalog.find_stale_hives(60_000).await?;
for hive in stale {
    catalog.update_hive_status(&hive.id, HiveStatus::Offline).await?;
}
```

#### find_hives_with_devices()
```rust
pub async fn find_hives_with_devices(&self) -> Result<Vec<HiveRecord>>
```
**Purpose**: Find hives with device capabilities  
**Use case**: List fully configured hives

#### find_hives_without_devices()
```rust
pub async fn find_hives_without_devices(&self) -> Result<Vec<HiveRecord>>
```
**Purpose**: Find hives needing device detection  
**Use case**: Trigger device detection

**Example**:
```rust
let need_detection = catalog.find_hives_without_devices().await?;
for hive in need_detection {
    trigger_device_detection(&hive).await?;
}
```

---

## Usage Patterns

### Pattern 1: Register Local Hive
```rust
// Register localhost hive
let hive = HiveRecord {
    id: "localhost".to_string(),
    host: "127.0.0.1".to_string(),
    port: 8600,
    ssh_host: None,  // Local hive, no SSH
    ssh_port: None,
    ssh_user: None,
    status: HiveStatus::Unknown,
    last_heartbeat_ms: None,
    devices: None,  // Will be detected later
    created_at_ms: chrono::Utc::now().timestamp_millis(),
    updated_at_ms: chrono::Utc::now().timestamp_millis(),
};

catalog.add_hive(hive).await?;
```

### Pattern 2: Register Remote Hive
```rust
// Register remote hive via SSH
let hive = HiveRecord {
    id: "hive-prod-01".to_string(),
    host: "10.0.1.100".to_string(),
    port: 8600,
    ssh_host: Some("10.0.1.100".to_string()),
    ssh_port: Some(22),
    ssh_user: Some("admin".to_string()),
    status: HiveStatus::Unknown,
    last_heartbeat_ms: None,
    devices: None,
    created_at_ms: chrono::Utc::now().timestamp_millis(),
    updated_at_ms: chrono::Utc::now().timestamp_millis(),
};

catalog.add_hive(hive).await?;
```

### Pattern 3: Device Detection Flow
```rust
// 1. Register hive (no devices yet)
catalog.add_hive(hive).await?;

// 2. Start hive and wait for health check
// ...

// 3. Query hive for device capabilities
let devices = hive_http_client.get("/v1/devices").await?;

// 4. Update catalog with detected devices
catalog.update_devices(&hive.id, devices).await?;

// 5. Mark as online
catalog.update_hive_status(&hive.id, HiveStatus::Online).await?;
```

### Pattern 4: Heartbeat Update Flow
```rust
// Called by heartbeat handler
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<...> {
    let timestamp_ms = parse_timestamp(&payload.timestamp);
    
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

### Pattern 5: Health Check Flow
```rust
// Periodic health check task
async fn health_check_task(catalog: Arc<HiveCatalog>) {
    loop {
        let hives = catalog.list_hives().await.unwrap();
        
        for hive in hives {
            match http_client.get(&format!("http://{}:{}/health", hive.host, hive.port)).send().await {
                Ok(_) => {
                    catalog.update_hive_status(&hive.id, HiveStatus::Online).await.unwrap();
                }
                Err(_) => {
                    catalog.update_hive_status(&hive.id, HiveStatus::Offline).await.unwrap();
                }
            }
        }
        
        tokio::time::sleep(Duration::from_secs(30)).await;
    }
}
```

## Testing

### Test Coverage
```
running 12 tests
✅ test_create_catalog          - Initialization
✅ test_add_hive                - CREATE operation
✅ test_get_hive                - READ single
✅ test_get_hive_not_found      - READ single (not found)
✅ test_list_hives              - READ all
✅ test_update_hive             - UPDATE full
✅ test_update_status           - UPDATE partial (status)
✅ test_update_heartbeat        - UPDATE partial (heartbeat)
✅ test_update_devices          - UPDATE partial (devices)
✅ test_remove_hive             - DELETE
✅ test_device_serialization    - DeviceCapabilities JSON
✅ test_concurrent_access       - Thread safety

test result: ok. 12 passed; 0 failed
```

## Performance Characteristics

### Read Operations
- `get_hive()`: **~1ms** (indexed lookup)
- `list_hives()`: **~5ms** (100 hives)

### Write Operations
- `add_hive()`: **~2ms** (insert + indexes)
- `update_heartbeat()`: **~1ms** (single field update)
- `update_devices()`: **~2ms** (JSON serialization)

### Concurrency
- SQLite with WAL mode
- Multiple readers simultaneously
- Single writer at a time
- No locking issues in practice

## File Structure

```
hive-catalog/
├── Cargo.toml
├── SPECS.md (this file)
├── CRUD_PATTERN.md
├── README.md
├── src/
│   ├── lib.rs               # Public API exports
│   ├── catalog.rs           # Main CRUD implementation
│   ├── types.rs             # HiveRecord, HiveStatus
│   ├── device_types.rs      # DeviceCapabilities, CpuDevice, GpuDevice
│   ├── schema.rs            # Database schema initialization
│   ├── row_mapper.rs        # SQLite row → HiveRecord
│   └── heartbeat_traits.rs  # Trait for heartbeat handler
└── tests/
    └── integration_tests.rs
```

## Dependencies

```toml
[dependencies]
sqlx = { version = "0.8", features = ["sqlite", "runtime-tokio-rustls"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
chrono = "0.4"
```

## Catalog vs Registry Summary

| Feature | Catalog (SQLite) | Registry (RAM) |
|---------|------------------|----------------|
| **Persistence** | ✅ Survives restarts | ❌ Lost on restart |
| **Data** | Config, SSH, devices | Workers, VRAM usage, heartbeat |
| **Update Frequency** | Low (on changes) | High (every heartbeat) |
| **Read Speed** | ~1ms | <1μs |
| **Use Cases** | Config management | Real-time scheduling |
| **Size** | ~1KB per hive | ~10KB per hive |

## Success Criteria

✅ All CRUD operations implemented  
✅ SQLite persistence working  
✅ Device capabilities storage  
✅ 12 tests passing  
✅ Thread-safe operations  
✅ Documentation complete  
✅ Used by heartbeat handler  
✅ Used by hive lifecycle manager

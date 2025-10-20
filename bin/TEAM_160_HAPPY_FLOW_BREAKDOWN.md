# Happy Flow Breakdown: Add Localhost to Hive Catalog

**Source:** `/home/vince/Projects/llama-orch/bin/a_human_wrote_this.md`  
**Test:** `cargo xtask e2e:test`

---

## Overview

**User Command:** `rbee-keeper add-hive localhost`

**Flow:**
1. Keeper wakes queen (if asleep)
2. Queen spawns hive on localhost:8600
3. Hive sends first heartbeat
4. Queen detects unknown capabilities
5. Queen asks hive for device detection
6. Hive responds with CPU/GPU info
7. Queen stores in hive catalog
8. **Cascading shutdown:** Queen dies → Hive dies

**Key Point:** This is NOT an inference job - it's just registering a hive.

---

## Step-by-Step Breakdown

### Step 1: Keeper Checks if Queen is Running

**Crate:** `rbee-keeper` (bin)  
**Function:** Health check polling

```rust
// rbee-keeper/src/main.rs
async fn ensure_queen_running(queen_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let health_url = format!("{}/health", queen_url);
    
    // Try to ping queen
    match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => {
            println!("👑 Queen is awake and healthy");
            return Ok(());
        }
        _ => {
            println!("👑 Queen is asleep, waking queen...");
            spawn_queen()?;
            poll_until_healthy(&health_url).await?;
            println!("👑 Queen is awake and healthy");
        }
    }
    Ok(())
}
```

**Narration:**
- `(bee keeper -> stdout): queen is asleep, waking queen.`
- `(bee keeper): queen is awake and healthy.`

**Missing Crates:**
- ✅ `reqwest` - Already available
- ⚠️ Polling logic - Needs implementation

---

### Step 2: Keeper Spawns Queen (if needed)

**Crate:** `rbee-keeper` (bin)  
**Function:** Process spawning

```rust
// rbee-keeper/src/main.rs
fn spawn_queen() -> Result<Child> {
    // HARDCODED LOCATION FOR DEVELOPMENT
    let child = Command::new("target/debug/queen-rbee")
        .arg("--port")
        .arg("8500")
        .arg("--database")
        .arg("/tmp/queen.db")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    Ok(child)
}

async fn poll_until_healthy(health_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    for _ in 0..20 {
        match client.get(health_url).send().await {
            Ok(response) if response.status().is_success() => return Ok(()),
            _ => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
    anyhow::bail!("Queen failed to start")
}
```

**Narration:**
- `(bee keeper -> stdout): queen is asleep, waking queen.`
- `(bee keeper): queen is awake and healthy.`

---

### Step 3: Keeper Sends "Add Hive" Request to Queen

**Crate:** `rbee-keeper` (bin)  
**Function:** HTTP POST to queen

```rust
// rbee-keeper/src/main.rs
async fn add_hive_localhost(queen_url: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let add_hive_url = format!("{}/add-hive", queen_url);
    
    let request = serde_json::json!({
        "host": "localhost",
        "port": 8600
    });
    
    let response = client
        .post(&add_hive_url)
        .json(&request)
        .send()
        .await?;
    
    let hive_id = response.json::<serde_json::Value>().await?["hive_id"]
        .as_str()
        .unwrap()
        .to_string();
    
    Ok(hive_id)
}
```

**Narration:**
- `(bee keeper): Requesting queen to add localhost to hive catalog`

---

### Step 4: Queen Adds Localhost to Hive Catalog

**Crate:** `queen-rbee-hive-catalog` (lib)  
**Function:** SQLite database insert

```rust
// queen-rbee-hive-catalog/src/catalog.rs
impl HiveCatalog {
    pub async fn add_hive(&self, hive: HiveRecord) -> Result<()> {
        sqlx::query(
            "INSERT INTO hives (id, host, port, status, devices, created_at_ms, updated_at_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&hive.id)
        .bind(&hive.host)
        .bind(hive.port)
        .bind(hive.status.to_string())
        .bind(None::<String>) // devices unknown initially
        .bind(hive.created_at_ms)
        .bind(hive.updated_at_ms)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
}
```

**Narration:**
- `(queen bee -> sse -> bee keeper -> stdout): Adding local pc to hive catalog.`

**Missing Crates:**
- ✅ `queen-rbee-hive-catalog` - Already exists
- ⚠️ SSE connection - Needs implementation

---

### Step 5: Queen Spawns Rbee-Hive on Localhost

**Crate:** `queen-rbee` (bin)  
**Function:** Process spawning

```rust
// queen-rbee/src/hive_spawner.rs (NEW FILE)
pub fn spawn_hive_localhost(port: u16) -> Result<Child> {
    println!("👑 Waking up the bee hive at localhost");
    
    // HARDCODED LOCATION FOR DEVELOPMENT
    let child = Command::new("target/debug/rbee-hive")
        .arg("--port")
        .arg(port.to_string())
        .arg("--queen-url")
        .arg("http://localhost:8500")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    
    Ok(child)
}
```

**Narration:**
- `(queen bee): waking up the bee hive at localhost`

**Missing Crates:**
- ⚠️ `hive_spawner` module - Needs creation

---

### Step 6: Hive Sends First Heartbeat to Queen

**Crate:** `rbee-hive` (bin)  
**Function:** Periodic heartbeat sender

```rust
// rbee-hive/src/heartbeat_sender.rs (NEW FILE)
pub async fn start_heartbeat_task(queen_url: String, hive_id: String) {
    let client = reqwest::Client::new();
    let heartbeat_url = format!("{}/heartbeat", queen_url);
    
    loop {
        let payload = serde_json::json!({
            "hive_id": hive_id,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "workers": [] // Empty initially
        });
        
        let _ = client.post(&heartbeat_url).json(&payload).send().await;
        
        tokio::time::sleep(Duration::from_secs(15)).await;
    }
}
```

**Narration:**
- `(bee hive): Sending first heartbeat to queen`

**Missing Crates:**
- ⚠️ `heartbeat_sender` module - Needs creation

---

### Step 7: Queen Receives First Heartbeat

**Crate:** `queen-rbee` (bin)  
**Function:** Heartbeat handler

```rust
// queen-rbee/src/http/heartbeat.rs
pub async fn handle_hive_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HeartbeatAcknowledgement>, (StatusCode, String)> {
    // Check if this is first heartbeat (devices unknown)
    let hive = state.hive_catalog.get_hive(&payload.hive_id).await?;
    
    if hive.devices.is_none() {
        println!("👑 First heartbeat from {} - checking capabilities...", payload.hive_id);
        
        // Trigger device detection
        trigger_device_detection(&state, &payload.hive_id).await?;
    }
    
    // Update last_heartbeat timestamp
    state.hive_catalog.update_heartbeat(&payload.hive_id, now_ms).await?;
    
    Ok(Json(HeartbeatAcknowledgement { status: "ok" }))
}
```

**Narration:**
- `(queen bee): first heartbeat from a bee hive is received from localhost. checking its capabilities...`

**Missing Crates:**
- ⚠️ Device detection trigger - Needs implementation

---

### Step 8: Queen Asks Hive for Device Detection

**Crate:** `queen-rbee` (bin)  
**Function:** HTTP GET to hive

```rust
// queen-rbee/src/device_detector.rs
async fn trigger_device_detection(state: &AppState, hive_id: &str) -> Result<()> {
    let hive = state.hive_catalog.get_hive(hive_id).await?;
    let hive_url = format!("http://{}:{}", hive.host, hive.port);
    
    println!("👑 Unknown capabilities of beehive {}. Asking the beehive to detect devices", hive_id);
    
    let client = reqwest::Client::new();
    let devices_url = format!("{}/v1/devices", hive_url);
    
    let response = client.get(&devices_url).send().await?;
    let devices: DeviceCapabilities = response.json().await?;
    
    // Store in catalog
    state.hive_catalog.update_devices(hive_id, devices.clone()).await?;
    
    println!("👑 The beehive {} has: {:?}", hive_id, devices);
    
    Ok(())
}
```

**Narration:**
- `(queen bee): unknown capabilities of beehive localhost. asking the beehive to detect devices`

---

### Step 9: Hive Responds with Device Info

**Crate:** `rbee-hive` (bin)  
**Function:** Device detection endpoint

```rust
// rbee-hive/src/http/devices.rs
pub async fn handle_devices(
    State(state): State<AppState>,
) -> Result<Json<DeviceResponse>, (StatusCode, String)> {
    // Call device detection crate
    let devices = device_detection::detect_all_devices()?;
    
    let response = DeviceResponse {
        cpu: devices.cpu,
        gpus: devices.gpus,
        model_catalog: vec![], // Empty initially
        worker_catalog: vec![], // Empty initially
    };
    
    Ok(Json(response))
}
```

**Narration:**
- `(bee hive): Detected CPU and GPUs, responding to queen`

**Missing Crates:**
- ✅ `device-detection` - Already exists
- ⚠️ `/v1/devices` endpoint - Needs implementation

---

### Step 10: Queen Stores Devices in Catalog

**Crate:** `queen-rbee-hive-catalog` (lib)  
**Function:** SQLite update

```rust
// queen-rbee-hive-catalog/src/catalog.rs
impl HiveCatalog {
    pub async fn update_devices(&self, hive_id: &str, devices: DeviceCapabilities) -> Result<()> {
        let devices_json = serde_json::to_string(&devices)?;
        
        sqlx::query(
            "UPDATE hives SET devices = ?, status = 'Online', updated_at_ms = ? WHERE id = ?"
        )
        .bind(devices_json)
        .bind(chrono::Utc::now().timestamp_millis())
        .bind(hive_id)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
}
```

**Narration:**
- `(queen bee): the beehive localhost has a cpu gpu0 and 1 and blabla and model catalog has 0 models and 0 workers available`

---

### Step 11: Cascading Shutdown

**Crate:** `rbee-keeper` (bin)  
**Function:** Shutdown signal

```rust
// rbee-keeper/src/main.rs
async fn shutdown_queen(queen_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let shutdown_url = format!("{}/shutdown", queen_url);
    
    println!("🐝 Sending shutdown signal to queen");
    client.post(&shutdown_url).send().await?;
    
    Ok(())
}
```

**Queen Shutdown:**
```rust
// queen-rbee/src/http/shutdown.rs
pub async fn handle_shutdown(State(state): State<AppState>) -> StatusCode {
    println!("👑 Received shutdown signal, shutting down all hives...");
    
    // Get all hives from catalog
    let hives = state.hive_catalog.list_hives().await.unwrap();
    
    // Shutdown each hive
    for hive in hives {
        let shutdown_url = format!("http://{}:{}/shutdown", hive.host, hive.port);
        let _ = reqwest::Client::new().post(&shutdown_url).send().await;
    }
    
    println!("👑 Queen shutting down");
    std::process::exit(0);
}
```

**Narration:**
- `(bee keeper): Shutting down queen`
- `(queen bee): Shutting down all hives`
- `(bee hive): Received shutdown signal, goodbye`

---

## Crate Mapping

| Step | Crate | Status | Missing |
|------|-------|--------|---------|
| 1. Health check | `rbee-keeper` | ⚠️ Stub | Polling logic |
| 2. Spawn queen | `rbee-keeper` | ⚠️ Stub | Process management |
| 3. Add hive request | `rbee-keeper` | ⚠️ Stub | `/add-hive` endpoint |
| 4. Store in catalog | `queen-rbee-hive-catalog` | ✅ Exists | - |
| 5. Spawn hive | `queen-rbee` | ❌ Missing | `hive_spawner` module |
| 6. Send heartbeat | `rbee-hive` | ❌ Missing | `heartbeat_sender` module |
| 7. Receive heartbeat | `queen-rbee` | ⚠️ Partial | Device detection trigger |
| 8. Ask for devices | `queen-rbee` | ⚠️ Partial | HTTP client logic |
| 9. Detect devices | `rbee-hive` | ⚠️ Partial | `/v1/devices` endpoint |
| 10. Store devices | `queen-rbee-hive-catalog` | ✅ Exists | - |
| 11. Cascading shutdown | All | ❌ Missing | `/shutdown` endpoints |

---

## Test Command

```bash
cargo xtask e2e:test
```

**Expected Output:**
```
🚀 Starting E2E test: Add Localhost to Hive Catalog

🔨 Building binaries...
✅ All binaries built

👑 Starting queen-rbee on port 8500...
⏳ Waiting for queen to be ready...
✅ Queen ready after 3 attempts

🐝 Starting rbee-keeper on port 8400...
⏳ Waiting for keeper to be ready...
✅ Keeper ready after 2 attempts

📝 Adding localhost to hive catalog via keeper...
✅ Add hive request sent: localhost

⏳ Waiting for queen to spawn rbee-hive...
✅ Hive spawned after 5 attempts

🔍 Checking hive status...
✅ Hive status: Online

✅ E2E test PASSED
   Hive ID: localhost
   Hive Status: Online
   Queen and Hive will shutdown now (cascading shutdown)

🧹 Cleaning up processes...
✅ Cleanup complete
```

---

## Next Steps for Implementation

1. **rbee-keeper:**
   - Add health check polling
   - Add process spawning for queen
   - Add `/add-hive` command

2. **queen-rbee:**
   - Add `hive_spawner` module
   - Add device detection trigger
   - Add `/shutdown` endpoint

3. **rbee-hive:**
   - Add `heartbeat_sender` module
   - Implement `/v1/devices` endpoint
   - Add `/shutdown` endpoint

4. **All:**
   - Add SSE connections for narration
   - Add proper error handling
   - Add logging/tracing

---

**TEAM-160: Happy flow broken down into testable parts. Each step mapped to crates. 🚀**

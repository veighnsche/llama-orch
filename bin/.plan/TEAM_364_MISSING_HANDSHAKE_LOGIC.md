# TEAM-364: Missing Handshake Logic Analysis

**Date:** Oct 30, 2025  
**Status:** 🚨 CRITICAL GAP IDENTIFIED  
**Priority:** HIGH

---

## 🎯 PROBLEM STATEMENT

The **bidirectional discovery handshake** specified in `HEARTBEAT_ARCHITECTURE.md` is **NOT IMPLEMENTED**.

### **What's Specified (Canonical Spec)**

**File:** `bin/.specs/HEARTBEAT_ARCHITECTURE.md`

**Discovery Protocol:**
1. **Queen starts first** → Reads SSH config → Sends `GET /capabilities?queen_url=X` → Hive starts heartbeats
2. **Hive starts first** → Exponential backoff (5 tries: 0s, 2s, 4s, 8s, 16s) → Queen responds → Normal heartbeats
3. **Both start simultaneously** → Both mechanisms work, first success wins

### **What's Actually Implemented**

**Current State:**
1. ✅ Hive has `/capabilities` endpoint
2. ❌ `/capabilities` **does NOT accept `queen_url` parameter**
3. ❌ **No exponential backoff discovery** in hive
4. ❌ Hive just starts heartbeats immediately (assumes Queen is always there)
5. ❌ **No SSH config parsing** in Queen for discovery
6. ❌ Queen **does NOT send** `GET /capabilities?queen_url=X` to hives

---

## 📊 CURRENT IMPLEMENTATION

### **Hive Side**

**File:** `bin/20_rbee_hive/src/main.rs:184-242`

```rust
async fn get_capabilities() -> Json<CapabilitiesResponse> {
    n!("caps_request", "📡 Received capabilities request from queen");
    
    // Detect GPUs
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    
    // ... build devices list ...
    
    Json(CapabilitiesResponse { devices })
}
```

**Issues:**
- ❌ No `queen_url` query parameter extraction
- ❌ No storing of `queen_url`
- ❌ No triggering of heartbeat task
- ❌ Heartbeat task starts immediately on hive startup (assumes Queen exists)

**File:** `bin/20_rbee_hive/src/main.rs:149-153`

```rust
// Start heartbeat task (runs in background)
let _heartbeat_handle = heartbeat::start_heartbeat_task(hive_info, args.queen_url.clone());

n!("heartbeat_start", "💓 Heartbeat task started (sending to {})", args.queen_url);
```

**Issues:**
- ❌ Starts immediately (no exponential backoff)
- ❌ Runs forever (no "stop after 5 tries" logic)
- ❌ No discovery phase vs normal telemetry phase

### **Queen Side**

**File:** `bin/10_queen_rbee/src/main.rs`

**Issues:**
- ❌ No SSH config reading
- ❌ No hive discovery on startup
- ❌ No `GET /capabilities?queen_url=X` requests
- ❌ Just waits for hives to send heartbeats

---

## 🚨 CONSEQUENCES

### **Scenario 1: Hive Starts Before Queen**

**Current Behavior:**
1. Hive starts
2. Hive immediately starts sending heartbeats to `http://localhost:7833`
3. Queen not running → **Heartbeats fail forever**
4. Hive keeps trying forever (no exponential backoff, no stop after 5 tries)
5. **Logs fill up with errors**

**Expected Behavior (Per Spec):**
1. Hive starts
2. Hive sends 5 discovery heartbeats with exponential backoff (0s, 2s, 4s, 8s, 16s)
3. All fail → **Hive stops trying**
4. Hive waits for Queen to discover it via `/capabilities?queen_url=X`
5. When Queen starts and sends discovery request → Hive starts normal heartbeats

### **Scenario 2: Queen Starts Before Hive**

**Current Behavior:**
1. Queen starts
2. Queen does nothing (waits for heartbeats)
3. Hive starts later
4. Hive sends heartbeats
5. Works, but **Queen doesn't know about hive until first heartbeat arrives**

**Expected Behavior (Per Spec):**
1. Queen starts
2. Queen waits 5 seconds
3. **Queen reads SSH config** (list of hive hostnames)
4. **Queen sends `GET /capabilities?queen_url=http://queen:7833` to all hives**
5. Hive responds with capabilities
6. **Hive stores `queen_url` and starts heartbeats**
7. Queen knows about all hives immediately

### **Scenario 3: Both Start Simultaneously**

**Current Behavior:**
1. Both start
2. Hive sends heartbeats immediately
3. Queen not ready → Heartbeats fail
4. **Hive keeps failing forever**

**Expected Behavior (Per Spec):**
1. Both start
2. Hive tries exponential backoff (5 attempts)
3. Queen becomes ready after 5s
4. Queen sends discovery requests
5. **First success wins** (either backoff succeeds or discovery request arrives)
6. Normal operation begins

---

## 📋 MISSING COMPONENTS

### **1. SSH Config Parser (Shared Crate)**

**Status:** ❌ NOT EXTRACTED

**Current Location:** `bin/00_rbee_keeper/src/tauri_commands.rs:415-482`

**Function:** `ssh_list()`

**What it does:**
- Reads `~/.ssh/config`
- Parses SSH hosts
- Deduplicates by hostname
- Returns list of SSH targets

**Required:** Extract to `bin/99_shared_crates/ssh-config-parser`

**API Needed:**
```rust
pub struct SshTarget {
    pub host: String,      // Alias
    pub hostname: String,  // Actual hostname/IP
    pub user: String,
    pub port: u16,
}

pub fn parse_ssh_config(path: &Path) -> Result<Vec<SshTarget>>;
pub fn get_default_ssh_config_path() -> PathBuf;  // ~/.ssh/config
```

---

### **2. Hive Discovery (Queen Side)**

**Status:** ❌ NOT IMPLEMENTED

**Required File:** `bin/10_queen_rbee/src/discovery.rs` (NEW)

**What it needs to do:**
1. Wait 5 seconds after Queen startup
2. Read SSH config using shared crate
3. For each hive in SSH config:
   - Send `GET http://{hostname}:7835/capabilities?queen_url=http://queen:7833`
   - Store capabilities in HiveRegistry
4. Log results (discovered/failed)

**Pseudocode:**
```rust
pub async fn discover_hives_on_startup(
    ssh_config_path: &Path,
    queen_url: &str,
    hive_registry: Arc<HiveRegistry>,
) -> Result<()> {
    // Wait for services to stabilize
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Read SSH config
    let targets = ssh_config_parser::parse_ssh_config(ssh_config_path)?;
    
    // Discover all hives in parallel
    let mut tasks = vec![];
    for target in targets {
        let queen_url = queen_url.to_string();
        let hive_registry = Arc::clone(&hive_registry);
        
        tasks.push(tokio::spawn(async move {
            discover_single_hive(&target, &queen_url, hive_registry).await
        }));
    }
    
    // Wait for all discoveries
    for task in tasks {
        let _ = task.await;
    }
    
    Ok(())
}

async fn discover_single_hive(
    target: &SshTarget,
    queen_url: &str,
    hive_registry: Arc<HiveRegistry>,
) -> Result<()> {
    let url = format!(
        "http://{}:7835/capabilities?queen_url={}",
        target.hostname,
        urlencoding::encode(queen_url)
    );
    
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    
    if response.status().is_success() {
        let capabilities: CapabilitiesResponse = response.json().await?;
        hive_registry.register_hive(target.host.clone(), capabilities);
        tracing::info!("Discovered hive: {}", target.host);
    }
    
    Ok(())
}
```

---

### **3. Enhanced Capabilities Endpoint (Hive Side)**

**Status:** ❌ PARTIALLY IMPLEMENTED (missing queen_url handling)

**Current:** `bin/20_rbee_hive/src/main.rs:184-242`

**Needs:**
1. Accept `queen_url` query parameter
2. Store `queen_url` in shared state
3. Trigger heartbeat task (if not already running)

**Required Changes:**
```rust
#[derive(Debug, Deserialize)]
struct CapabilitiesQuery {
    queen_url: Option<String>,
}

async fn get_capabilities(
    Query(params): Query<CapabilitiesQuery>,
    State(state): State<Arc<HiveState>>,  // Need shared state
) -> Json<CapabilitiesResponse> {
    n!("caps_request", "📡 Received capabilities request");
    
    // TEAM-364: Handle queen_url parameter
    if let Some(queen_url) = params.queen_url {
        n!("caps_queen_url", "🔗 Queen URL received: {}", queen_url);
        
        // Store queen_url and start heartbeat task
        state.set_queen_url(queen_url.clone()).await;
        state.start_heartbeat_task(queen_url).await;
    }
    
    // Detect devices...
    let devices = detect_devices();
    
    Json(CapabilitiesResponse { devices })
}
```

---

### **4. Exponential Backoff Discovery (Hive Side)**

**Status:** ❌ NOT IMPLEMENTED

**Required:** Modify `bin/20_rbee_hive/src/heartbeat.rs`

**What it needs:**
1. Discovery phase with exponential backoff
2. Stop after 5 failed attempts
3. Transition to normal telemetry on first success

**Pseudocode:**
```rust
pub async fn start_discovery_with_backoff(
    hive_info: HiveInfo,
    queen_url: String,
) -> Result<()> {
    let delays = [0, 2, 4, 8, 16];  // Exponential backoff in seconds
    
    for (attempt, delay) in delays.iter().enumerate() {
        if *delay > 0 {
            tokio::time::sleep(Duration::from_secs(*delay)).await;
        }
        
        tracing::info!("Discovery attempt {} (delay: {}s)", attempt + 1, delay);
        
        // Send discovery heartbeat (same format as normal)
        match send_heartbeat_to_queen(&hive_info, &queen_url).await {
            Ok(_) => {
                tracing::info!("Discovery successful! Starting normal telemetry");
                // Start normal telemetry task
                start_normal_telemetry_task(hive_info, queen_url).await;
                return Ok(());
            }
            Err(e) => {
                tracing::warn!("Discovery attempt {} failed: {}", attempt + 1, e);
            }
        }
    }
    
    // All 5 attempts failed
    tracing::warn!("All discovery attempts failed. Waiting for Queen to discover us via /capabilities");
    Ok(())
}

async fn start_normal_telemetry_task(hive_info: HiveInfo, queen_url: String) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            if let Err(e) = send_heartbeat_to_queen(&hive_info, &queen_url).await {
                tracing::warn!("Heartbeat failed: {}", e);
            }
        }
    });
}
```

---

### **5. Shared State for Queen URL (Hive Side)**

**Status:** ❌ NOT IMPLEMENTED

**Required:** Shared state to store `queen_url` dynamically

**Pseudocode:**
```rust
pub struct HiveState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub worker_catalog: Arc<WorkerCatalog>,
    pub queen_url: Arc<RwLock<Option<String>>>,  // TEAM-364: Dynamic queen URL
    pub heartbeat_running: Arc<AtomicBool>,      // TEAM-364: Prevent duplicate tasks
}

impl HiveState {
    pub async fn set_queen_url(&self, url: String) {
        *self.queen_url.write().await = Some(url);
    }
    
    pub async fn start_heartbeat_task(&self, queen_url: String) {
        // Only start if not already running
        if self.heartbeat_running.swap(true, Ordering::SeqCst) {
            return;  // Already running
        }
        
        let hive_info = self.get_hive_info();
        start_normal_telemetry_task(hive_info, queen_url).await;
    }
}
```

---

## 🎯 IMPLEMENTATION PLAN

### **Phase 1: Extract SSH Config Parser (1 hour)**

- [ ] Create `bin/99_shared_crates/ssh-config-parser`
- [ ] Extract logic from `rbee-keeper/src/tauri_commands.rs:415-482`
- [ ] Add tests
- [ ] Update rbee-keeper to use shared crate

### **Phase 2: Implement Queen Discovery (2 hours)**

- [ ] Create `bin/10_queen_rbee/src/discovery.rs`
- [ ] Implement `discover_hives_on_startup()`
- [ ] Call from `main.rs` after 5-second wait
- [ ] Add logging/narration

### **Phase 3: Enhance Hive Capabilities Endpoint (1 hour)**

- [ ] Add `queen_url` query parameter handling
- [ ] Add shared state for dynamic queen URL
- [ ] Trigger heartbeat task on discovery
- [ ] Prevent duplicate heartbeat tasks

### **Phase 4: Implement Exponential Backoff (2 hours)**

- [ ] Add discovery phase to heartbeat.rs
- [ ] Implement 5-attempt exponential backoff
- [ ] Stop after failures
- [ ] Transition to normal telemetry on success

### **Phase 5: Integration Testing (1 hour)**

- [ ] Test Scenario 1: Hive starts first
- [ ] Test Scenario 2: Queen starts first
- [ ] Test Scenario 3: Both start simultaneously
- [ ] Verify no duplicate heartbeats
- [ ] Verify logs are clean

**Total Estimated Time:** 7 hours

---

## 📝 ACCEPTANCE CRITERIA

### **Scenario 1: Hive Starts First**
- [ ] Hive sends 5 discovery heartbeats with exponential backoff
- [ ] Hive stops after 5 failures
- [ ] Hive waits silently for Queen discovery
- [ ] When Queen starts and sends `/capabilities?queen_url=X`, hive starts normal heartbeats
- [ ] No error spam in logs

### **Scenario 2: Queen Starts First**
- [ ] Queen waits 5 seconds
- [ ] Queen reads SSH config
- [ ] Queen sends `GET /capabilities?queen_url=X` to all hives
- [ ] Hives respond and start heartbeats
- [ ] Queen knows about all hives immediately

### **Scenario 3: Both Start Simultaneously**
- [ ] Both mechanisms work in parallel
- [ ] First success wins (no duplicate heartbeats)
- [ ] Normal operation begins smoothly

---

## 🚨 PRIORITY JUSTIFICATION

**Why This Is Critical:**

1. **Production Readiness** - Current implementation breaks if startup order is wrong
2. **User Experience** - Error spam in logs when Queen not running
3. **Reliability** - No graceful handling of startup races
4. **Spec Compliance** - Canonical spec exists but not implemented

**Impact:**
- **High** - Affects all deployments
- **Frequency** - Happens on every startup
- **Severity** - System works but logs are messy, discovery is fragile

---

## 📚 RELATED DOCUMENTS

- **Canonical Spec:** `bin/.specs/HEARTBEAT_ARCHITECTURE.md`
- **Current Hive:** `bin/20_rbee_hive/src/main.rs`
- **Current Queen:** `bin/10_queen_rbee/src/main.rs`
- **SSH Config Logic:** `bin/00_rbee_keeper/src/tauri_commands.rs:415-482`
- **Heartbeat Logic:** `bin/20_rbee_hive/src/heartbeat.rs`

---

**Status:** 🚨 CRITICAL GAP IDENTIFIED  
**Next Action:** Extract SSH config parser (Phase 1)  
**Estimated Total Effort:** 7 hours

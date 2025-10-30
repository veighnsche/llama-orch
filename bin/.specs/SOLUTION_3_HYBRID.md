# Solution 3: Hybrid (Registry + Parent-Child)

**Approach:** Registry Service + Spawner Notification  
**Complexity:** MEDIUM-HIGH  
**Recommended:** ⭐⭐⭐⭐⭐ **BEST OF BOTH WORLDS**

---

## 🎯 Core Idea

Combine Solution 1 (Registry Service) with Solution 2 (Parent-Child Registration) to get the best of both:

- **Registry Service** - Solves discovery for all scenarios
- **Parent-Child Registration** - Optimizes the common case (fast notification)

**Key Insight:** Use registry for discovery, use parent-child for optimization.

---

## 🏗️ Architecture

```
┌──────────────────┐
│  rbee-registry   │  Port: 7830
│  (Discovery)     │
└────────┬─────────┘
         │
         │ All components register here
         │
    ┌────┼────┬────────┬─────────┬─────────┐
    │    │    │        │         │         │
    ▼    ▼    ▼        ▼         ▼         ▼
┌────┐ ┌────┐ ┌────┐ ┌────┐  ┌────┐   ┌────┐
│GUI │ │Hive│ │Work│ │Work│  │Queen│  │Work│
│    │ │    │ │er1 │ │er2 │  │    │  │er3 │
└─┬──┘ └─┬──┘ └────┘ └────┘  └─┬───┘  └────┘
  │      │                      │
  │      │ Spawner also tells   │
  │      │ Queen directly       │
  └──────┴──────────────────────┘
         (OPTIMIZATION)
```

---

## 📋 Component Responsibilities

### **rbee-registry (Port 7830)**

**Purpose:** Discovery (same as Solution 1)

**API:**
```
POST   /v1/registry/register     - Register component
DELETE /v1/registry/unregister   - Unregister component
POST   /v1/registry/heartbeat    - Keep-alive
GET    /v1/registry/hives        - List all hives
GET    /v1/registry/workers      - List all workers
GET    /v1/registry/queen        - Get queen info
GET    /v1/registry/components   - List all components
```

---

### **Queen (Port 7833)**

**New APIs:**
```
POST /v1/queen/register-worker     - Spawner notifies Queen (OPTIMIZATION)
POST /v1/queen/register-hive       - Spawner notifies Queen (OPTIMIZATION)
```

**On Startup:**
```rust
// 1. Register self with registry
register_with_registry().await?;

// 2. Discover all components from registry
let hives = query_registry_hives().await?;
let workers = query_registry_workers().await?;

// 3. Fetch capabilities from each
for hive in hives {
    fetch_hive_capabilities(&hive).await?;
}
for worker in workers {
    fetch_worker_capabilities(&worker).await?;
}

// 4. Start registry heartbeat
start_registry_heartbeat_task().await;
```

**On Worker Registration (Fast Path):**
```rust
// Spawner notified Queen directly
async fn handle_register_worker(worker: WorkerRegistration) {
    // 1. Store in WorkerRegistry immediately
    worker_registry.register(worker).await;
    
    // 2. Fetch capabilities
    let caps = fetch_worker_capabilities(&worker).await?;
    
    // 3. Update registry
    worker_registry.update_capabilities(worker.id, caps).await;
    
    // Result: Queen knows about worker instantly (no need to query registry)
}
```

---

### **GUI/Hive (Spawners)**

**When Spawning Worker:**
```rust
async fn spawn_worker(...) {
    let port = find_available_port(8080..8200)?;
    let worker_process = spawn_worker_process(..., port).await?;
    
    // Wait for worker to be ready
    wait_for_worker_health(port).await?;
    
    let worker_info = WorkerInfo {
        worker_id: format!("worker-{}-{}", worker_type, port),
        hostname: "localhost",
        port,
        worker_type,
        model: Some(model),
        spawned_by: "gui",  // or "hive-{id}"
    };
    
    // 1. Register with registry (REQUIRED - for discovery)
    register_with_registry(worker_info).await?;
    
    // 2. Notify Queen directly (OPTIMIZATION - if Queen exists)
    if let Some(queen_url) = try_get_queen_url().await {
        // Fast path: Tell Queen immediately
        let _ = notify_queen_of_worker(&queen_url, worker_info).await;
        // (Ignore error - Queen will discover via registry anyway)
    }
    
    // 3. Store in local registry (for GUI's own use)
    local_workers.insert(worker_id, worker_info);
}
```

---

### **Worker**

**On Startup:**
```rust
// 1. Register with registry
register_with_registry().await?;

// 2. Start registry heartbeat (15s)
start_registry_heartbeat_task().await;

// 3. Query registry for Queen (if needed)
if let Some(queen) = query_registry_for_queen().await? {
    // 4. Start Queen heartbeat (30s)
    start_queen_heartbeat_task(queen.url).await;
}

// If Queen doesn't exist yet, that's fine!
// Queen will discover us via registry when it starts
```

---

## 🔄 Discovery Flows

### **Flow 1: Queen Starts First (Normal Case)**

```
1. Queen starts
2. Queen registers with registry
3. Queen queries registry (finds no workers yet)
4. GUI spawns Worker on port 8080
5. Worker registers with registry
6. GUI notifies Queen: POST /v1/queen/register-worker (FAST PATH)
7. Queen stores worker immediately
8. Queen fetches worker capabilities
9. Worker starts Queen heartbeat
10. ✅ Queen discovers worker instantly!
```

---

### **Flow 2: Worker Starts Before Queen (Queenless)**

```
1. GUI spawns Worker on port 8080
2. Worker registers with registry
3. GUI tries to notify Queen → 404 (no Queen yet)
4. GUI stores worker locally
5. User does inference via GUI → Worker (Queenless operation)
6. Queen starts later
7. Queen queries registry
8. Queen discovers Worker! (via registry)
9. Queen fetches worker capabilities
10. Worker queries registry, finds Queen
11. Worker starts Queen heartbeat
12. ✅ Queen discovers pre-existing worker!
```

---

### **Flow 3: Multiple Workers, Mixed Startup**

```
1. GUI spawns Worker 1 (8080)
2. Worker 1 registers with registry
3. Queen starts
4. Queen queries registry, discovers Worker 1
5. Hive spawns Worker 2 (8081)
6. Worker 2 registers with registry
7. Hive notifies Queen directly (FAST PATH)
8. Queen discovers Worker 2 instantly
9. GUI spawns Worker 3 (8082)
10. Worker 3 registers with registry
11. GUI notifies Queen directly (FAST PATH)
12. Queen discovers Worker 3 instantly
13. ✅ All workers discovered (mixed via registry + notification)
```

---

## ✅ Advantages

### **1. Solves ALL Requirements**

- ✅ Queen discovers hives (via registry)
- ✅ Queen discovers workers (via registry)
- ✅ Works if Queen starts first
- ✅ Works if Workers start first (Queenless)
- ✅ Works with dynamic worker ports
- ✅ No Hive worker registry
- ✅ No Hive worker heartbeats
- ✅ Works cross-machine
- ✅ Systematic (registry is source of truth)

### **2. Fast Path Optimization**

- Spawner notifies Queen directly when possible
- Queen doesn't have to wait for next registry query
- Worker available immediately

### **3. Resilient**

- If fast path fails, registry provides fallback
- If registry is down, components cache last-known info
- Self-healing (components re-register when registry recovers)

### **4. GUI-Friendly**

- GUI can operate Queenless
- GUI can spawn workers anytime
- GUI can query registry to see all components

---

## ❌ Disadvantages

### **1. More Complexity**

- Two mechanisms (registry + notification)
- More code to maintain
- More coordination

### **2. Still Requires Registry**

- Registry must start first (or components retry)
- Registry is single point of failure for discovery
- (Mitigation: components cache, keep working)

### **3. Potential Double-Registration**

- Spawner notifies Queen
- Worker also registers with registry
- (Mitigation: Queen deduplicates, registry is source of truth)

---

## 🛠️ Implementation

### **Phase 1: Registry Service (Same as Solution 1)**

- Create `bin/05_rbee_registry`
- Implement registration/heartbeat/query APIs
- Create `bin/99_shared_crates/registry-client`

---

### **Phase 2: Queen Notification API**

**File:** `bin/10_queen_rbee/src/http/registration.rs` (NEW)

```rust
#[derive(Deserialize)]
pub struct WorkerRegistration {
    pub worker_id: String,
    pub hostname: String,
    pub port: u16,
    pub worker_type: String,
    pub model: Option<String>,
    pub spawned_by: String,
}

/// POST /v1/queen/register-worker
/// Fast path: Spawner notifies Queen about new worker
pub async fn handle_register_worker(
    State(state): State<QueenState>,
    Json(registration): Json<WorkerRegistration>,
) -> Result<StatusCode> {
    // 1. Store in WorkerRegistry
    state.worker_registry.register_worker(registration.clone()).await?;
    
    // 2. Fetch capabilities (async)
    let caps_url = format!("http://{}:{}/capabilities", 
        registration.hostname, registration.port);
    
    tokio::spawn(async move {
        if let Ok(caps) = fetch_capabilities(&caps_url).await {
            state.worker_registry.update_capabilities(
                &registration.worker_id, 
                caps
            ).await;
        }
    });
    
    Ok(StatusCode::OK)
}
```

---

### **Phase 3: Update Spawners**

**GUI/Hive Spawner:**

```rust
async fn spawn_worker(...) {
    // ... spawn worker process ...
    
    // 1. Register with registry (REQUIRED)
    registry_client.register(worker_info).await?;
    
    // 2. Notify Queen (OPTIMIZATION)
    if let Some(queen_url) = get_queen_url_from_registry().await {
        let _ = reqwest::Client::new()
            .post(format!("{}/v1/queen/register-worker", queen_url))
            .json(&worker_info)
            .send()
            .await;
        // Ignore error - registry provides fallback
    }
}
```

---

## 🧪 Testing

### **Test 1: Normal Case (Queen First)**

```bash
# Terminal 1: Start registry
./rbee-registry

# Terminal 2: Start Queen
./queen-rbee

# Terminal 3: Spawn worker via GUI
# (GUI notifies Queen directly - FAST PATH)

# Result: Queen discovers worker in <1s
```

---

### **Test 2: Queenless Case (Worker First)**

```bash
# Terminal 1: Start registry
./rbee-registry

# Terminal 2: Spawn worker via GUI
# (No Queen yet, worker registered with registry)

# Terminal 3: Do inference via GUI
# (Works! Queenless operation)

# Terminal 4: Start Queen
# (Queen queries registry, discovers existing worker)

# Result: Queen discovers pre-existing worker
```

---

### **Test 3: Registry Down**

```bash
# Terminal 1: Start Queen (registry not running)
# (Queen caches last-known config, keeps working)

# Terminal 2: Spawn worker via GUI
# (GUI notifies Queen directly - FAST PATH still works!)

# Result: System degrades gracefully, fast path compensates
```

---

## 🎯 Rollout Plan

### **Phase 1: Implement Registry**
- [ ] Create `rbee-registry` binary
- [ ] Create `registry-client` crate
- [ ] Test registry in isolation

### **Phase 2: Update All Components**
- [ ] Update Queen to query registry on startup
- [ ] Update Hive to register with registry
- [ ] Update Worker to register with registry
- [ ] Update GUI to register with registry

### **Phase 3: Add Fast Path**
- [ ] Add `/v1/queen/register-worker` endpoint to Queen
- [ ] Update spawners to notify Queen
- [ ] Test fast path optimization

### **Phase 4: Test All Scenarios**
- [ ] Test Queen-first scenario
- [ ] Test Worker-first scenario (Queenless)
- [ ] Test cross-machine discovery
- [ ] Test registry failure graceful degradation

---

## 🏆 Verdict

**⭐⭐⭐⭐⭐ HIGHLY RECOMMENDED**

**Why:**
- ✅ Solves ALL discovery problems
- ✅ Optimizes common case (fast path)
- ✅ Resilient (dual mechanisms)
- ✅ Works Queenless
- ✅ No Hive worker registry/heartbeats
- ✅ Systematic (registry is source of truth)

**Trade-offs:**
- More complexity (two mechanisms)
- Need to maintain registry daemon
- More coordination logic

**Recommendation:** **IMPLEMENT THIS SOLUTION**

---

**This hybrid approach gives you the best of both worlds: systematic discovery via registry + optimized fast path via direct notification.**

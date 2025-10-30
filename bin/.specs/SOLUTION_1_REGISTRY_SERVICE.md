# Solution 1: Registry Service (rbee-registry)

**Approach:** Create a separate daemon that tracks all components  
**Complexity:** HIGH  
**Recommended:** ⭐⭐⭐⭐⭐ **BEST SOLUTION**

---

## 🎯 Core Idea

Create a new lightweight daemon `rbee-registry` that acts as a central registry for all rbee components (hives, workers, queen).

**Key Insight:** Separate the concern of "discovery" from "orchestration".

---

## 🏗️ Architecture

```
┌──────────────────┐
│  rbee-registry   │  Port: 7830 (new)
│  (Central        │
│   Registry)      │
└────────┬─────────┘
         │
         │ All components register here
         │
    ┌────┼────┬────────┬─────────┐
    │    │    │        │         │
    ▼    ▼    ▼        ▼         ▼
┌────┐ ┌────┐ ┌────┐ ┌────┐  ┌────┐
│GUI │ │Hive│ │Work│ │Work│  │Queen│
│    │ │    │ │er1 │ │er2 │  │    │
└────┘ └────┘ └────┘ └────┘  └────┘
```

---

## 📋 Component Responsibilities

### **rbee-registry (NEW)**

**Port:** `7830` (static, well-known)

**Purpose:** Track all live rbee components

**Data Stored (In-Memory Only):**
```rust
struct ComponentRegistry {
    hives: HashMap<String, HiveEntry>,
    workers: HashMap<String, WorkerEntry>,
    queen: Option<QueenEntry>,
    gui: Option<GuiEntry>,
}

struct HiveEntry {
    hive_id: String,
    hostname: String,
    port: u16,          // 7835
    registered_at: DateTime<Utc>,
    last_seen: DateTime<Utc>,
}

struct WorkerEntry {
    worker_id: String,
    hostname: String,
    port: u16,          // 8000, 8080, 8188, etc.
    worker_type: String, // "llm", "vllm", "comfy"
    model: Option<String>,
    registered_at: DateTime<Utc>,
    last_seen: DateTime<Utc>,
}

struct QueenEntry {
    hostname: String,
    port: u16,          // 7833
    registered_at: DateTime<Utc>,
    last_seen: DateTime<Utc>,
}

struct GuiEntry {
    hostname: String,
    port: u16,          // 5173
    registered_at: DateTime<Utc>,
    last_seen: DateTime<Utc>,
}
```

**API:**
```
POST   /v1/registry/register     - Register component
DELETE /v1/registry/unregister   - Unregister component
POST   /v1/registry/heartbeat    - Keep-alive heartbeat
GET    /v1/registry/hives        - List all hives
GET    /v1/registry/workers      - List all workers
GET    /v1/registry/queen        - Get queen info
GET    /v1/registry/components   - List all components
```

**Lifecycle:**
- Started first (before any other component)
- Runs forever
- No dependencies

---

### **All Components (Hive, Worker, Queen, GUI)**

**On Startup:**
```rust
// 1. Check if registry exists
let registry_url = "http://localhost:7830";
if registry_reachable(&registry_url).await {
    // 2. Register self
    register_with_registry(&registry_url, component_info).await?;
    
    // 3. Start registry heartbeat (every 15s)
    start_registry_heartbeat_task(&registry_url, component_info);
}

// 4. Query registry for other components (if needed)
if needs_queen {
    let queen = query_registry_for_queen(&registry_url).await?;
    // Now we know Queen's URL!
}
```

**Heartbeat to Registry:**
- Every 15 seconds (faster than Queen heartbeat)
- Simple keep-alive, no payload
- Registry marks component as "online" if heartbeat received within 30s

---

## 🔄 Discovery Flows

### **Flow 1: Queen Discovers Hives/Workers**

```
1. Queen starts
2. Queen registers with rbee-registry
3. Queen queries GET /v1/registry/hives
4. Queen queries GET /v1/registry/workers
5. Queen now knows all hives/workers (hostname + port)
6. Queen fetches capabilities from each
```

---

### **Flow 2: Worker Starts Before Queen**

```
1. Worker starts (via GUI)
2. Worker registers with rbee-registry:
   {
     "worker_id": "worker-llm-8080",
     "hostname": "localhost",
     "port": 8080,
     "worker_type": "llm",
     "model": "llama-3.2-1b"
   }
3. Worker starts registry heartbeat (every 15s)
4. Queen starts later
5. Queen queries GET /v1/registry/workers
6. Queen discovers existing worker!
7. Queen fetches capabilities from worker
8. Worker starts Queen heartbeat (now that it knows Queen URL)
```

---

### **Flow 3: Queenless Operation (GUI)**

```
1. GUI starts
2. GUI registers with rbee-registry
3. User spawns worker via GUI
4. Worker registers with rbee-registry
5. GUI queries GET /v1/registry/workers
6. GUI discovers worker
7. GUI sends inference request directly to worker
8. (Queen doesn't exist, but everything works!)
```

---

### **Flow 4: Queen Starts After Everything**

```
1. Hive 1 running (registered)
2. Hive 2 running (registered)
3. Worker 1 running (registered)
4. Worker 2 running (registered)
5. GUI running (registered)
6. Queen starts
7. Queen queries GET /v1/registry/components
8. Queen discovers everyone!
9. Queen becomes orchestrator
```

---

## ✅ Advantages

### **1. Solves ALL Edge Cases**

- ✅ Works if Queen starts first
- ✅ Works if Workers start first
- ✅ Works Queenless (GUI only)
- ✅ Handles dynamic worker ports
- ✅ No Hive worker registry needed
- ✅ No Hive worker heartbeats
- ✅ Works across multiple machines
- ✅ Self-healing (auto-discover new components)

### **2. Systematic**

- Single source of truth for discovery
- Clear registration protocol
- Consistent across all components

### **3. Simple Component Logic**

```rust
// Every component does the same thing:
1. Register with registry
2. Heartbeat to registry
3. Query registry for other components
```

### **4. No Persistent Data**

- Registry is in-memory only
- No database, no files
- Clean shutdown = clean slate

### **5. GUI-Friendly**

- GUI can query registry to show all components
- GUI doesn't need Queen
- GUI can spawn workers, registry tracks them

---

## ❌ Disadvantages

### **1. New Daemon**

- One more binary to maintain
- One more thing to start
- One more failure point

### **2. Bootstrap Order**

- Registry must start first
- If registry crashes, components can't discover each other
- (Mitigation: components cache last-known URLs)

### **3. Single Point of Failure**

- If registry dies, discovery stops
- (Mitigation: components keep working with cached info)

### **4. Port 7830**

- Need to reserve another port
- Could conflict if something else uses 7830

---

## 🛠️ Implementation

### **Phase 1: Create rbee-registry Binary**

**File:** `bin/05_rbee_registry/src/main.rs`

```rust
use axum::{Router, routing::{get, post, delete}};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
struct RegistryState {
    components: Arc<RwLock<ComponentRegistry>>,
}

#[tokio::main]
async fn main() {
    let state = RegistryState {
        components: Arc::new(RwLock::new(ComponentRegistry::new())),
    };

    let app = Router::new()
        .route("/v1/registry/register", post(handle_register))
        .route("/v1/registry/unregister", delete(handle_unregister))
        .route("/v1/registry/heartbeat", post(handle_heartbeat))
        .route("/v1/registry/hives", get(list_hives))
        .route("/v1/registry/workers", get(list_workers))
        .route("/v1/registry/queen", get(get_queen))
        .route("/v1/registry/components", get(list_all_components))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:7830").await.unwrap();
    println!("🗃️  rbee-registry listening on http://0.0.0.0:7830");
    axum::serve(listener, app).await.unwrap();
}
```

---

### **Phase 2: Update All Components**

**Add to each component (Hive, Worker, Queen, GUI):**

```rust
// On startup
async fn register_with_registry() {
    let registry_url = "http://localhost:7830";
    
    let component = ComponentInfo {
        component_type: "worker",
        component_id: "worker-llm-8080",
        hostname: "localhost",
        port: 8080,
        metadata: json!({
            "worker_type": "llm",
            "model": "llama-3.2-1b"
        }),
    };
    
    let client = reqwest::Client::new();
    client.post(format!("{}/v1/registry/register", registry_url))
        .json(&component)
        .send()
        .await?;
    
    // Start registry heartbeat
    start_registry_heartbeat_task(registry_url, component).await;
}
```

---

### **Phase 3: Update Discovery Logic**

**Queen Discovery (NEW):**

```rust
// Queen startup
async fn discover_all_components() {
    let registry_url = "http://localhost:7830";
    
    // Discover hives
    let hives: Vec<HiveEntry> = query_registry_hives(registry_url).await?;
    for hive in hives {
        fetch_hive_capabilities(&hive).await?;
    }
    
    // Discover workers
    let workers: Vec<WorkerEntry> = query_registry_workers(registry_url).await?;
    for worker in workers {
        fetch_worker_capabilities(&worker).await?;
    }
}
```

---

## 🧪 Testing

### **Test 1: Registry Starts First**

```bash
# Terminal 1
./rbee-registry

# Terminal 2
./rbee-hive --id localhost

# Terminal 3
./llm-worker-rbee --port 8080

# Terminal 4
./queen-rbee

# Result: Queen discovers both hive and worker
```

---

### **Test 2: Components Start Before Registry**

```bash
# Terminal 1
./rbee-hive --id localhost
# (Hive tries to register, fails, caches Queen URL from config)

# Terminal 2
./rbee-registry
# (Registry starts)

# Terminal 3
./rbee-hive --id localhost
# (Hive retries registration, succeeds)

# Result: Components auto-register when registry becomes available
```

---

## 🎯 Rollout Plan

### **Phase 1: Create Registry Binary**
- [ ] Implement `bin/05_rbee_registry`
- [ ] Add registration API
- [ ] Add query API
- [ ] Add heartbeat tracking
- [ ] Add cleanup (remove stale components)

### **Phase 2: Add Registry Client Library**
- [ ] Create `bin/99_shared_crates/registry-client`
- [ ] Implement registration logic
- [ ] Implement heartbeat logic
- [ ] Implement query logic

### **Phase 3: Update All Components**
- [ ] Update Queen to use registry for discovery
- [ ] Update Hive to register with registry
- [ ] Update Worker to register with registry
- [ ] Update GUI to use registry for discovery

### **Phase 4: Test All Scenarios**
- [ ] Test Queen starts first
- [ ] Test Worker starts first
- [ ] Test Queenless operation
- [ ] Test cross-machine discovery

---

## 🏆 Verdict

**⭐⭐⭐⭐⭐ HIGHLY RECOMMENDED**

**Why:**
- Solves ALL discovery problems
- Systematic and clean
- Works for all scenarios (Queenless, cross-machine, dynamic ports)
- No Hive worker registry needed
- No Hive worker heartbeats

**Trade-offs:**
- One more daemon to run
- Need to start registry first (or implement auto-retry)

**Recommendation:** **IMPLEMENT THIS SOLUTION**

---

**This is the cleanest, most systematic solution to the discovery problem.**

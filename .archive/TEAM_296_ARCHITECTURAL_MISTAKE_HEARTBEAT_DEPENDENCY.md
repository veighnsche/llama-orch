# TEAM-296: Architectural Mistake - Heartbeat Dependency & Operation Modes

**Date:** Oct 26, 2025  
**Status:** 🚨 CRITICAL ARCHITECTURAL ISSUE  
**Impact:** Queen is required for all operations, even when not needed

---

## Executive Summary

**The Problem:** The current architecture makes Queen **mandatory** for all operations because:
1. Heartbeats flow through Queen (Workers → Queen → Keeper)
2. Queen SDK holds the heartbeat stream
3. Keeper GUI needs heartbeats to show alive instances
4. This prevents "Queenless Hives" from working

**The Root Cause:** We conflated two distinct use cases:
1. **GUI Mode** - User manually manages hives/workers through Keeper GUI
2. **API Mode** - Queen auto-orchestrates workers via HTTP API

**The Solution:** Introduce two operational modes with different heartbeat strategies.

---

## Current Architecture (Broken)

### Heartbeat Flow

```
Worker → Queen → Keeper GUI
  ↓        ↓         ↓
Sends    Forwards  Displays
HB       via SSE   status
```

### The Dependency Chain

```
Keeper GUI needs heartbeats
    ↓
Heartbeats come from Queen SDK
    ↓
Queen SDK requires Queen running
    ↓
Queen must be started for ANY operation
    ↓
❌ Can't run Queenless Hives
```

### Code Evidence

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`
```rust
// Workers send heartbeats DIRECTLY to queen
pub async fn handle_worker_heartbeat(
    State(state): State<HeartbeatState>,
    Json(heartbeat): Json<WorkerHeartbeat>,
) -> Result<...> {
    // Update worker registry
    state.worker_registry.update_worker(heartbeat.clone());
    
    // Broadcast event for real-time streaming
    let event = HeartbeatEvent::Worker { ... };
    state.event_tx.send(event); // ← Keeper listens to this
}
```

**File:** `bin/10_queen_rbee/src/http/heartbeat_stream.rs`
```rust
// GET /v1/heartbeats/stream - SSE endpoint
pub async fn handle_heartbeat_stream(...) {
    // Subscribe to broadcast channel
    let mut event_rx = state.event_tx.subscribe();
    
    // Queen sends her own heartbeat every 2.5 seconds
    let mut queen_interval = interval(Duration::from_millis(2500));
    
    // Stream events to Keeper GUI
    // ← Keeper GUI depends on this!
}
```

### The Problem

**Scenario 1: User wants to use GUI to manage workers**
```bash
$ ./rbee hive start      # ❌ Requires Queen running!
$ ./rbee worker spawn    # ❌ Requires Queen running!
```

Why? Because Keeper GUI needs heartbeats to show status, and heartbeats come from Queen.

**Scenario 2: User wants to use HTTP API**
```bash
$ curl http://localhost:7833/v1/jobs  # ✅ This makes sense
```

Queen routes jobs, schedules workers, load balances. This is Queen's purpose!

---

## The Two Modes

### Mode 1: GUI Mode (Direct Management)

**User Intent:** Manually manage hives and workers through Keeper GUI

**Workflow:**
```
User → Keeper GUI → Hive (direct HTTP)
                  → Worker (direct HTTP)
```

**Characteristics:**
- ✅ User controls everything manually
- ✅ No scheduling needed
- ✅ No load balancing needed
- ✅ Direct communication (Keeper → Hive/Worker)
- ❌ Queen NOT needed
- ❌ Heartbeats NOT needed (use Zustand state instead)

**Example:**
```typescript
// In Keeper GUI
const [hives, setHives] = useHivesStore();
const [workers, setWorkers] = useWorkersStore();

// Start hive directly
async function startHive(hiveId: string) {
  await fetch(`http://localhost:9000/v1/hive/start`);
  setHives(prev => [...prev, { id: hiveId, status: 'running' }]);
}

// Start worker directly
async function startWorker(workerId: string) {
  await fetch(`http://localhost:9001/v1/worker/start`);
  setWorkers(prev => [...prev, { id: workerId, status: 'running' }]);
}
```

**State Management:** Zustand stores in Keeper GUI
- `useHivesStore()` - Track which hives are running
- `useWorkersStore()` - Track which workers are running
- No heartbeats needed - user sees what they started

### Mode 2: API Mode (Auto-Orchestration)

**User Intent:** Use Queen's HTTP API for intelligent job routing

**Workflow:**
```
Client → Queen API → Hive (via forwarding)
                  → Worker (via scheduling)
```

**Characteristics:**
- ✅ Queen schedules jobs
- ✅ Queen load balances
- ✅ Queen auto-selects workers
- ✅ Heartbeats needed (for scheduling decisions)
- ✅ Multi-hive support
- ✅ OpenAI-compatible API

**Example:**
```bash
# User activates hives for Queen
$ ./rbee queen activate-hive localhost
$ ./rbee queen activate-hive gpu-server-1

# Queen now knows which hives are available
# Queen queries hive catalogs
# Queen makes scheduling decisions

# Submit inference job
$ curl -X POST http://localhost:7833/v1/inference \
  -d '{"model": "llama-3-8b", "prompt": "Hello"}'

# Queen:
# 1. Checks which workers have llama-3-8b loaded
# 2. Selects least-loaded worker
# 3. Routes request
# 4. Streams response
```

**State Management:** Queen's registries + heartbeats
- `WorkerRegistry` - Track worker health via heartbeats
- `HiveRegistry` - Track hive health via heartbeats
- Heartbeats needed for scheduling decisions

---

## The Heartbeat Problem

### Current: Heartbeats Always Required

```
Worker → Queen → Keeper GUI
  ↓        ↓         ↓
Sends    Forwards  Displays
HB       via SSE   status
```

**Problem:** Keeper GUI depends on Queen's heartbeat stream

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/`
```typescript
// Queen SDK provides heartbeat stream
const heartbeats = await queenSdk.streamHeartbeats();

// Keeper GUI uses this to show status
heartbeats.on('worker', (event) => {
  updateWorkerStatus(event.worker_id, event.status);
});
```

**Issue:** Can't import Queen SDK into Keeper without pulling in entire Queen dependency tree.

### Proposed: Mode-Specific Heartbeats

#### GUI Mode: No Heartbeats

```typescript
// Keeper GUI manages state directly
const hivesStore = create<HivesState>((set) => ({
  hives: [],
  addHive: (hive) => set((state) => ({ 
    hives: [...state.hives, hive] 
  })),
  removeHive: (id) => set((state) => ({ 
    hives: state.hives.filter(h => h.id !== id) 
  })),
}));

// User starts hive
await startHive('localhost');
hivesStore.addHive({ id: 'localhost', status: 'running' });

// User stops hive
await stopHive('localhost');
hivesStore.removeHive('localhost');
```

**Benefits:**
- ✅ No Queen dependency
- ✅ No heartbeat complexity
- ✅ Simple state management
- ✅ User sees exactly what they started

#### API Mode: Heartbeats for Scheduling

```rust
// Workers send heartbeats to Queen
POST /v1/worker-heartbeat
{
  "worker_id": "worker-123",
  "status": "idle",
  "model_loaded": "llama-3-8b",
  "gpu_utilization": 0.45
}

// Queen uses this for scheduling
impl Queen {
    fn select_worker(&self, model: &str) -> Result<WorkerId> {
        let workers = self.registry.list_available_workers();
        
        // Filter by model
        let candidates: Vec<_> = workers.iter()
            .filter(|w| w.model_loaded == model)
            .collect();
        
        // Select least loaded
        candidates.iter()
            .min_by_key(|w| w.gpu_utilization)
            .ok_or_else(|| anyhow!("No workers available"))
    }
}
```

**Benefits:**
- ✅ Queen makes informed decisions
- ✅ Load balancing works
- ✅ Auto-failover possible
- ✅ Multi-hive support

---

## The Queen's Responsibility Shift

### Original Design: Queen Does Everything

```
Queen:
- Routes jobs
- Schedules workers
- Load balances
- Forwards to hives
- Manages worker lifecycle
- Manages hive lifecycle
```

**Problem:** Too much responsibility, becomes bottleneck

### New Design: Queen as Smart API Gateway

```
Queen (API Mode):
- ✅ Routes inference jobs
- ✅ Schedules workers (load balancing)
- ✅ OpenAI-compatible API
- ✅ Multi-hive orchestration
- ❌ Does NOT manage hive lifecycle
- ❌ Does NOT manage worker lifecycle

Keeper (GUI Mode):
- ✅ Manages hive lifecycle (start/stop)
- ✅ Manages worker lifecycle (spawn/kill)
- ✅ Direct communication with hives
- ❌ Does NOT route inference jobs
- ❌ Does NOT do load balancing
```

**Key Insight:** Queen is for **smart routing**, not **lifecycle management**.

---

## The Catalog Problem

### Current: Queen Forwards Catalog Requests

```
Keeper → Queen → Hive
         (forwards)
```

**File:** `bin/10_queen_rbee/src/hive_forwarder.rs`
```rust
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
) -> Result<()> {
    match operation {
        Operation::ModelList { hive_id } => {
            // Forward to hive
            let hive_url = get_hive_url(&hive_id)?;
            forward_via_http(hive_url, operation).await
        }
        // ... other operations
    }
}
```

**Problem:** Why does Queen need to forward catalog requests? Keeper can talk to Hive directly!

### Proposed: Direct Catalog Access

```
Keeper → Hive (direct HTTP)
```

**GUI Mode:**
```typescript
// Keeper talks to hive directly
async function getModels(hiveId: string) {
  const hiveUrl = getHiveUrl(hiveId); // localhost:9000
  const response = await fetch(`${hiveUrl}/v1/models`);
  return response.json();
}

// No Queen needed!
```

**API Mode:**
```rust
// Queen queries hives for scheduling
impl Queen {
    async fn find_worker_with_model(&self, model: &str) -> Result<WorkerId> {
        // Query all activated hives
        for hive_id in &self.activated_hives {
            let hive_url = self.get_hive_url(hive_id)?;
            let workers = query_hive_workers(hive_url).await?;
            
            // Find worker with model
            if let Some(worker) = workers.iter().find(|w| w.model == model) {
                return Ok(worker.id.clone());
            }
        }
        
        Err(anyhow!("No worker found with model {}", model))
    }
}
```

---

## Implementation Plan

### Phase 1: Separate Modes

**1. Add Mode Configuration**

**File:** `bin/00_rbee_keeper/src/config.rs`
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeeperConfig {
    /// Operation mode
    pub mode: OperationMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperationMode {
    /// GUI mode - direct management, no Queen
    Gui,
    /// API mode - Queen orchestration
    Api,
}
```

**2. GUI Mode: Remove Queen Dependency**

**File:** `bin/00_rbee_keeper/ui/src/stores/hivesStore.ts`
```typescript
import { create } from 'zustand';

interface Hive {
  id: string;
  status: 'running' | 'stopped';
  url: string;
}

interface HivesState {
  hives: Hive[];
  addHive: (hive: Hive) => void;
  removeHive: (id: string) => void;
  updateStatus: (id: string, status: 'running' | 'stopped') => void;
}

export const useHivesStore = create<HivesState>((set) => ({
  hives: [],
  addHive: (hive) => set((state) => ({ 
    hives: [...state.hives, hive] 
  })),
  removeHive: (id) => set((state) => ({ 
    hives: state.hives.filter(h => h.id !== id) 
  })),
  updateStatus: (id, status) => set((state) => ({
    hives: state.hives.map(h => 
      h.id === id ? { ...h, status } : h
    )
  })),
}));
```

**3. GUI Mode: Direct Hive Communication**

**File:** `bin/00_rbee_keeper/ui/src/api/hiveClient.ts`
```typescript
// Direct HTTP client for hive operations
export class HiveClient {
  constructor(private baseUrl: string) {}
  
  async start() {
    await fetch(`${this.baseUrl}/v1/start`, { method: 'POST' });
  }
  
  async stop() {
    await fetch(`${this.baseUrl}/v1/stop`, { method: 'POST' });
  }
  
  async listModels() {
    const response = await fetch(`${this.baseUrl}/v1/models`);
    return response.json();
  }
  
  async listWorkers() {
    const response = await fetch(`${this.baseUrl}/v1/workers`);
    return response.json();
  }
}

// Usage in GUI
const hiveClient = new HiveClient('http://localhost:9000');
await hiveClient.start();
useHivesStore.addHive({ 
  id: 'localhost', 
  status: 'running',
  url: 'http://localhost:9000'
});
```

**4. API Mode: Queen Activation**

**File:** `bin/10_queen_rbee/src/hive_activation.rs`
```rust
/// Hives that Queen can use for scheduling
pub struct ActivatedHives {
    hives: HashSet<String>,
}

impl ActivatedHives {
    pub fn activate(&mut self, hive_id: String) {
        self.hives.insert(hive_id);
    }
    
    pub fn deactivate(&mut self, hive_id: &str) {
        self.hives.remove(hive_id);
    }
    
    pub fn list(&self) -> Vec<String> {
        self.hives.iter().cloned().collect()
    }
}

// CLI command
// $ rbee queen activate-hive localhost
// $ rbee queen activate-hive gpu-server-1
```

### Phase 2: Update Documentation

**1. User Guide**

```markdown
# rbee Operation Modes

## GUI Mode (Default)

Use Keeper GUI to manually manage hives and workers.

**When to use:**
- Testing locally
- Manual control
- Single machine
- Learning the system

**How to use:**
1. Start Keeper: `./rbee`
2. Go to Services page
3. Start hive
4. Spawn workers
5. Use workers directly

**No Queen needed!**

## API Mode

Use Queen's HTTP API for intelligent job routing.

**When to use:**
- Production deployments
- Multi-machine setups
- Load balancing needed
- OpenAI-compatible API

**How to use:**
1. Start Queen: `./rbee queen start`
2. Activate hives: `./rbee queen activate-hive localhost`
3. Submit jobs: `curl http://localhost:7833/v1/inference`
4. Queen handles routing

**Queen required!**
```

**2. Architecture Docs**

Update `.arch/00_OVERVIEW_PART_1.md` to explain modes.

### Phase 3: Deprecate Forwarding

**Remove:** Queen forwarding of lifecycle operations

**Keep:** Queen forwarding of inference operations (API mode only)

**File:** `bin/10_queen_rbee/src/job_router.rs`
```rust
// BEFORE (everything forwarded)
Operation::HiveStart { .. } => forward_to_hive(...).await,
Operation::WorkerSpawn { .. } => forward_to_hive(...).await,
Operation::ModelList { .. } => forward_to_hive(...).await,
Operation::Infer { .. } => forward_to_hive(...).await,

// AFTER (only inference forwarded)
Operation::Infer { .. } => {
    // Queen schedules and routes
    schedule_inference(...).await
}

// Lifecycle operations removed - use Keeper GUI directly
Operation::HiveStart { .. } => {
    Err(anyhow!("Use Keeper GUI for lifecycle management"))
}
```

---

## Benefits of New Architecture

### GUI Mode Benefits

1. **✅ No Queen Dependency**
   - Start hives without Queen
   - Spawn workers without Queen
   - View catalogs without Queen

2. **✅ Simpler State Management**
   - Zustand stores instead of heartbeats
   - User sees what they started
   - No SSE complexity

3. **✅ Direct Communication**
   - Keeper → Hive (no middleman)
   - Faster operations
   - Easier debugging

4. **✅ Better for Learning**
   - Understand each component
   - See direct cause/effect
   - No hidden orchestration

### API Mode Benefits

1. **✅ Smart Routing**
   - Queen selects best worker
   - Load balancing
   - Auto-failover

2. **✅ Multi-Hive Support**
   - Query multiple hives
   - Aggregate catalogs
   - Distribute load

3. **✅ OpenAI Compatibility**
   - Standard API
   - Drop-in replacement
   - Familiar interface

4. **✅ Production Ready**
   - Heartbeat monitoring
   - Health checks
   - Graceful degradation

---

## Migration Path

### For Users

**Current (broken):**
```bash
$ ./rbee queen start  # Required for everything
$ ./rbee hive start   # Fails without Queen
```

**After (fixed):**
```bash
# GUI Mode (default)
$ ./rbee              # Opens GUI
# Click "Start Hive" - works without Queen!

# API Mode (explicit)
$ ./rbee queen start
$ ./rbee queen activate-hive localhost
$ curl http://localhost:7833/v1/inference
```

### For Developers

**Remove:**
- Queen forwarding of lifecycle operations
- Heartbeat dependency in Keeper GUI
- Queen SDK import in Keeper

**Add:**
- Zustand stores for GUI state
- Direct hive client in Keeper
- Mode configuration

**Update:**
- Documentation
- Examples
- Tests

---

## Conclusion

**The Mistake:** Making Queen mandatory for all operations by tying heartbeats to GUI state.

**The Fix:** Separate GUI Mode (direct management) from API Mode (orchestration).

**The Result:**
- ✅ Queenless Hives work
- ✅ Simpler for beginners
- ✅ More powerful for production
- ✅ Clear separation of concerns

**Next Steps:**
1. Implement Zustand stores for GUI mode
2. Add direct hive client to Keeper
3. Add Queen activation commands
4. Update documentation
5. Deprecate forwarding

---

**TEAM-296: This is a critical architectural issue that must be addressed before v1.0.**

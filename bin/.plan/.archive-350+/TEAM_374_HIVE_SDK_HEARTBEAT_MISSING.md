# TEAM-374: Hive SDK Missing HeartbeatMonitor

**Date:** Oct 31, 2025  
**Status:** 🔴 ISSUE FOUND  
**Priority:** HIGH (needed for Hive UI Phase 3)

---

## Problem

The **Hive SDK** (`bin/20_rbee_hive/ui/packages/rbee-hive-sdk`) does **NOT** have a `HeartbeatMonitor` class, but:

1. ✅ Hive backend HAS the SSE endpoint: `GET /v1/heartbeats/stream`
2. ✅ Queen SDK HAS `HeartbeatMonitor` (connects to Queen's stream)
3. ❌ Hive SDK MISSING `HeartbeatMonitor` (can't connect to Hive's stream)

---

## Impact

### Hive UI Phase 3 Blocked

**From:** `bin/20_rbee_hive/ui/ARCHITECTURE.md`

```markdown
### Phase 3: Real-Time Updates ⏳

**Features:**
- ⏳ Auto-refresh worker list (every 2 seconds)
- ⏳ Live narration feed (SSE events)
- ⏳ Toast notifications (spawn success, errors)
- ⏳ Progress indicators (downloads, spawns)
```

**Without HeartbeatMonitor:**
- ❌ Can't get real-time worker updates
- ❌ Must poll `/v1/jobs` repeatedly (inefficient)
- ❌ No live status indicators
- ❌ Delayed UI updates

---

## What Exists

### Hive Backend

**File:** `bin/20_rbee_hive/src/main.rs`

```rust
.route("/v1/heartbeats/stream", get(http::handle_heartbeat_stream))
```

**Endpoint:** ✅ `GET /v1/heartbeats/stream` exists

**File:** `bin/20_rbee_hive/src/http/heartbeat_stream.rs`

```rust
/// GET /v1/heartbeats/stream - SSE endpoint for hive telemetry
///
/// TEAM-372: Queen subscribes to this after discovery handshake completes.
/// Broadcasts worker telemetry every 1 second.
pub async fn handle_heartbeat_stream(
    State(state): State<HeartbeatStreamState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
```

**Broadcasts:** `HiveHeartbeatEvent` every 1 second

### Queen SDK (Reference Implementation)

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/heartbeat.rs`

```rust
#[wasm_bindgen]
pub struct HeartbeatMonitor {
    event_source: Option<EventSource>,
    base_url: String,
}

#[wasm_bindgen]
impl HeartbeatMonitor {
    pub fn new(base_url: String) -> Self
    pub fn start(&mut self, on_update: Function) -> Result<(), JsValue>
    pub fn stop(&mut self)
    pub fn is_connected(&self) -> bool
}
```

**Usage:**
```javascript
const monitor = new HeartbeatMonitor('http://localhost:7833');
monitor.start((event) => {
  console.log('Heartbeat:', event);
});
```

### Hive SDK (Current State)

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/lib.rs`

```rust
pub use client::HiveClient;
pub use operations::OperationBuilder;
```

**Exports:**
- ✅ `HiveClient` (job submission)
- ✅ `OperationBuilder` (operation builder)
- ❌ `HeartbeatMonitor` (MISSING!)

---

## What's Needed

### 1. Add HeartbeatMonitor to Hive SDK

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/heartbeat.rs` (NEW)

**Implementation:** Copy from Queen SDK with minor adjustments

**Differences from Queen SDK:**
- Connects to Hive's stream (not Queen's)
- Receives `HiveHeartbeatEvent` (not `HeartbeatEvent`)
- Same SSE pattern, different data structure

### 2. Update Hive SDK Exports

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/lib.rs`

```rust
mod heartbeat; // NEW

pub use client::HiveClient;
pub use operations::OperationBuilder;
pub use heartbeat::HeartbeatMonitor; // NEW
```

### 3. Add TypeScript Types

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/index.ts` (NEW or UPDATE)

```typescript
export interface HiveHeartbeatEvent {
  type: 'telemetry'
  hive_id: string
  hive_info: HiveInfo
  timestamp: string
  workers: ProcessStats[]
}

export interface ProcessStats {
  pid: number
  group: string
  instance: string
  cpu_pct: number
  rss_mb: number
  gpu_util_pct: number
  vram_mb: number
  total_vram_mb: number
  model: string | null
}
```

---

## Data Structure Comparison

### Queen's HeartbeatEvent

```typescript
type HeartbeatEvent = 
  | { type: 'hive_telemetry', hive_id, workers }
  | { type: 'queen', workers_online, hives_online, ... }
```

**Purpose:** Aggregated view of ALL hives + Queen's own status

### Hive's HiveHeartbeatEvent

```typescript
interface HiveHeartbeatEvent {
  type: 'telemetry'
  hive_id: string
  hive_info: HiveInfo
  workers: ProcessStats[]
}
```

**Purpose:** Single hive's telemetry (workers + hive info)

---

## Implementation Plan

### Step 1: Create heartbeat.rs

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/heartbeat.rs`

**Copy from:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/heartbeat.rs`

**Changes:**
1. Update comments (Hive instead of Queen)
2. Keep same SSE pattern
3. Same API: `new()`, `start()`, `stop()`, `is_connected()`

### Step 2: Update lib.rs

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/lib.rs`

```rust
mod heartbeat;
pub use heartbeat::HeartbeatMonitor;
```

### Step 3: Add TypeScript Types

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/index.ts`

```typescript
export interface HiveHeartbeatEvent { ... }
export interface ProcessStats { ... }
export type { HeartbeatMonitor } from './pkg/bundler/rbee_hive_sdk'
```

### Step 4: Update Cargo.toml

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/Cargo.toml`

```toml
[dependencies.web-sys]
features = [
  "EventSource",
  "MessageEvent",
  "Event",
]
```

### Step 5: Rebuild WASM

```bash
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
pnpm build
```

---

## Usage Example (After Implementation)

### Hive UI Component

```typescript
import { HeartbeatMonitor } from '@rbee/rbee-hive-sdk';

function WorkerList() {
  const [workers, setWorkers] = useState([]);
  
  useEffect(() => {
    const monitor = new HeartbeatMonitor('http://localhost:7835');
    
    monitor.start((event) => {
      // event.workers is ProcessStats[]
      setWorkers(event.workers);
    });
    
    return () => monitor.stop();
  }, []);
  
  return (
    <div>
      {workers.map(w => (
        <div key={w.pid}>
          Worker {w.pid}: {w.model} ({w.gpu_util_pct}% GPU)
        </div>
      ))}
    </div>
  );
}
```

---

## Benefits

### With HeartbeatMonitor

✅ **Real-time updates** (1s interval)  
✅ **Single SSE connection** (efficient)  
✅ **Automatic reconnection** (built-in)  
✅ **Type-safe** (TypeScript types)  
✅ **Consistent API** (same as Queen SDK)

### Without HeartbeatMonitor

❌ **Polling required** (every 2s)  
❌ **Multiple HTTP requests** (wasteful)  
❌ **Manual reconnection** (complex)  
❌ **Delayed updates** (2s lag)  
❌ **Higher server load** (repeated requests)

---

## Verification

### Test After Implementation

```bash
# Terminal 1: Start Hive
cargo run --bin rbee-hive -- --port 7835

# Terminal 2: Test SSE stream directly
curl -N http://localhost:7835/v1/heartbeats/stream

# Terminal 3: Test SDK (after rebuild)
cd bin/20_rbee_hive/ui/app
pnpm dev
# Open browser, check console for heartbeat events
```

---

## Summary

### Current State

- ✅ Hive backend has SSE endpoint
- ✅ Queen SDK has HeartbeatMonitor
- ❌ Hive SDK MISSING HeartbeatMonitor

### Required Action

1. Copy `heartbeat.rs` from Queen SDK to Hive SDK
2. Update exports in `lib.rs`
3. Add TypeScript types
4. Rebuild WASM
5. Test with Hive UI

### Priority

**HIGH** - Blocks Hive UI Phase 3 (Real-Time Updates)

---

**TEAM-374: Hive SDK needs HeartbeatMonitor to match Queen SDK architecture!**

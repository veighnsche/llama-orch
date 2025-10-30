# TEAM-364: Heartbeat Data Flow - Backend to Frontend

**Date:** Oct 30, 2025  
**Status:** ✅ Backend Fixed, Frontend Needs Update

---

## 🔄 DATA FLOW OVERVIEW

```
Hive (every 1s)
  └─> POST /v1/hive-heartbeat
      └─> HeartbeatEvent::HiveTelemetry { hive_id, workers: Vec<ProcessStats> }
          └─> Broadcast to SSE channel
              └─> GET /v1/heartbeats/stream (SSE)
                  ├─> Queen heartbeat (every 2.5s)
                  └─> Hive telemetry (every 1s)
                      └─> Frontend receives events
                          └─> HeartbeatMonitor.tsx displays
```

---

## 📊 BACKEND DATA STRUCTURES

### **HeartbeatEvent Enum**

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs:19-39`

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HeartbeatEvent {
    /// Hive telemetry with worker details (sent every 1s from each hive)
    HiveTelemetry {
        hive_id: String,
        timestamp: String,
        workers: Vec<ProcessStats>,  // ← This is the key data!
    },
    
    /// Queen's own heartbeat (sent every 2.5s)
    Queen {
        workers_online: usize,
        workers_available: usize,
        hives_online: usize,
        hives_available: usize,
        worker_ids: Vec<String>,
        hive_ids: Vec<String>,
        timestamp: String,
    },
}
```

### **ProcessStats Structure**

**File:** `bin/25_rbee_hive_crates/monitor/src/lib.rs:51-82`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessStats {
    pub pid: u32,
    pub group: String,          // e.g., "llm"
    pub instance: String,       // e.g., "8080"
    pub cpu_pct: f64,           // CPU percentage
    pub rss_mb: u64,            // Memory in MB
    pub io_r_mb_s: f64,         // I/O read rate
    pub io_w_mb_s: f64,         // I/O write rate
    pub uptime_s: u64,          // Uptime in seconds
    
    // GPU telemetry
    pub gpu_util_pct: f64,      // GPU utilization (0.0 = idle, >0 = busy)
    pub vram_mb: u64,           // VRAM used in MB
    pub total_vram_mb: u64,     // Total VRAM available in MB
    
    // Model detection
    pub model: Option<String>,  // Model name (e.g., "llama-3.2-1b")
}
```

---

## 🌐 SSE EVENT FORMAT

### **Example: HiveTelemetry Event**

```json
{
  "type": "hive_telemetry",
  "hive_id": "hive-gpu-1",
  "timestamp": "2025-10-30T20:15:30Z",
  "workers": [
    {
      "pid": 12345,
      "group": "llm",
      "instance": "8080",
      "cpu_pct": 0.0,
      "rss_mb": 2048,
      "io_r_mb_s": 0.0,
      "io_w_mb_s": 0.0,
      "uptime_s": 300,
      "gpu_util_pct": 85.5,
      "vram_mb": 4096,
      "total_vram_mb": 24576,
      "model": "llama-3.2-1b"
    }
  ]
}
```

### **Example: Queen Event**

```json
{
  "type": "queen",
  "workers_online": 3,
  "workers_available": 2,
  "hives_online": 2,
  "hives_available": 2,
  "worker_ids": ["worker-1", "worker-2", "worker-3"],
  "hive_ids": ["hive-gpu-1", "hive-gpu-2"],
  "timestamp": "2025-10-30T20:15:30Z"
}
```

---

## 🔧 BACKEND FIXES APPLIED

### **1. Fixed Cargo.toml**

Added missing dependencies:
- `rbee-hive-monitor` - For ProcessStats type
- `tracing` - For cleanup task logging

### **2. Fixed Tests**

**File:** `bin/10_queen_rbee/src/http/heartbeat_stream.rs:77-169`

- Renamed `test_create_snapshot_*` → `test_create_queen_heartbeat_*`
- Added missing imports (`Arc`, `broadcast`)
- Fixed function calls to use `create_queen_heartbeat`
- Updated assertions to match `HeartbeatEvent::Queen` enum

### **3. Verified Data Flow**

✅ Hive sends telemetry → `handle_hive_heartbeat`  
✅ Telemetry stored in `HiveRegistry`  
✅ Event broadcast to SSE channel  
✅ SSE stream sends events to frontend  
✅ Backend compiles successfully

---

## 🎨 FRONTEND CURRENT STATE

### **HeartbeatMonitor Component**

**File:** `bin/10_queen_rbee/ui/app/src/components/HeartbeatMonitor.tsx`

**Current Props:**
```typescript
interface HeartbeatMonitorProps {
  workersOnline: number;
  hives: any[];  // ← Too generic!
}
```

**Current Display:**
- ✅ Workers online count
- ✅ Hives count
- ✅ Hive ID
- ✅ Worker count per hive
- ⚠️ Worker details (id, model_id) - but data structure doesn't match!

---

## 🚨 FRONTEND ISSUES TO FIX

### **Issue #1: Type Mismatch**

**Problem:** Frontend expects `worker.id` and `worker.model_id`, but backend sends `ProcessStats` which has:
- `pid` (not `id`)
- `model` (not `model_id`)
- `group` and `instance` (not combined into `id`)

**Current Code:**
```typescript
{hive.workers.map((worker: any) => (
  <div key={worker.id}>  {/* ← worker.id doesn't exist! */}
    <span>{worker.id}</span>  {/* ← Should be worker.pid or construct from group/instance */}
    <Badge>{worker.model_id}</Badge>  {/* ← Should be worker.model */}
  </div>
))}
```

### **Issue #2: Missing Telemetry Data**

**Available but not displayed:**
- GPU utilization (`gpu_util_pct`)
- VRAM usage (`vram_mb` / `total_vram_mb`)
- CPU percentage (`cpu_pct`)
- Memory usage (`rss_mb`)
- Uptime (`uptime_s`)

### **Issue #3: SDK Type Definitions**

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts`

**Current:**
```typescript
export interface WorkerInfo {
  id: string;           // ← Doesn't match ProcessStats
  model_id: string;     // ← Should be 'model: string | null'
  device: number;       // ← Doesn't exist in ProcessStats
  port: number;         // ← Should be 'instance: string'
  status: string;       // ← Doesn't exist in ProcessStats
  last_heartbeat: string;  // ← Doesn't exist
}
```

**Should be:**
```typescript
export interface ProcessStats {
  pid: number;
  group: string;
  instance: string;
  cpu_pct: number;
  rss_mb: number;
  io_r_mb_s: number;
  io_w_mb_s: number;
  uptime_s: number;
  gpu_util_pct: number;
  vram_mb: number;
  total_vram_mb: number;
  model: string | null;
}

export interface HiveTelemetry {
  hive_id: string;
  timestamp: string;
  workers: ProcessStats[];
}

export interface QueenHeartbeat {
  workers_online: number;
  workers_available: number;
  hives_online: number;
  hives_available: number;
  worker_ids: string[];
  hive_ids: string[];
  timestamp: string;
}

export type HeartbeatEvent = 
  | { type: 'hive_telemetry' } & HiveTelemetry
  | { type: 'queen' } & QueenHeartbeat;
```

---

## ✅ FRONTEND FIXES NEEDED

### **Step 1: Update SDK Types**

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts`

1. Replace `WorkerInfo` with `ProcessStats`
2. Add `HiveTelemetry` interface
3. Add `QueenHeartbeat` interface
4. Add `HeartbeatEvent` union type

### **Step 2: Update React Hook**

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useHeartbeat.ts`

1. Update `HeartbeatData` interface to match `HeartbeatEvent`
2. Handle both event types (hive_telemetry and queen)
3. Aggregate hive telemetry into a usable structure

### **Step 3: Update HeartbeatMonitor Component**

**File:** `bin/10_queen_rbee/ui/app/src/components/HeartbeatMonitor.tsx`

1. Update props to accept `ProcessStats[]` per hive
2. Display worker ID as `${group}/${instance}` or just `pid`
3. Display model as `worker.model` (nullable)
4. Add GPU utilization display
5. Add VRAM usage display
6. Add memory usage display

---

## 📝 RECOMMENDED FRONTEND IMPLEMENTATION

### **Updated HeartbeatMonitor Props**

```typescript
interface WorkerTelemetry {
  pid: number;
  group: string;
  instance: string;
  model: string | null;
  gpu_util_pct: number;
  vram_mb: number;
  total_vram_mb: number;
  rss_mb: number;
  uptime_s: number;
}

interface HiveData {
  hive_id: string;
  workers: WorkerTelemetry[];
  last_update: string;
}

interface HeartbeatMonitorProps {
  workersOnline: number;
  hivesOnline: number;
  hives: HiveData[];
}
```

### **Updated Worker Display**

```typescript
{hive.workers.map((worker) => (
  <div key={worker.pid} className="flex items-center gap-3 p-2">
    <PulseBadge variant="success" size="sm" animated />
    
    {/* Worker ID */}
    <span className="font-mono text-xs">
      {worker.group}/{worker.instance}
    </span>
    
    {/* Model */}
    {worker.model && (
      <Badge variant="outline">{worker.model}</Badge>
    )}
    
    {/* GPU Utilization */}
    <div className="ml-auto flex items-center gap-2">
      <span className="text-xs text-muted-foreground">
        GPU: {worker.gpu_util_pct.toFixed(1)}%
      </span>
      <span className="text-xs text-muted-foreground">
        VRAM: {worker.vram_mb}MB / {worker.total_vram_mb}MB
      </span>
    </div>
  </div>
))}
```

---

## 🎯 IMPLEMENTATION CHECKLIST

### **Backend** ✅ COMPLETE
- [x] Add `rbee-hive-monitor` dependency
- [x] Add `tracing` dependency
- [x] Fix test function names
- [x] Fix test imports
- [x] Verify compilation

### **Frontend** 🔄 TODO
- [ ] Update SDK types (`ProcessStats`, `HiveTelemetry`, `QueenHeartbeat`)
- [ ] Update `useHeartbeat` hook to handle new event types
- [ ] Update `HeartbeatMonitor` component props
- [ ] Update worker display to show correct fields
- [ ] Add GPU utilization display
- [ ] Add VRAM usage display
- [ ] Add memory usage display
- [ ] Test with real data

---

## 🚀 NEXT STEPS

1. **Update SDK types** - Match backend `ProcessStats` structure
2. **Update React hook** - Handle `HeartbeatEvent` union type
3. **Update component** - Display telemetry data correctly
4. **Test end-to-end** - Verify data flows from hive → queen → UI

**Estimated Time:** 1-2 hours

---

**Backend Status:** ✅ COMPLETE  
**Frontend Status:** 🔄 NEEDS UPDATE  
**Data Flow:** ✅ VERIFIED

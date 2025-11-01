# TEAM-362 CHECKLIST - Queen Storage & SSE Integration

**Date:** Oct 30, 2025  
**Mission:** Store worker telemetry in Queen and stream to UI via SSE

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Update HeartbeatEvent Enum**
- [x] Add `HiveTelemetry` variant with workers array
- [x] Keep existing `Hive`, `Queen` variants
- [x] Mark deprecated `Worker` and `Hive` variants

### **Phase 2: Update HiveRegistry**
- [x] Add method to store workers: `update_workers(hive_id, Vec<ProcessStats>)`
- [x] Store workers in HashMap<String, Vec<ProcessStats>>
- [x] Add method to query workers: `get_workers(hive_id) -> Vec<ProcessStats>`
- [x] Add scheduling methods: `find_idle_workers()`, `find_workers_with_model()`, `find_workers_with_capacity()`

### **Phase 3: Update handle_hive_heartbeat**
- [x] Extract workers from HiveHeartbeat
- [x] Store workers in HiveRegistry
- [x] Create HeartbeatEvent::HiveTelemetry with workers
- [x] Broadcast event to SSE stream

### **Phase 4: Update SSE Stream**
- [x] Forward HiveTelemetry events to UI (automatic via broadcast)
- [x] Include worker details in SSE data
- [x] Keep Queen heartbeat (every 2.5s)

### **Phase 5: Verification**
- [x] Compile: `cargo check -p queen-rbee-hive-registry`
- [x] Compile: `cargo check -p queen-rbee`
- [ ] Test: Workers appear in SSE stream (manual test)
- [ ] Test: Worker data queryable for scheduling (manual test)

---

## 🎯 **EXPECTED RESULT**

**Queen stores worker telemetry:**
```rust
hive_registry.update_workers("localhost", vec![
    ProcessStats { pid: 12345, gpu_util_pct: 85.0, ... }
]);

// Query for scheduling
let idle_workers = hive_registry.find_idle_workers();
```

**UI receives SSE events:**
```json
{
  "type": "hive_telemetry",
  "hive_id": "localhost",
  "timestamp": "2025-10-30T18:30:00Z",
  "workers": [
    {
      "pid": 12345,
      "group": "llm",
      "instance": "8080",
      "gpu_util_pct": 85.0,
      "vram_mb": 8192,
      "model": "llama-3.2-1b"
    }
  ]
}
```

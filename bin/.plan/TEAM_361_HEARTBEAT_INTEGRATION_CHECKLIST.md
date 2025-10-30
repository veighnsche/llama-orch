# TEAM-361 CHECKLIST - Heartbeat Integration

**Date:** Oct 30, 2025  
**Mission:** Wire up worker telemetry collection to heartbeat system

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Update hive-contract Types**
- [x] Add `workers: Vec<ProcessStats>` field to HiveHeartbeat
- [x] Import ProcessStats from rbee-hive-monitor
- [x] Update serialization
- [x] Add `with_workers()` constructor

### **Phase 2: Update heartbeat.rs**
- [x] Import `rbee_hive_monitor::collect_all_workers`
- [x] Call `collect_all_workers()` in heartbeat loop
- [x] Add workers to HiveHeartbeat payload
- [x] Handle collection errors gracefully

### **Phase 3: Change Interval**
- [x] Change from 30s to 1s (as per spec)
- [x] Add comment explaining 1s interval

### **Phase 4: Verification**
- [x] Compile: `cargo check -p hive-contract`
- [x] Compile: `cargo check -p rbee-hive`
- [x] Verify heartbeat includes worker data
- [ ] Test with actual workers running (manual test)

---

## 🎯 **EXPECTED RESULT**

```rust
// Every 1 second:
let workers = collect_all_workers().await?;

let heartbeat = HiveHeartbeat {
    hive_info,
    workers,  // ← Vec<ProcessStats> with GPU, model, etc.
};

POST to Queen /v1/hive-heartbeat
```

**Queen receives:**
- Hive info
- All workers with GPU utilization, VRAM, model name
- Every 1 second (real-time scheduling data)

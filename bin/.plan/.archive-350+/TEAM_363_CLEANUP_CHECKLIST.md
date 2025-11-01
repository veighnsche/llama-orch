# TEAM-363 CHECKLIST - Remove Deprecated Code (RULE ZERO)

**Date:** Oct 30, 2025  
**Mission:** Delete all deprecated code and TODOs - let compiler find breakages

---

## 🔥 **RULE ZERO VIOLATIONS TO FIX**

### **Phase 1: Remove Deprecated HeartbeatEvent Variants**
- [x] Delete `Worker` variant from HeartbeatEvent enum
- [x] Delete `Hive` variant from HeartbeatEvent enum
- [x] Keep only `HiveTelemetry` and `Queen` variants
- [x] Fix any compilation errors

### **Phase 2: Remove Deprecated Worker Heartbeat Endpoint**
- [x] Delete `handle_worker_heartbeat()` function entirely
- [x] Remove route from main.rs
- [x] Remove export from mod.rs
- [x] Fix any compilation errors

### **Phase 3: Remove All TODO Comments**
- [x] Remove TODOs from heartbeat.rs (queen)
- [x] Remove TODOs from heartbeat_stream.rs
- [x] Remove TODOs from hive heartbeat.rs (rbee-hive)
- [x] Verify no TODOs remain

### **Phase 4: Verification**
- [x] Compile: `cargo check -p queen-rbee`
- [x] Compile: `cargo check -p rbee-hive`
- [x] No deprecated code remains
- [x] No TODO markers remain

---

## 📊 **EXPECTED RESULT**

**Clean HeartbeatEvent:**
```rust
pub enum HeartbeatEvent {
    HiveTelemetry { hive_id, timestamp, workers },
    Queen { ... },
}
```

**No deprecated endpoints.**
**No TODO markers.**
**Compiler errors = things to fix.**

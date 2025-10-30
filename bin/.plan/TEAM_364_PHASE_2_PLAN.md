# TEAM-364 PHASE 2: QUICK WINS - PLAN

**Team:** TEAM-364  
**Phase:** 2 of 6  
**Status:** 📋 PLANNED  
**Estimated Duration:** 0.2 days (1.5 hours)

---

## 🎯 OBJECTIVES

Fix the 2 easiest critical issues that provide immediate value:
1. Auto stale cleanup (prevents dead worker accumulation)
2. Dynamic VRAM detection (fixes hardcoded 24GB limit)

---

## 📋 TASKS

### **Task 1: Auto Stale Cleanup (30 minutes)**

**Problem:** Dead workers accumulate in HiveRegistry, never removed

**Solution:** Background task calling `cleanup_stale()` every 60 seconds

**Files to Modify:**
- `bin/10_queen_rbee/src/main.rs`

**Implementation:**
```rust
// TEAM-364: Spawn background task for stale worker cleanup
let registry_clone = Arc::clone(&hive_registry);
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    loop {
        interval.tick().await;
        registry_clone.cleanup_stale();
        tracing::debug!("Cleaned up stale workers");
    }
});
```

**Test:** `test_worker_dies_removed_from_registry` (integration test)

---

### **Task 2: Dynamic VRAM Detection (1 hour)**

**Problem:** Hardcoded 24GB VRAM limit breaks on different GPUs

**Solution:** Query actual GPU VRAM from nvidia-smi, store per-worker

**Files to Modify:**
1. `bin/25_rbee_hive_crates/monitor/src/monitor.rs` - Query total VRAM
2. `bin/25_rbee_hive_crates/monitor/src/lib.rs` - Add `total_vram_mb` field to `ProcessStats`
3. `bin/15_queen_rbee_crates/hive-registry/src/registry.rs` - Use worker's total VRAM

**Implementation:**

**Step 1: Query GPU VRAM**
```rust
// TEAM-364: Query total GPU VRAM
fn query_gpu_vram() -> Result<u64> {
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()?;
    
    if !output.status.success() {
        return Ok(24576); // Default 24GB
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let vram_mb = stdout.trim().parse().unwrap_or(24576);
    Ok(vram_mb)
}
```

**Step 2: Add field to ProcessStats**
```rust
pub struct ProcessStats {
    // ... existing fields
    pub vram_mb: u64,
    pub total_vram_mb: u64, // TEAM-364: Total GPU VRAM
    pub model: Option<String>,
}
```

**Step 3: Use in capacity check**
```rust
// TEAM-364: Use worker's actual total VRAM instead of hardcoded limit
pub fn find_workers_with_capacity(&self, required_vram_mb: u64) -> Vec<ProcessStats> {
    self.get_all_workers()
        .into_iter()
        .filter(|w| {
            let total = if w.total_vram_mb > 0 { w.total_vram_mb } else { 24576 };
            w.vram_mb + required_vram_mb < total
        })
        .collect()
}
```

**Test:** `test_find_workers_with_capacity_checks_vram` (update existing)

---

## ✅ ACCEPTANCE CRITERIA

- [ ] Background cleanup task spawned in Queen main
- [ ] Stale workers removed after 90 seconds
- [ ] GPU VRAM queried from nvidia-smi
- [ ] ProcessStats includes `total_vram_mb` field
- [ ] Capacity check uses worker's actual VRAM
- [ ] All existing tests still pass
- [ ] No regressions

---

## 📊 EXPECTED RESULTS

| Metric | Before | After |
|--------|--------|-------|
| Dead worker cleanup | Manual | Automatic (60s) |
| VRAM detection | Hardcoded 24GB | Dynamic per-GPU |
| Critical issues fixed | 2/7 | 4/7 |
| Tests passing | 20 | 20+ |

---

## 🔗 HANDOFF TO PHASE 3

After Phase 2:
- 4 of 7 critical issues fixed
- All quick wins complete
- Ready for ProcessMonitor tests (requires Linux + cgroups)

---

**Phase 2 Plan Created:** Oct 30, 2025  
**Estimated Effort:** 1.5 hours  
**Next Phase:** Phase 3 - ProcessMonitor Tests

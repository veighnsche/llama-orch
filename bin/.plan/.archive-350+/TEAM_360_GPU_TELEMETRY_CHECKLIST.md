# TEAM-360 CHECKLIST - GPU Telemetry Implementation

**Date:** Oct 30, 2025  
**Mission:** Add GPU telemetry and model detection to ProcessStats (simple approach)

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Update ProcessStats Structure**
- [x] Add `gpu_util_pct: f64` field to ProcessStats
- [x] Change `vram_mb: Option<u64>` to `vram_mb: u64` (always query)
- [x] Add `model: Option<String>` field to ProcessStats

### **Phase 2: Implement GPU Query (nvidia-smi)**
- [x] Add `query_nvidia_smi(pid: u32) -> Result<(f64, u64)>` function
- [x] Parse nvidia-smi output for: gpu_util_pct, vram_mb
- [x] Handle case where process not using GPU (return 0.0, 0)
- [x] Linux-only implementation (#[cfg(target_os = "linux")])

### **Phase 3: Implement Model Detection**
- [x] Add `extract_model_from_cmdline(pid: u32) -> Result<Option<String>>` function
- [x] Read `/proc/{pid}/cmdline`
- [x] Parse null-separated args
- [x] Find `--model` argument and return value
- [x] Linux-only implementation (#[cfg(target_os = "linux")])

### **Phase 4: Update collect_stats_linux()**
- [x] Call `query_nvidia_smi(pid)` to get GPU stats
- [x] Call `extract_model_from_cmdline(pid)` to get model name
- [x] Populate new fields in ProcessStats
- [x] Handle errors gracefully (default to 0.0/None if query fails)

### **Phase 5: Update Documentation**
- [x] Update ProcessStats field documentation
- [x] Add examples showing GPU usage detection
- [x] Document that gpu_util_pct > 0 means worker is busy

### **Phase 6: Verification**
- [x] Compile: `cargo check -p rbee-hive-monitor`
- [x] Verify ProcessStats has all new fields
- [x] Test nvidia-smi parsing (if GPU available)

---

## 🎯 **EXPECTED RESULT**

```rust
pub struct ProcessStats {
    pub pid: u32,
    pub group: String,
    pub instance: String,
    
    // OS stats (from cgroup)
    pub cpu_pct: f64,
    pub rss_mb: u64,
    pub uptime_s: u64,
    pub io_r_mb_s: f64,
    pub io_w_mb_s: f64,
    
    // GPU stats (from nvidia-smi) - NEW
    pub gpu_util_pct: f64,   // ← 0.0 = idle, >0 = busy
    pub vram_mb: u64,         // ← GPU memory used
    
    // Model info (from /proc/pid/cmdline) - NEW
    pub model: Option<String>, // ← e.g., "llama-3.2-1b"
}
```

---

## 📊 **SCHEDULING LOGIC (For Queen)**

```rust
// Can accept new job?
if worker.vram_mb + required_vram < gpu_capacity {
    // Has capacity
}

// Is worker busy?
if worker.gpu_util_pct > 0.0 {
    // Currently processing
}

// Which model?
if worker.model == Some("llama-3.2-1b") {
    // Model already loaded
}
```

---

**Start implementation after checklist review.**

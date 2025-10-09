# M0 Remaining Tasks

**Date:** 2025-10-09T17:34:00+02:00  
**Status:** HIGH PRIORITY  
**Must complete before M1**

---

## Critical M0 Tasks (Before CP4)

### 1. Backend Catalog Detection ⏳
**File:** `bin/rbees-pool/src/commands/backends.rs` (new)

**Task:** Detect available backends on pool

**Implementation:**
```rust
pub fn detect_backends() -> Vec<Backend> {
    let mut backends = vec![Backend::Cpu]; // Always available
    
    #[cfg(target_os = "macos")]
    if has_metal() {
        backends.push(Backend::Metal);
    }
    
    #[cfg(not(target_os = "macos"))]
    if has_cuda() {
        backends.push(Backend::Cuda);
    }
    
    backends
}
```

**Command:**
```bash
rbees-pool backends list
```

**Output:**
```
Available Backends on mac.home.arpa
====================================
✅ CPU (always available)
✅ Metal (Apple GPU)
❌ CUDA (not available on Mac)
```

**Priority:** MEDIUM (nice to have for CP4)

---

### 2. Worker Cancellation Support ⏳
**File:** `bin/rbees-workerd/src/http/cancel.rs` (new)

**Task:** Add POST /cancel endpoint to worker

**API:**
```rust
POST /cancel
{
  "job_id": "job-123"
}

Response:
{
  "status": "cancelled",
  "job_id": "job-123"
}
```

**Implementation:** See `COMPONENT_RESPONSIBILITIES_FINAL.md` Section "Worker Cancellation"

**Priority:** HIGH (will happen during development)

---

### 3. Orphaned Worker Cleanup ⏳
**File:** `bin/rbees-pool/src/commands/worker.rs`

**Task:** Add `cleanup` subcommand

**Command:**
```bash
rbees-pool worker cleanup
```

**Implementation:**
```rust
pub fn cleanup_orphaned_workers() -> Result<()> {
    let workers = read_worker_files()?;
    let mut cleaned = 0;
    
    for worker in workers {
        if !is_process_alive(worker.pid) {
            println!("Found orphaned worker: {}", worker.worker_id);
            
            // Remove metadata file
            remove_file(format!(".runtime/workers/{}.json", worker.worker_id))?;
            
            cleaned += 1;
        }
    }
    
    println!("Cleaned up {} orphaned workers", cleaned);
    Ok(())
}
```

**Priority:** HIGH (will happen a lot during development)

---

## CP4 Tasks (Multi-Model Testing)

### 1. Download Remaining Models
```bash
rbees-pool models download tinyllama
rbees-pool models download phi3
rbees-pool models download mistral
```

**Priority:** HIGH (CP4 requirement)

---

### 2. Create Test Script
**File:** `.docs/testing/test_all_models.sh`

**Priority:** HIGH (CP4 requirement)

---

### 3. Document Results
**File:** `MODEL_SUPPORT.md`

**Priority:** HIGH (CP4 requirement)

---

## M0 Completion Checklist

### Core Features:
- [x] Workers (rbees-workerd) ✅
- [x] Pool CLI (rbees-pool) ✅
- [x] Remote CLI (llorch) ✅
- [x] Model catalog ✅
- [x] Worker spawning ✅
- [x] Token generation ✅
- [x] Inference testing (llorch infer) ✅

### Missing Features (M0):
- [ ] Backend catalog detection
- [ ] Worker cancellation
- [ ] Orphaned worker cleanup

### CP4 (Before M1):
- [ ] Download all models
- [ ] Test all backends
- [ ] Document results

---

## After M0 Complete → Start M1

### M1 Tasks (rbees-orcd):
1. HTTP server (port 8080)
2. Worker registry (SQLite)
3. Rhai scripting engine
4. Prompt constructor (shared crate)
5. Admission control
6. Queue management
7. Scheduling
8. SSE relay

**Estimated Time:** 2-3 weeks

---

## Quick Commands

### Test Current State:
```bash
# Build
cargo build --release

# Test worker
llorch infer --worker localhost:8080 --prompt "Hello" --max-tokens 20

# Check catalog
rbees-pool models catalog

# List workers
rbees-pool worker list
```

### Cleanup Orphans (Manual):
```bash
# Find orphaned workers
ps aux | grep rbees-workerd

# Kill manually
kill <PID>

# Remove metadata
rm .runtime/workers/*.json
```

### Test Cancellation (When Implemented):
```bash
# Start long inference
llorch infer --worker localhost:8080 --prompt "Write a long story" --max-tokens 1000 &

# Cancel it
curl -X POST http://localhost:8080/cancel -d '{"job_id":"<job-id>"}'
```

---

**Priority Order:**
1. CP4 (download models, test, document)
2. Orphaned worker cleanup
3. Worker cancellation
4. Backend catalog detection
5. M1 (rbees-orcd)

---

**Signed:** TEAM-024  
**Status:** Ready for TEAM-025

# FT-014: VRAM Residency Verification

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: S (1 day)  
**Days**: 27 - 27  
**Spec Ref**: M0-W-1012, CUDA-5421

---

## Story Description

Implement periodic VRAM residency verification to detect RAM fallback or UMA violations. This provides runtime safety checks to ensure VRAM-only policy is maintained throughout worker lifetime.

---

## Acceptance Criteria

- [ ] Health check function verifies all model weights are in VRAM
- [ ] Uses cudaPointerGetAttributes to validate pointer type
- [ ] Checks pointer type is cudaMemoryTypeDevice (not managed/host)
- [ ] Checks no host pointer exists (hostPointer == nullptr)
- [ ] Periodic check runs every 60 seconds (configurable)
- [ ] Worker marks itself unhealthy if residency check fails
- [ ] Unit tests validate residency checking logic
- [ ] Integration tests validate detection of RAM fallback
- [ ] Health endpoint exposes residency status

---

## Dependencies

### Upstream (Blocks This Story)
- FT-011: VRAM tracking (Expected completion: Day 23)
- FT-013: Device memory RAII (Expected completion: Day 26)

### Downstream (This Story Blocks)
- FT-026: Error handling integration needs residency checks
- Health endpoint (FT-001) needs residency status

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/health.h` - Health check interface
- `bin/worker-orcd/cuda/src/health.cpp` - Health implementation
- `bin/worker-orcd/src/health/mod.rs` - Rust health module
- `bin/worker-orcd/cuda/tests/health_test.cpp` - Unit tests

### Key Interfaces
```cpp
// health.h
#ifndef WORKER_HEALTH_H
#define WORKER_HEALTH_H

#include <cuda_runtime.h>
#include "vram_tracker.h"
#include "cuda_error.h"

namespace worker {

class Health {
public:
    /**
     * Check VRAM residency for all tracked allocations.
     * 
     * @param tracker VRAM tracker with allocations to check
     * @return true if all allocations are VRAM-resident, false otherwise
     */
    static bool check_vram_residency(const VramTracker& tracker);
    
    /**
     * Check VRAM residency for specific pointer.
     * 
     * @param ptr Device pointer to check
     * @return true if pointer is VRAM-resident, false otherwise
     */
    static bool check_pointer_residency(const void* ptr);
    
    /**
     * Get process-wide VRAM usage.
     * 
     * @return VRAM bytes used by this process
     */
    static uint64_t get_process_vram_usage();
    
    /**
     * Get detailed residency report.
     * 
     * @param tracker VRAM tracker
     * @return Human-readable report
     */
    static std::string residency_report(const VramTracker& tracker);
};

} // namespace worker

#endif // WORKER_HEALTH_H

// health.cpp
#include "health.h"
#include <sstream>

namespace worker {

bool Health::check_vram_residency(const VramTracker& tracker) {
    // Use VramTracker's built-in verification
    return tracker.verify_vram_residency();
}

bool Health::check_pointer_residency(const void* ptr) {
    if (ptr == nullptr) {
        return false;
    }
    
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    
    if (err != cudaSuccess) {
        // Clear error
        cudaGetLastError();
        return false;
    }
    
    // Verify pointer is device memory (not managed/host)
    if (attrs.type != cudaMemoryTypeDevice) {
        return false;
    }
    
    // Verify no host pointer (no UMA)
    if (attrs.hostPointer != nullptr) {
        return false;
    }
    
    return true;
}

uint64_t Health::get_process_vram_usage() {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (err != cudaSuccess) {
        return 0;
    }
    
    return total_bytes - free_bytes;
}

std::string Health::residency_report(const VramTracker& tracker) {
    std::ostringstream oss;
    
    bool resident = check_vram_residency(tracker);
    uint64_t process_usage = get_process_vram_usage();
    uint64_t tracked_usage = tracker.total_usage();
    
    oss << "VRAM Residency Report:\n";
    oss << "  Status: " << (resident ? "RESIDENT" : "VIOLATION") << "\n";
    oss << "  Process VRAM Usage: " << (process_usage / 1024.0 / 1024.0) << " MB\n";
    oss << "  Tracked VRAM Usage: " << (tracked_usage / 1024.0 / 1024.0) << " MB\n";
    oss << "  Allocations: " << tracker.allocation_count() << "\n";
    
    if (!resident) {
        oss << "  WARNING: RAM fallback or UMA detected!\n";
    }
    
    return oss.str();
}

} // namespace worker

// FFI wrapper
extern "C" bool cuda_check_vram_residency(CudaModel* model, int* error_code) {
    try {
        auto* m = reinterpret_cast<worker::Model*>(model);
        bool resident = worker::Health::check_vram_residency(m->vram_tracker());
        *error_code = CUDA_SUCCESS;
        return resident;
    } catch (const worker::CudaError& e) {
        *error_code = e.code();
        return false;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    }
}

extern "C" uint64_t cuda_get_process_vram_usage(CudaContext* ctx) {
    return worker::Health::get_process_vram_usage();
}

// Rust integration
// src/health/mod.rs
use std::time::{Duration, Instant};
use tokio::time;
use tracing::{error, info, warn};

pub struct HealthMonitor {
    last_check: Instant,
    check_interval: Duration,
    is_healthy: bool,
}

impl HealthMonitor {
    pub fn new(check_interval_secs: u64) -> Self {
        Self {
            last_check: Instant::now(),
            check_interval: Duration::from_secs(check_interval_secs),
            is_healthy: true,
        }
    }
    
    pub async fn run(&mut self, model: &crate::cuda::CudaModel) {
        loop {
            time::sleep(self.check_interval).await;
            
            match model.check_vram_residency() {
                Ok(true) => {
                    if !self.is_healthy {
                        info!("VRAM residency check passed, worker now healthy");
                        self.is_healthy = true;
                    }
                }
                Ok(false) => {
                    error!("VRAM residency check FAILED: RAM fallback detected");
                    self.is_healthy = false;
                }
                Err(e) => {
                    error!("VRAM residency check error: {}", e);
                    self.is_healthy = false;
                }
            }
            
            self.last_check = Instant::now();
        }
    }
    
    pub fn is_healthy(&self) -> bool {
        self.is_healthy
    }
    
    pub fn last_check_elapsed(&self) -> Duration {
        self.last_check.elapsed()
    }
}

#[derive(Debug, serde::Serialize)]
pub struct HealthStatus {
    pub status: String,
    pub resident: bool,
    pub vram_bytes_used: u64,
    pub last_check_secs: u64,
}

impl HealthStatus {
    pub fn from_monitor(monitor: &HealthMonitor, model: &crate::cuda::CudaModel) -> Self {
        let resident = model.check_vram_residency().unwrap_or(false);
        let status = if monitor.is_healthy() && resident {
            "healthy".to_string()
        } else {
            "unhealthy".to_string()
        };
        
        Self {
            status,
            resident,
            vram_bytes_used: model.vram_usage(),
            last_check_secs: monitor.last_check_elapsed().as_secs(),
        }
    }
}
```

### Implementation Notes
- cudaPointerGetAttributes validates pointer type and location
- Periodic check runs in background task (tokio::spawn)
- Worker marks itself unhealthy if residency check fails
- Health endpoint returns residency status in response
- Check interval configurable (default: 60 seconds)
- No performance impact (check runs in background)
- Logs critical error if RAM fallback detected

---

## Testing Strategy

### Unit Tests
- Test check_pointer_residency() with device pointer returns true
- Test check_pointer_residency() with host pointer returns false
- Test check_pointer_residency() with nullptr returns false
- Test get_process_vram_usage() returns positive value
- Test residency_report() generates readable output
- Test HealthMonitor tracks healthy/unhealthy state

### Integration Tests
- Test residency check with real VRAM allocations
- Test residency check detects RAM fallback (simulate with cudaMallocManaged)
- Test periodic health check runs in background
- Test health endpoint exposes residency status
- Test worker marks itself unhealthy on residency failure

### Manual Verification
1. Start worker: `cargo run --features cuda`
2. Query health: `curl http://localhost:8080/health`
3. Verify `resident: true` in response
4. Check logs for periodic residency checks

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing (5+ tests)
- [ ] Documentation updated (Health class docs, HealthMonitor docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §2.2 VRAM Residency Verification (M0-W-1012, CUDA-5421)
- Related Stories: FT-011 (VRAM tracking), FT-001 (health endpoint)
- CUDA Pointer Attributes: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g0c38f4e0e21a3d5c8de6e28f2f5b7e8a

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **Residency check passed** (ACTION_VERIFY)
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VERIFY,
       target: format!("GPU{}", device_id),
       device: Some(format!("GPU{}", device_id)),
       human: format!("VRAM residency verified on GPU{}: all {} allocations resident", device_id, allocation_count),
       ..Default::default()
   });
   ```

2. **Residency check FAILED** (CRITICAL)
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VERIFY,
       target: format!("GPU{}", device_id),
       device: Some(format!("GPU{}", device_id)),
       error_kind: Some("residency_violation".to_string()),
       human: format!("CRITICAL: VRAM residency violated on GPU{}: RAM fallback or UMA detected", device_id),
       ..Default::default()
   });
   ```

3. **Worker marked unhealthy**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "health_status",
       target: "worker".to_string(),
       human: "Worker marked unhealthy due to VRAM residency violation".to_string(),
       ..Default::default()
   });
   ```

**Why this matters**: VRAM residency violations are CRITICAL policy failures. Narration ensures these are immediately visible in logs and alerts.

---
*Narration guidance added by Narration-Core Team 🎀*

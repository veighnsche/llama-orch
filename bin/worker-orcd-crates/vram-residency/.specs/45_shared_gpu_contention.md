# 45. Shared GPU Contention - Home/Desktop Use

**Status**: ✅ **DOCUMENTED**  
**Created**: 2025-10-02  
**Priority**: **CRITICAL** - Home/desktop deployment requirement  
**Applies to**: `worker-orcd`, `pool-managerd`, `vram-residency`

---

## 0. Executive Summary

**Context**: On home/desktop systems, the GPU is shared between llama-orch and user applications (gaming, video editing, 3D rendering, browser GPU acceleration, etc.).

**Question**: What happens when llama-orch and user applications compete for VRAM?

**Answer**: ✅ **Graceful failure** - Returns `InsufficientVram` error, does NOT crash or evict user applications.

**Critical Issue**: User applications may fail to start if llama-orch is using VRAM. This requires **automatic detection and eviction** for acceptable UX.

---

## 1. VRAM Contention Scenarios

### Scenario 1: User Application Active + Model Load Request

**Setup**:
- GPU: 12GB VRAM total
- User application (game/video editor/3D render): Using 10GB VRAM
- Available: 2GB VRAM
- Model request: 8GB model

**What Happens**:

```
1. User launches GPU-intensive application
   └─> Application allocates 10GB VRAM
   
2. llama-orch receives inference request
   └─> Attempts to load 8GB model
   
3. VramManager.seal_model() called
   └─> Checks available VRAM: get_free_vram() → 2GB
   └─> Compares: need 8GB > have 2GB
   └─> Returns: Err(VramError::InsufficientVram(8GB, 2GB))
   
4. Request fails gracefully
   └─> No crash
   └─> No eviction of game
   └─> Error propagated to client
```

**Result**: ✅ **Safe failure** - User application continues uninterrupted

---

### Scenario 2: Model Loaded + User Launches Application ⚠️ **CRITICAL UX ISSUE**

**Setup**:
- GPU: 12GB VRAM total
- llama-orch: Using 8GB VRAM (model loaded and sealed)
- User launches GPU-intensive application that needs 10GB

**What Happens**:

```
1. llama-orch loads 8GB model
   └─> Model sealed in VRAM (8GB allocated)
   └─> Available: 4GB
   
2. User launches game
   └─> Game requests 10GB from CUDA driver
   └─> CUDA driver checks: need 10GB, have 4GB
   └─> CUDA returns: cudaErrorMemoryAllocation
   
3. Game's response (depends on game implementation):
   
   Option A: Hard failure (most common)
   ├─> Game shows error: "Insufficient GPU memory"
   ├─> Game refuses to start
   └─> User is confused (doesn't know about llama-orch)
   
   Option B: Automatic quality reduction (rare)
   ├─> Game detects available VRAM (4GB)
   ├─> Automatically reduces texture quality
   ├─> Lowers resolution/effects
   └─> Runs at degraded quality
   
   Option C: User prompt (some games)
   ├─> "Insufficient VRAM detected"
   ├─> "Close other GPU applications and retry?"
   └─> User must manually investigate
```

**Result**: ⚠️ **MAJOR UX PROBLEM**
- ❌ Game can't start or runs poorly
- ❌ User doesn't know llama-orch is the cause
- ❌ No automatic resolution
- ❌ Requires manual intervention
- ❌ Poor user experience

**Why This Is Worse Than Scenario 1**:
- Gaming is typically **interactive** and **user-initiated**
- Inference is often **background** and **automated**
- User expects gaming to "just work"
- User may not even know llama-orch is running

---

### Scenario 3: Multiple Model Load Requests

**Setup**:
- GPU: 12GB VRAM total
- Request 1: 6GB model
- Request 2: 8GB model (concurrent)

**What Happens**:

```
1. Request 1 starts
   └─> Checks: need 6GB, have 12GB ✓
   └─> Allocates 6GB
   
2. Request 2 starts (before Request 1 completes)
   └─> Checks: need 8GB, have 12GB ✓ (race condition!)
   └─> Attempts to allocate 8GB
   └─> CUDA allocation fails (only 6GB actually available)
   └─> Returns: Err(VramError::CudaAllocationFailed)
```

**Result**: ⚠️ **Race condition possible** - Check-then-allocate is not atomic

---

## 2. Current Protection Mechanisms

### ✅ What Works

1. **Pre-allocation check** (vram_manager.rs:158-174)
   ```rust
   let available = self.context.get_free_vram()?;
   if vram_needed > available {
       return Err(VramError::InsufficientVram(vram_needed, available));
   }
   ```

2. **Graceful error handling**
   - No panic on OOM
   - Error is retriable: `VramError::is_retriable() → true`
   - Audit event emitted: `VramAllocationFailed`

3. **CUDA driver protection**
   - CUDA driver prevents over-allocation
   - Returns error if allocation would exceed capacity
   - Does NOT evict other processes

### ⚠️ What Doesn't Work

1. **No reservation system**
   - Check-then-allocate race condition
   - Multiple concurrent requests can pass the check

2. **No priority system**
   - llama-orch has same priority as gaming
   - Cannot preempt or request priority

3. **No capacity planning**
   - No way to reserve VRAM in advance
   - No admission control based on expected load

---

## 3. Recommended Solutions

### Solution 0: GPU Process Detection & Auto-Eviction (CRITICAL for Desktop/Home Use)

**Where**: `worker-orcd` or system daemon

**Problem**: User can't start GPU-intensive application because llama-orch is using VRAM

**Solution**: Detect other GPU processes and automatically free VRAM

**Implementation (RECOMMENDED - NVML API)**:
```rust
// Use NVIDIA Management Library (NVML) to detect ANY GPU process
// No hardcoded app list - detects ALL GPU usage automatically
use nvml_wrapper::Nvml;

struct GpuProcessMonitor {
    nvml: Nvml,
    gpu_device: u32,
    our_pid: u32,
}

impl GpuProcessMonitor {
    fn new(gpu_device: u32) -> Result<Self> {
        Ok(Self {
            nvml: Nvml::init()?,
            gpu_device,
            our_pid: std::process::id(),
        })
    }
    
    /// Check if ANY other process is using the GPU
    /// Returns true if we should evict our models
    fn should_evict(&self) -> Result<bool> {
        let device = self.nvml.device_by_index(self.gpu_device)?;
        
        // Get ALL processes using this GPU (compute + graphics)
        let compute_procs = device.running_compute_processes()?;
        let graphics_procs = device.running_graphics_processes()?;
        
        // Check if any process other than us is using GPU
        let other_compute = compute_procs.iter()
            .any(|p| p.pid != self.our_pid);
        let other_graphics = graphics_procs.iter()
            .any(|p| p.pid != self.our_pid);
        
        Ok(other_compute || other_graphics)
    }
    
    /// Get details of other GPU processes (for logging)
    fn get_other_processes(&self) -> Result<Vec<GpuProcessInfo>> {
        let device = self.nvml.device_by_index(self.gpu_device)?;
        
        let mut all_procs = Vec::new();
        all_procs.extend(device.running_compute_processes()?);
        all_procs.extend(device.running_graphics_processes()?);
        
        Ok(all_procs.into_iter()
            .filter(|p| p.pid != self.our_pid)
            .collect())
    }
}

// In worker-orcd main loop
async fn monitor_gpu_and_evict() {
    let monitor = GpuProcessMonitor::new(0).expect("Failed to init NVML");
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    
    loop {
        interval.tick().await;
        
        match monitor.should_evict() {
            Ok(true) => {
                // Other processes detected - evict our models
                let other_procs = monitor.get_other_processes().unwrap_or_default();
                tracing::warn!(
                    other_process_count = other_procs.len(),
                    "Other GPU processes detected, evicting models"
                );
                
                // Gracefully unload all models
                for shard in active_shards.drain(..) {
                    vram_manager.deallocate(&shard)?;
                }
                
                // Notify user
                notify_user("Models unloaded to free GPU for other applications");
                
                // Pause worker
                worker_state.set_paused(true);
            }
            Ok(false) if worker_state.is_paused() => {
                // No other processes - resume
                tracing::info!("GPU available again, resuming inference");
                worker_state.set_paused(false);
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to check GPU processes");
            }
            _ => {}
        }
    }
}
```

**Why This Approach is Better**:
- ✅ **No hardcoded app list** - Detects ANY GPU usage
- ✅ **Automatic** - Works with all applications (games, video editors, browsers, etc.)
- ✅ **Accurate** - Uses NVIDIA's official API
- ✅ **Maintainable** - No list to update
- ✅ **Respects user** - Doesn't assume what they run
- ✅ **Future-proof** - Works with apps that don't exist yet

**Benefits**:
- Detects compute processes (ML, rendering, encoding)
- Detects graphics processes (games, desktop compositors)
- Excludes self (doesn't evict because of own process)
- Provides process details for logging/debugging

---

### Solution 1: Admission Control (Orchestrator Level)

**Where**: `pool-managerd` or `worker-orcd`

**Implementation**:
```rust
// In pool-managerd
struct GpuPool {
    total_vram: usize,
    reserved_vram: AtomicUsize,  // Track reservations
    allocated_vram: AtomicUsize, // Track actual allocations
}

impl GpuPool {
    fn can_admit(&self, model_size: usize) -> bool {
        let current_reserved = self.reserved_vram.load(Ordering::SeqCst);
        let new_total = current_reserved + model_size;
        
        // Leave 10% headroom for system/gaming
        let max_allowed = (self.total_vram * 90) / 100;
        
        new_total <= max_allowed
    }
    
    fn reserve(&self, model_size: usize) -> Result<Reservation> {
        // Atomic compare-and-swap to prevent race
        loop {
            let current = self.reserved_vram.load(Ordering::SeqCst);
            let new = current + model_size;
            
            if new > self.max_allowed() {
                return Err(InsufficientCapacity);
            }
            
            if self.reserved_vram.compare_exchange(
                current, new, 
                Ordering::SeqCst, 
                Ordering::SeqCst
            ).is_ok() {
                return Ok(Reservation { size: model_size });
            }
        }
    }
}
```

**Benefits**:
- ✅ Prevents race conditions
- ✅ Leaves headroom for user apps
- ✅ Atomic reservation

---

### Solution 2: Graceful Degradation

**Where**: Client/API level

**Implementation**:
```rust
// In client code
async fn load_model_with_retry(model: &str) -> Result<Handle> {
    let mut attempts = 0;
    let max_attempts = 3;
    
    loop {
        match load_model(model).await {
            Ok(handle) => return Ok(handle),
            Err(e) if e.is_retriable() && attempts < max_attempts => {
                attempts += 1;
                
                // Exponential backoff
                let delay = Duration::from_secs(2u64.pow(attempts));
                tokio::time::sleep(delay).await;
                
                // Optionally: try smaller model variant
                if attempts == 2 {
                    model = model.replace("70b", "13b");
                }
            }
            Err(e) => return Err(e),
        }
    }
}
```

**Benefits**:
- ✅ Automatic retry on transient failures
- ✅ Can fallback to smaller models
- ✅ User-friendly error messages

---

### Solution 3: VRAM Headroom Policy

**Where**: `vram-residency` configuration

**Implementation**:
```rust
pub struct VramConfig {
    pub gpu_device: u32,
    pub max_model_size: usize,
    pub reserved_system_vram: usize,  // NEW: Reserve for system/gaming
}

impl VramManager {
    fn effective_available_vram(&self) -> Result<usize> {
        let total = self.context.get_free_vram()?;
        let reserved = self.config.reserved_system_vram;
        
        // Never use the last 10% or 2GB (whichever is larger)
        let min_reserved = std::cmp::max(
            reserved,
            std::cmp::max(total / 10, 2 * 1024 * 1024 * 1024)
        );
        
        Ok(total.saturating_sub(min_reserved))
    }
}
```

**Benefits**:
- ✅ Always leaves headroom for user apps
- ✅ Configurable per deployment
- ✅ Prevents "greedy" behavior

---

## 4. User Experience Considerations

### For Gaming Users

**Problem**: User starts game, llama-orch already using VRAM

**Solutions**:
1. **Detection**: Monitor for gaming apps (Steam, Epic, etc.)
2. **Auto-pause**: Pause inference when game detected
3. **Priority**: Give gaming higher priority
4. **Notification**: Warn user before starting inference

**Example**:
```rust
// In worker-orcd
if gaming_app_detected() {
    if config.auto_pause_for_gaming {
        pause_inference_workers();
        notify_user("Inference paused for gaming");
    } else {
        warn_user("Gaming may be impacted by active inference");
    }
}
```

---

### For Inference Users

**Problem**: Inference request fails because user is gaming

**Solutions**:
1. **Clear error messages**: "GPU busy with other applications"
2. **Retry suggestions**: "Try again in a few minutes"
3. **Alternative**: "Use CPU inference (slower)"
4. **Queue**: "Request queued, will run when GPU available"

**Example Error Response**:
```json
{
  "error": "insufficient_vram",
  "message": "GPU has insufficient VRAM (need 8GB, have 2GB available)",
  "details": "GPU may be in use by other applications (gaming, video editing, etc.)",
  "suggestions": [
    "Close GPU-intensive applications",
    "Try a smaller model variant",
    "Wait and retry in a few minutes"
  ],
  "retry_after": 300
}
```

---

## 5. Production Deployment Recommendations

### Dedicated Inference GPU (Recommended)

**Setup**:
- Multi-GPU system
- GPU 0: User applications (gaming, desktop)
- GPU 1: llama-orch inference (dedicated)

**Benefits**:
- ✅ No contention
- ✅ Predictable performance
- ✅ User experience unaffected

**Configuration**:
```toml
[worker]
gpu_device = 1  # Use second GPU
allow_shared_gpu = false  # Fail if GPU has other processes
```

---

### Shared GPU (Home/Desktop Use)

**Setup**:
- Single GPU system
- Shared between user apps and inference

**Requirements**:
- ✅ VRAM headroom policy (leave 2GB minimum)
- ✅ Auto-pause on gaming detection
- ✅ Clear user notifications
- ✅ Retry logic in clients

**Configuration**:
```toml
[worker]
gpu_device = 0
allow_shared_gpu = true
reserved_system_vram = 2147483648  # 2GB
auto_pause_for_gaming = true
```

---

## 6. Monitoring & Alerts

### Metrics to Track

```rust
// Prometheus metrics
vram_allocation_failures_total{reason="insufficient_vram"}
vram_allocation_failures_total{reason="cuda_error"}
vram_utilization_percent{gpu="0"}
vram_headroom_bytes{gpu="0"}
concurrent_allocation_attempts_total
```

### Alerts

```yaml
# Alert if VRAM allocation failures spike
- alert: HighVramAllocationFailures
  expr: rate(vram_allocation_failures_total[5m]) > 0.1
  annotations:
    summary: "High rate of VRAM allocation failures"
    description: "GPU may be oversubscribed or contended"

# Alert if VRAM headroom too low
- alert: LowVramHeadroom
  expr: vram_headroom_bytes < 2147483648  # 2GB
  annotations:
    summary: "Low VRAM headroom"
    description: "May impact user applications"
```

---

## 7. Testing Scenarios

### Test 1: Concurrent Allocation

```rust
#[test]
fn test_concurrent_allocation_contention() {
    let manager = Arc::new(Mutex::new(VramManager::new()));
    let mut handles = vec![];
    
    // Spawn 10 threads trying to allocate 2GB each
    for _ in 0..10 {
        let mgr = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let data = vec![0u8; 2 * 1024 * 1024 * 1024];
            mgr.lock().unwrap().seal_model(&data, 0)
        });
        handles.push(handle);
    }
    
    // Some should succeed, some should fail with InsufficientVram
    let mut success = 0;
    let mut failures = 0;
    
    for handle in handles {
        match handle.join().unwrap() {
            Ok(_) => success += 1,
            Err(VramError::InsufficientVram(_, _)) => failures += 1,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    assert!(success > 0, "Some allocations should succeed");
    assert!(failures > 0, "Some allocations should fail");
}
```

---

### Test 2: Headroom Enforcement

```rust
#[test]
fn test_vram_headroom_enforced() {
    let mut config = VramConfig::default();
    config.reserved_system_vram = 2 * 1024 * 1024 * 1024; // 2GB
    
    let manager = VramManager::new_with_config(config);
    let total = manager.total_vram().unwrap();
    let available = manager.effective_available_vram().unwrap();
    
    // Should reserve at least 2GB
    assert!(total - available >= 2 * 1024 * 1024 * 1024);
}
```

---

## 8. Summary

### Current Behavior: ✅ Safe

- Graceful failure on insufficient VRAM
- No crashes or evictions
- Retriable errors
- Audit logging

### Recommended Improvements:

1. **Admission control** at orchestrator level (prevents races)
2. **VRAM headroom policy** (leaves space for user apps)
3. **Gaming detection** (auto-pause or warn)
4. **Clear error messages** (user-friendly)
5. **Retry logic** in clients (handle transient failures)

### Deployment Modes:

- **Dedicated GPU**: Best for production servers
- **Shared GPU**: Requires headroom policy + user awareness

### Next Steps:

- [ ] Implement admission control in pool-managerd
- [ ] Add VRAM headroom configuration
- [ ] Add gaming app detection (optional)
- [ ] Improve error messages with suggestions
- [ ] Add monitoring/alerting for VRAM contention

**Status**: ✅ Current behavior is safe, improvements recommended for production

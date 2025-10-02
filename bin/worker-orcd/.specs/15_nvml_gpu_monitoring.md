# 15. NVML GPU Monitoring & Process Detection

**Status**: ✅ **REQUIRED**  
**Priority**: **CRITICAL** for home/desktop deployments  
**Created**: 2025-10-02  
**Applies to**: `worker-orcd`, `pool-managerd`

---

## 0. Executive Summary

**NVML (NVIDIA Management Library)** is **CRITICAL** for worker-orcd to:
1. Detect other GPU processes (gaming, video editing, etc.)
2. Automatically evict models when users need the GPU
3. Monitor GPU health (temperature, utilization, errors)
4. Enable intelligent capacity planning

**Without NVML**: Home/desktop users cannot use their GPU for anything else while llama-orch is running → **Unacceptable UX**

---

## 1. Requirements

### WORKER-5000: NVML Integration (CRITICAL)

**Priority**: CRITICAL for home/desktop, HIGH for servers

worker-orcd MUST integrate NVML for:
- GPU process detection
- Health monitoring
- Capacity planning
- Error recovery

### WORKER-5001: GPU Process Detection (CRITICAL)

**Priority**: CRITICAL for home/desktop

worker-orcd MUST:
- Poll GPU for other processes every 5 seconds
- Detect compute processes (ML, rendering, encoding)
- Detect graphics processes (games, desktop compositors)
- Exclude self from detection
- Automatically evict models when other processes detected
- Resume when other processes stop

**Rationale**: On shared GPUs (home/desktop), users must be able to use GPU for other applications without manually managing llama-orch.

### WORKER-5002: GPU Health Monitoring (HIGH)

**Priority**: HIGH for all deployments

worker-orcd SHOULD:
- Monitor GPU temperature
- Monitor GPU utilization
- Monitor power usage
- Monitor ECC errors
- Throttle or pause on thermal issues
- Alert on GPU errors

**Rationale**: Prevents hardware damage and ensures reliability.

### WORKER-5003: Capacity Planning (MEDIUM)

**Priority**: MEDIUM

worker-orcd SHOULD:
- Query actual VRAM available
- Track VRAM usage over time
- Make intelligent model placement decisions
- Emit capacity metrics

**Rationale**: Enables better resource utilization and placement.

---

## 2. API Overview

### NVML Capabilities

```rust
use nvml_wrapper::Nvml;

// Initialize NVML
let nvml = Nvml::init()?;
let device = nvml.device_by_index(0)?;

// Process detection (CRITICAL)
let compute_procs = device.running_compute_processes()?;
let graphics_procs = device.running_graphics_processes()?;

// Health monitoring (HIGH)
let temperature = device.temperature(TemperatureSensor::Gpu)?;
let utilization = device.utilization_rates()?;
let power = device.power_usage()?;

// Memory info (MEDIUM)
let memory_info = device.memory_info()?;
let free_vram = memory_info.free;
let total_vram = memory_info.total;

// Error detection (HIGH)
let ecc_errors = device.total_ecc_errors(ErrorType::Corrected)?;
let perf_state = device.performance_state()?;
```

---

## 3. Implementation

### 3.1 Process Detection (CRITICAL)

**Location**: `worker-orcd/src/gpu_monitor.rs`

```rust
use nvml_wrapper::Nvml;
use std::time::Duration;
use tokio::time::interval;

pub struct GpuProcessMonitor {
    nvml: Nvml,
    gpu_device: u32,
    our_pid: u32,
}

impl GpuProcessMonitor {
    pub fn new(gpu_device: u32) -> Result<Self> {
        Ok(Self {
            nvml: Nvml::init()?,
            gpu_device,
            our_pid: std::process::id(),
        })
    }
    
    /// Check if ANY other process is using the GPU
    pub fn should_evict(&self) -> Result<bool> {
        let device = self.nvml.device_by_index(self.gpu_device)?;
        
        // Get ALL processes using this GPU
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
    pub fn get_other_processes(&self) -> Result<Vec<ProcessInfo>> {
        let device = self.nvml.device_by_index(self.gpu_device)?;
        
        let mut all_procs = Vec::new();
        all_procs.extend(device.running_compute_processes()?);
        all_procs.extend(device.running_graphics_processes()?);
        
        Ok(all_procs.into_iter()
            .filter(|p| p.pid != self.our_pid)
            .collect())
    }
}

// Main monitoring loop
pub async fn monitor_and_evict(
    monitor: GpuProcessMonitor,
    vram_manager: Arc<Mutex<VramManager>>,
    worker_state: Arc<WorkerState>,
) {
    let mut interval = interval(Duration::from_secs(5));
    
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
                let mut mgr = vram_manager.lock().await;
                for shard in mgr.active_shards().drain(..) {
                    mgr.deallocate(&shard)?;
                }
                
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

---

### 3.2 Health Monitoring (HIGH)

**Location**: `worker-orcd/src/gpu_health.rs`

```rust
pub struct GpuHealthMonitor {
    nvml: Nvml,
    gpu_device: u32,
    temp_threshold: u32,  // Default: 85°C
}

impl GpuHealthMonitor {
    pub fn check_health(&self) -> Result<GpuHealth> {
        let device = self.nvml.device_by_index(self.gpu_device)?;
        
        let temperature = device.temperature(TemperatureSensor::Gpu)?;
        let utilization = device.utilization_rates()?;
        let power = device.power_usage()?;
        
        Ok(GpuHealth {
            temperature,
            gpu_utilization: utilization.gpu,
            memory_utilization: utilization.memory,
            power_watts: power / 1000,  // mW to W
            healthy: temperature < self.temp_threshold,
        })
    }
    
    pub async fn monitor_health(
        &self,
        worker_state: Arc<WorkerState>,
    ) {
        let mut interval = interval(Duration::from_secs(10));
        
        loop {
            interval.tick().await;
            
            match self.check_health() {
                Ok(health) => {
                    // Emit metrics
                    metrics::gauge!("gpu_temperature_celsius", health.temperature as f64);
                    metrics::gauge!("gpu_utilization_percent", health.gpu_utilization as f64);
                    metrics::gauge!("gpu_power_watts", health.power_watts as f64);
                    
                    // Check thermal throttling
                    if !health.healthy {
                        tracing::warn!(
                            temperature = health.temperature,
                            threshold = self.temp_threshold,
                            "GPU temperature too high, throttling"
                        );
                        worker_state.set_throttled(true);
                    } else if worker_state.is_throttled() {
                        tracing::info!("GPU temperature normal, resuming");
                        worker_state.set_throttled(false);
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to check GPU health");
                }
            }
        }
    }
}
```

---

## 4. Configuration

### 4.1 Enable/Disable NVML

```toml
[worker]
# Enable NVML monitoring (CRITICAL for home/desktop)
enable_nvml = true

# GPU process detection
auto_evict_for_user_apps = true
process_check_interval_secs = 5

# Health monitoring
enable_health_monitoring = true
health_check_interval_secs = 10
temperature_threshold_celsius = 85

# Capacity planning
enable_capacity_tracking = true
```

### 4.2 Deployment Profiles

**Home/Desktop** (shared GPU):
```toml
enable_nvml = true  # MANDATORY
auto_evict_for_user_apps = true  # MANDATORY
enable_health_monitoring = true
```

**Dedicated Server** (inference only):
```toml
enable_nvml = true  # RECOMMENDED
auto_evict_for_user_apps = false  # Not needed
enable_health_monitoring = true  # RECOMMENDED
```

---

## 5. Dependencies

### 5.1 Rust Crate

```toml
[dependencies]
nvml-wrapper = "0.9"  # NVIDIA Management Library bindings
```

### 5.2 System Requirements

**Linux** (CachyOS/Arch):
```bash
# Install NVIDIA drivers (includes NVML)
sudo pacman -S nvidia nvidia-utils

# Verify NVML is available
nvidia-smi  # Uses NVML under the hood
```

**Library path**:
- `/usr/lib/libnvidia-ml.so` (provided by `nvidia-utils`)

---

## 6. Error Handling

### 6.1 NVML Initialization Failure

```rust
match Nvml::init() {
    Ok(nvml) => {
        // NVML available - enable monitoring
        tracing::info!("NVML initialized successfully");
    }
    Err(e) => {
        if config.require_nvml {
            // CRITICAL: Fail fast if NVML required
            return Err(WorkerError::NvmlRequired(e));
        } else {
            // WARN: Continue without NVML (degraded mode)
            tracing::warn!(
                error = %e,
                "NVML not available, GPU monitoring disabled"
            );
        }
    }
}
```

### 6.2 Query Failures

```rust
match monitor.should_evict() {
    Ok(should_evict) => {
        // Handle normally
    }
    Err(e) => {
        // Log error but don't crash
        tracing::error!(error = %e, "NVML query failed");
        
        // Increment error counter
        metrics::counter!("nvml_query_errors_total", 1);
        
        // If too many errors, disable NVML
        if error_count > 10 {
            tracing::error!("Too many NVML errors, disabling monitoring");
            disable_nvml_monitoring();
        }
    }
}
```

---

## 7. Metrics & Observability

### 7.1 Prometheus Metrics

```rust
// Process detection
metrics::gauge!("gpu_other_processes_count", other_procs.len() as f64);
metrics::counter!("gpu_evictions_total", 1);

// Health monitoring
metrics::gauge!("gpu_temperature_celsius", temperature as f64);
metrics::gauge!("gpu_utilization_percent", utilization as f64);
metrics::gauge!("gpu_power_watts", power_watts as f64);
metrics::gauge!("gpu_memory_free_bytes", free_vram as f64);
metrics::gauge!("gpu_memory_total_bytes", total_vram as f64);

// Errors
metrics::counter!("nvml_query_errors_total", 1);
metrics::counter!("nvml_init_failures_total", 1);
```

### 7.2 Structured Logging

```rust
tracing::info!(
    gpu_device = gpu_device,
    other_processes = other_procs.len(),
    "GPU process detection active"
);

tracing::warn!(
    temperature = temp,
    threshold = threshold,
    "GPU temperature threshold exceeded"
);
```

---

## 8. Testing

### 8.1 Unit Tests

```rust
#[test]
fn test_nvml_initialization() {
    // Should succeed on systems with NVIDIA GPU
    let result = Nvml::init();
    
    if cfg!(target_os = "linux") && has_nvidia_gpu() {
        assert!(result.is_ok());
    }
}

#[test]
fn test_process_detection() {
    let monitor = GpuProcessMonitor::new(0).unwrap();
    
    // Should not detect self
    let should_evict = monitor.should_evict().unwrap();
    // May be true or false depending on other processes
}
```

### 8.2 Integration Tests

```rust
#[tokio::test]
async fn test_auto_eviction() {
    // Start worker with model loaded
    let worker = start_worker().await;
    assert!(worker.has_models_loaded());
    
    // Simulate other GPU process
    let _other_proc = start_gpu_process();
    
    // Wait for detection
    tokio::time::sleep(Duration::from_secs(6)).await;
    
    // Worker should have evicted models
    assert!(!worker.has_models_loaded());
    assert!(worker.is_paused());
}
```

---

## 9. Compliance & Priority

### 9.1 Deployment Requirements

| Deployment Type | NVML Required? | Priority |
|----------------|----------------|----------|
| **Home/Desktop** | ✅ **MANDATORY** | **CRITICAL** |
| **Dedicated Server** | ⚠️ Recommended | HIGH |
| **Cloud/Container** | ⚠️ Recommended | MEDIUM |

### 9.2 Feature Priority

| Feature | Priority | Rationale |
|---------|----------|-----------|
| Process detection | **CRITICAL** | Without it, home users can't use GPU |
| Health monitoring | HIGH | Prevents hardware damage |
| Capacity planning | MEDIUM | Nice to have |
| Error recovery | MEDIUM | Improves reliability |

---

## 10. Migration Path

### Phase 1: Core Integration (v0.2.0)
- [ ] Add `nvml-wrapper` dependency
- [ ] Implement `GpuProcessMonitor`
- [ ] Add process detection loop
- [ ] Add auto-eviction logic
- [ ] Add configuration options

### Phase 2: Health Monitoring (v0.3.0)
- [ ] Implement `GpuHealthMonitor`
- [ ] Add temperature monitoring
- [ ] Add thermal throttling
- [ ] Add health metrics

### Phase 3: Advanced Features (v0.4.0)
- [ ] Add capacity planning
- [ ] Add error recovery
- [ ] Add multi-GPU support
- [ ] Add advanced metrics

---

## 11. References

- **NVML Documentation**: https://docs.nvidia.com/deploy/nvml-api/
- **nvml-wrapper Crate**: https://crates.io/crates/nvml-wrapper
- **Related Spec**: `vram-residency/.specs/45_shared_gpu_contention.md`

---

## 12. Summary

**NVML is CRITICAL for worker-orcd** to provide acceptable UX on home/desktop systems where the GPU is shared. Without it, users cannot use their GPU for other applications while llama-orch is running.

**Status**: ✅ **REQUIRED** - Must be implemented before v1.0 release for home/desktop support.

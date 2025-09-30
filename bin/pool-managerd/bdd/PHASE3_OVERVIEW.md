# Phase 3: Device Masks, Observability & Advanced Features

**Status**: 📋 PLANNED  
**Estimated Effort**: 4-5 sessions  
**Prerequisites**: Phase 1 ✅ & Phase 2 ✅ Complete

---

## Overview

Phase 3 covers the remaining **50 behaviors** from the audit that require implementation:

- Device discovery and GPU inventory
- Device mask enforcement and placement
- Heterogeneous GPU split planning
- Metrics emission (Prometheus)
- Structured logging
- Configuration management
- Security and isolation

---

## Feature Breakdown

### 1. Device Discovery & GPU Inventory (13 behaviors)

**Spec**: CHECKLIST "Device Discovery & Snapshots"  
**Status**: ⏳ Not Yet Implemented

#### Behaviors

- **B-DEVICE-001**: DeviceMask parses comma-separated GPU IDs ("0,1")
- **B-DEVICE-002**: DeviceMask parses named GPUs ("GPU0,GPU1")
- **B-DEVICE-003**: DeviceMask validates against discovered devices
- **B-DEVICE-004**: DeviceMask rejects invalid device IDs
- **B-DEVICE-005**: DeviceMask converts to CUDA_VISIBLE_DEVICES format
- **B-DEVICE-010**: Engine only uses GPUs in device mask
- **B-DEVICE-011**: No cross-mask spillover occurs
- **B-DEVICE-012**: CUDA_VISIBLE_DEVICES is set correctly
- **B-DEVICE-013**: Device mask is persisted in registry

#### Implementation Files

```
src/placement/devicemasks.rs  - DeviceMask struct and parsing
src/placement/discovery.rs     - GPU enumeration (NEW)
```

#### Dependencies

```toml
nvml-wrapper = "0.9"  # NVIDIA Management Library
```

#### Key APIs

```rust
pub struct DeviceMask {
    device_ids: Vec<u32>,
}

impl DeviceMask {
    pub fn parse(mask_str: &str) -> Result<Self>;
    pub fn validate_against_discovered(devices: &[DeviceSnapshot]) -> Result<()>;
    pub fn to_cuda_visible_devices(&self) -> String;
}

pub struct DeviceSnapshot {
    pub id: u32,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
}

pub fn discover_gpus() -> Result<Vec<DeviceSnapshot>>;
```

---

### 2. Heterogeneous GPU Split Planning (6 behaviors)

**Spec**: OC-POOL-3021, OC-POOL-3052  
**Status**: ⏳ Not Yet Implemented

#### Behaviors

- **B-HETERO-001**: SplitPlan computes ratios for multi-GPU setup
- **B-HETERO-002**: Split ratios sum to 1.0
- **B-HETERO-003**: Smallest GPU capacity is respected
- **B-HETERO-004**: Split ratios are capped for smallest GPU
- **B-HETERO-005**: SplitPlan validates model fits in smallest GPU share
- **B-HETERO-006**: Default is no split (single GPU)

#### Implementation Files

```
src/placement/hetero_split.rs  - SplitPlan computation
```

#### Key APIs

```rust
pub struct SplitPlan {
    ratios: Vec<f32>,
    device_ids: Vec<u32>,
}

impl SplitPlan {
    pub fn compute(devices: &[DeviceSnapshot], model_size_bytes: u64) -> Result<Self>;
    pub fn ratios(&self) -> &[f32];
    pub fn validate(&self) -> Result<()>;
    pub fn to_tensor_split_flags(&self) -> Vec<String>;
}
```

---

### 3. Observability - Metrics (10 behaviors)

**Spec**: OC-POOL-3030  
**Status**: ⏳ Not Yet Implemented

#### Behaviors

- **B-METRICS-001**: Emit preload_success_total counter
- **B-METRICS-002**: Emit preload_failure_total counter with reason label
- **B-METRICS-003**: Emit vram_total_bytes gauge per device
- **B-METRICS-004**: Emit vram_free_bytes gauge per device
- **B-METRICS-005**: Emit driver_reset_total counter
- **B-METRICS-006**: Emit restart_total counter per pool
- **B-METRICS-007**: Emit backoff_delay_ms histogram
- **B-METRICS-008**: Emit slots_total gauge per pool
- **B-METRICS-009**: Emit slots_free gauge per pool
- **B-METRICS-010**: Emit active_leases gauge per pool

#### Implementation Files

```
src/observability/metrics.rs  - Prometheus metrics (NEW)
```

#### Dependencies

```toml
prometheus = "0.13"
```

#### Key Metrics

```rust
// Counters
pool_preload_success_total{pool_id, engine}
pool_preload_failure_total{pool_id, engine, reason}
pool_restart_total{pool_id, reason}
pool_drain_total{pool_id, outcome}
pool_reload_total{pool_id, outcome}
engine_crash_total{pool_id, reason}
circuit_breaker_open_total{pool_id}
restart_storm_total{pool_id}
driver_reset_total{pool_id}

// Gauges
pool_slots_total{pool_id}
pool_slots_free{pool_id}
pool_active_leases{pool_id}
device_vram_total_bytes{device_id}
device_vram_free_bytes{device_id}
circuit_breaker_state{pool_id}  // 0=closed, 1=open, 2=half-open

// Histograms
pool_drain_duration_ms{pool_id}
pool_reload_duration_ms{pool_id}
backoff_delay_ms{pool_id}
restart_rate{pool_id}
```

---

### 4. Observability - Structured Logging (8 behaviors)

**Spec**: OC-POOL-3030, ORCH-3027  
**Status**: ⏳ Partial (tracing present, fields need standardization)

#### Behaviors

- **B-LOG-001**: Logs include pool_id field
- **B-LOG-002**: Logs include engine field
- **B-LOG-003**: Logs include engine_version field
- **B-LOG-004**: Logs include device_mask field
- **B-LOG-005**: Logs include restart_count on restart
- **B-LOG-006**: Logs include backoff_ms on backoff
- **B-LOG-007**: Logs include last_error on failure
- **B-LOG-008**: Logs are structured JSON format

#### Implementation

```rust
// Standardize log fields across all modules
tracing::info!(
    pool_id = %pool_id,
    engine = "llamacpp",
    engine_version = %version,
    device_mask = %mask,
    restart_count = count,
    backoff_ms = delay,
    last_error = %error,
    "event description"
);
```

#### Required Changes

- Add JSON formatter to tracing-subscriber
- Standardize field names across all log calls
- Add correlation_id support (X-Correlation-Id)

---

### 5. Configuration Management (6 behaviors)

**Spec**: CHECKLIST "Configuration & Feature Toggles"  
**Status**: ⏳ Partial (POOL_MANAGERD_ADDR exists)

#### Behaviors

- **B-CONFIG-001**: POOL_MANAGERD_ADDR sets bind address
- **B-CONFIG-002**: POOL_MANAGERD_ADDR defaults to 127.0.0.1:9200
- **B-CONFIG-003**: Invalid bind address returns error
- **B-CONFIG-010**: Device masks are configurable per pool
- **B-CONFIG-011**: Device masks are validated against discovered devices
- **B-CONFIG-012**: Invalid device mask returns error

#### Implementation Files

```
src/config.rs  - Configuration struct (NEW)
```

#### Configuration Structure

```rust
pub struct PoolManagerConfig {
    pub bind_addr: String,
    pub health_check_interval_ms: u64,
    pub health_check_timeout_ms: u64,
    pub backoff_initial_ms: u64,
    pub backoff_max_ms: u64,
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_timeout_secs: u64,
    pub restart_rate_limit: u32,
    pub restart_window_secs: u64,
}

impl PoolManagerConfig {
    pub fn from_env() -> Result<Self>;
    pub fn validate(&self) -> Result<()>;
}
```

---

### 6. Security & Isolation (6 behaviors)

**Spec**: CHECKLIST "Security & Policy"  
**Status**: ⏳ Not Yet Implemented

#### Behaviors

- **B-SEC-001**: Engine processes run as non-root user
- **B-SEC-002**: Engine processes have isolated workdir
- **B-SEC-003**: Engine processes have minimal capabilities
- **B-SEC-010**: Containers run rootless (podman preferred)
- **B-SEC-011**: Containers have seccomp profile
- **B-SEC-012**: Containers have constrained network access

#### Implementation

```rust
// In preload.rs spawn logic
pub struct ProcessIsolation {
    pub user: Option<String>,
    pub group: Option<String>,
    pub workdir: PathBuf,
    pub drop_capabilities: Vec<String>,
}

impl ProcessIsolation {
    pub fn apply_to_command(&self, cmd: &mut Command) -> Result<()>;
}
```

---

## Implementation Priority

### High Priority (P0)

1. **Device Discovery** - Required for placement
2. **Metrics Emission** - Required for observability
3. **Structured Logging** - Required for debugging

### Medium Priority (P1)

4. **Device Mask Enforcement** - Required for multi-GPU
5. **Configuration Management** - Required for flexibility

### Low Priority (P2)

6. **Heterogeneous Split** - Optional optimization
7. **Security Isolation** - Nice-to-have hardening

---

## Estimated Timeline

### Week 1: Device Discovery & Masks

- Day 1-2: GPU discovery via nvml-wrapper
- Day 3-4: DeviceMask parsing and validation
- Day 5: CUDA_VISIBLE_DEVICES enforcement

### Week 2: Observability

- Day 1-2: Prometheus metrics setup
- Day 3: Emit all counters/gauges/histograms
- Day 4-5: Structured logging standardization

### Week 3: Configuration & Split Planning

- Day 1-2: Configuration management
- Day 3-4: Heterogeneous split planning
- Day 5: Validation and testing

### Week 4: Security & Integration

- Day 1-2: Process isolation
- Day 3-4: Integration testing
- Day 5: E2E testing and documentation

---

## Dependencies to Add

```toml
[dependencies]
# Already added in Phase 2
nix = { version = "0.27", features = ["signal", "process"] }
rand = "0.8"

# Phase 3 additions
nvml-wrapper = "0.9"           # GPU discovery
prometheus = "0.13"             # Metrics
tracing-subscriber = { version = "0.3", features = ["json"] }  # JSON logs
```

---

## Feature Files Status

Phase 3 feature files **do not exist yet**. They need to be created:

### To Create

1. `tests/features/placement/device_masks.feature` (9 scenarios)
2. `tests/features/placement/hetero_split.feature` (6 scenarios)
3. `tests/features/observability/metrics.feature` (10 scenarios)
4. `tests/features/observability/logging.feature` (8 scenarios)
5. `tests/features/config/configuration.feature` (6 scenarios)
6. `tests/features/security/isolation.feature` (6 scenarios)

**Total**: 45 scenarios for Phase 3

---

## Success Criteria

### Device Discovery ✓

- [ ] Enumerate all NVIDIA GPUs
- [ ] Report VRAM total/free per device
- [ ] Report compute capability
- [ ] Handle MIG partitions (optional)

### Device Masks ✓

- [ ] Parse "0,1,2" format
- [ ] Validate against discovered devices
- [ ] Set CUDA_VISIBLE_DEVICES correctly
- [ ] Prevent cross-mask spillover

### Metrics ✓

- [ ] All 10 required metrics emitting
- [ ] Prometheus scrape endpoint working
- [ ] Labels include pool_id, engine, reason
- [ ] Histograms have appropriate buckets

### Logging ✓

- [ ] JSON format output
- [ ] All required fields present
- [ ] Correlation ID support
- [ ] No secrets leaked

### Configuration ✓

- [ ] Environment variable loading
- [ ] Validation on startup
- [ ] Sensible defaults
- [ ] Error messages clear

### Security ✓

- [ ] Non-root execution
- [ ] Isolated workdir
- [ ] Capability dropping
- [ ] Rootless containers (if used)

---

## Integration with Orchestratord

Phase 3 features integrate with orchestratord:

1. **Device Discovery** → Placement decisions use VRAM data
2. **Metrics** → Orchestratord scrapes pool-managerd metrics
3. **Device Masks** → Orchestratord respects mask constraints
4. **Logging** → Correlation IDs flow through request chain

---

## Conclusion

Phase 3 completes the pool-managerd implementation with:

- ✅ GPU-aware placement
- ✅ Production observability
- ✅ Flexible configuration
- ✅ Security hardening

**Total Effort**: ~4 weeks for complete Phase 3 implementation

**Current Status**: Phase 1 ✅ & Phase 2 ✅ complete, Phase 3 ready to start

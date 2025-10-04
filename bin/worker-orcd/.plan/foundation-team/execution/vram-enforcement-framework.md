# VRAM-Only Enforcement Framework

**Team**: Foundation-Alpha  
**Purpose**: Validation framework for VRAM-only policy enforcement

---

## Overview

The VRAM-only policy is a **critical architectural constraint** for worker-orcd. This framework ensures no RAM fallback or UMA (Unified Memory Architecture) violations occur during inference.

---

## Policy Definition

### VRAM-Only Requirements

**MUST**:
- All model weights in VRAM (device memory)
- All KV cache in VRAM
- All intermediate tensors in VRAM
- All CUDA allocations use `cudaMalloc` (not `cudaMallocManaged`)

**MUST NOT**:
- Use `cudaMallocManaged` (UMA)
- Allow RAM fallback
- Use host-pinned memory for weights
- Use zero-copy memory

---

## Validation Framework

### Component 1: VramTracker

**Purpose**: Track all VRAM allocations and verify residency

**Implementation**: FT-011 (VRAM-Only Enforcement)

**Key Functions**:
```cpp
class VramTracker {
    void record_allocation(void* ptr, size_t bytes, VramPurpose purpose);
    void record_deallocation(void* ptr);
    bool verify_vram_residency() const;
    uint64_t total_usage() const;
};
```

**Validation**:
- Every allocation tracked
- Every deallocation tracked
- Periodic residency verification

---

### Component 2: Residency Verification

**Purpose**: Verify pointers are VRAM-resident (not host/managed)

**Implementation**: FT-014 (VRAM Residency Verification)

**Key Functions**:
```cpp
bool check_pointer_residency(const void* ptr) {
    cudaPointerAttributes attrs;
    cudaPointerGetAttributes(&attrs, ptr);
    
    // Must be device memory
    if (attrs.type != cudaMemoryTypeDevice) return false;
    
    // Must not have host pointer (no UMA)
    if (attrs.hostPointer != nullptr) return false;
    
    return true;
}
```

**Validation**:
- Check pointer type is `cudaMemoryTypeDevice`
- Check no host pointer exists
- Periodic checks every 60 seconds

---

### Component 3: Health Monitoring

**Purpose**: Continuous monitoring of VRAM residency

**Implementation**: FT-014 (VRAM Residency Verification)

**Key Functions**:
```rust
pub struct HealthMonitor {
    pub async fn run(&mut self, model: &CudaModel);
    pub fn is_healthy(&self) -> bool;
}
```

**Validation**:
- Background task runs every 60 seconds
- Worker marks itself unhealthy on violation
- Health endpoint exposes residency status

---

## Testing Framework

### Unit Tests

**Test 1: Allocation Tracking**
```rust
#[test]
fn test_vram_tracker_records_allocations() {
    let tracker = VramTracker::new();
    let ptr = allocate_vram(1024);
    
    tracker.record_allocation(ptr, 1024, VramPurpose::ModelWeights);
    assert_eq!(tracker.total_usage(), 1024);
}
```

**Test 2: Residency Verification**
```rust
#[test]
fn test_device_pointer_is_resident() {
    let ptr = cudaMalloc(1024);
    assert!(check_pointer_residency(ptr));
}

#[test]
fn test_managed_pointer_is_not_resident() {
    let ptr = cudaMallocManaged(1024);
    assert!(!check_pointer_residency(ptr)); // Should fail
}
```

**Test 3: Health Monitoring**
```rust
#[test]
fn test_health_monitor_detects_violation() {
    let mut monitor = HealthMonitor::new(60);
    let model = create_model_with_uma_violation();
    
    monitor.run(&model).await;
    assert!(!monitor.is_healthy());
}
```

---

### Integration Tests

**Test 1: Model Loading**
```rust
#[tokio::test]
async fn test_model_loads_vram_only() {
    let model = load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf").await;
    
    // Verify all weights in VRAM
    assert!(model.verify_vram_residency());
    
    // Verify no UMA
    for ptr in model.weight_pointers() {
        assert!(check_pointer_residency(ptr));
    }
}
```

**Test 2: Inference Execution**
```rust
#[tokio::test]
async fn test_inference_maintains_vram_only() {
    let model = load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf").await;
    
    let result = model.infer("Test prompt", 10).await;
    
    // Verify VRAM residency after inference
    assert!(model.verify_vram_residency());
}
```

**Test 3: KV Cache**
```rust
#[tokio::test]
async fn test_kv_cache_vram_only() {
    let model = load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf").await;
    let kv_cache = allocate_kv_cache(model.config());
    
    // Verify KV cache in VRAM
    assert!(check_pointer_residency(kv_cache.keys(0)));
    assert!(check_pointer_residency(kv_cache.values(0)));
}
```

---

## Validation Procedures

### Pre-Deployment Validation

**Step 1: Run Unit Tests**
```bash
cargo test --package worker-orcd vram_enforcement
```

**Step 2: Run Integration Tests**
```bash
cargo test --package worker-orcd --test vram_residency_integration
```

**Step 3: Manual Verification**
```bash
# Start worker
cargo run --features cuda

# Check health endpoint
curl http://localhost:8080/health | jq '.resident'

# Should return: true
```

---

### Runtime Monitoring

**Step 1: Enable Health Monitoring**
```rust
let mut health_monitor = HealthMonitor::new(60); // 60 second interval
tokio::spawn(async move {
    health_monitor.run(&model).await;
});
```

**Step 2: Check Health Endpoint**
```bash
curl http://localhost:8080/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "resident": true,
  "vram_bytes_used": 524288000,
  "last_check_secs": 30
}
```

**Step 3: Monitor Logs**
```bash
# Look for residency violations
grep "VRAM residency" logs/worker.log
```

---

## Violation Handling

### Detection

**Automatic Detection**:
- Health monitor runs every 60 seconds
- Checks all tracked allocations
- Logs critical error on violation

**Manual Detection**:
- Health endpoint shows `resident: false`
- Logs show "VRAM residency violated"

### Response

**Immediate Actions**:
1. Worker marks itself unhealthy
2. Critical log emitted
3. Health endpoint returns unhealthy status
4. Orchestrator stops routing requests

**Investigation**:
1. Check logs for violation details
2. Identify which pointer violated policy
3. Review allocation code path
4. Fix root cause

---

## Acceptance Criteria

### For FT-011 (VRAM-Only Enforcement)
- [ ] VramTracker implemented
- [ ] All allocations tracked
- [ ] All deallocations tracked
- [ ] Unit tests passing

### For FT-014 (VRAM Residency Verification)
- [ ] Residency check function implemented
- [ ] Health monitor implemented
- [ ] Periodic checks working
- [ ] Health endpoint integration
- [ ] Unit tests passing
- [ ] Integration tests passing

### For Production Deployment
- [ ] All VRAM enforcement tests passing
- [ ] No UMA allocations detected
- [ ] Health monitoring operational
- [ ] Violation handling tested

---

**Last Updated**: 2025-10-04  
**Updated By**: Foundation-Alpha

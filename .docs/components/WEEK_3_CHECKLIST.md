# Week 3 Checklist: Observability & Health

**Week:** 3 of 4  
**Goal:** Production monitoring, health checks, metrics  
**Duration:** 5-6 days  
**Target:** ~160-180/300 tests passing (53-60%)

**Reference:** See `ORCHESTRATOR_STANDARDS.md` for what rbee already does

---


## 📋 Priority 1: Heartbeat Mechanism (1-2 days)

### Task 1.1: Design Heartbeat System
- [ ] Define heartbeat interval config (default: 30s)
- [ ] Define stale worker threshold (default: 2x heartbeat interval = 60s)
- [ ] Design heartbeat payload structure (worker_id, timestamp, health_status)
- [ ] Document heartbeat flow in architecture docs

### Task 1.2: Worker-Side Implementation
- [ ] Add heartbeat config to llm-worker-rbee
- [ ] Implement periodic heartbeat sender (tokio::interval)
- [ ] Send POST /v1/heartbeat to rbee-hive
- [ ] Include worker health metrics in heartbeat
- [ ] Handle heartbeat failures gracefully (log, don't crash)

**Files to modify:**
- `bin/llm-worker-rbee/src/config.rs` - Add heartbeat_interval_secs
- `bin/llm-worker-rbee/src/main.rs` - Spawn heartbeat task
- `bin/llm-worker-rbee/src/heartbeat.rs` - NEW FILE - Heartbeat logic

### Task 1.3: rbee-hive Heartbeat Endpoint
- [ ] Add POST /v1/heartbeat endpoint to rbee-hive
- [ ] Update WorkerInfo.last_heartbeat timestamp
- [ ] Validate heartbeat payload
- [ ] Return 200 OK on success

**Files to modify:**
- `bin/rbee-hive/src/http/routes.rs` - Add heartbeat route
- `bin/rbee-hive/src/http/heartbeat.rs` - NEW FILE - Heartbeat handler
- `bin/rbee-hive/src/registry.rs` - Add update_heartbeat() method

### Task 1.4: Stale Worker Detection
- [ ] Add background task to check for stale workers
- [ ] Run every 30 seconds (half of heartbeat interval)
- [ ] Mark workers as stale if last_heartbeat > threshold
- [ ] Log stale worker detection events
- [ ] Optionally auto-restart stale workers (if restart policy enabled)

**Files to modify:**
- `bin/rbee-hive/src/monitor.rs` - Add stale worker checker
- `bin/rbee-hive/src/main.rs` - Spawn monitor task

### Task 1.5: Testing
- [ ] Unit test heartbeat sender
- [ ] Unit test heartbeat endpoint
- [ ] Unit test stale worker detection
- [ ] Integration test: worker sends heartbeat, rbee-hive receives
- [ ] Integration test: worker stops, marked as stale after threshold

**Impact:** ✅ Detect hung workers, automatic recovery

---

## 📋 Priority 2: Resource Limits (2-3 days)

### Task 2.1: Memory Limits
- [ ] Add memory limit config to worker (default: 8GB)
- [ ] Check available system memory before worker spawn
- [ ] Reject worker spawn if insufficient memory
- [ ] Add cgroups memory limits (if available on Linux)
- [ ] Monitor worker memory usage
- [ ] Kill worker if exceeds memory limit

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` - Add memory check before spawn
- `bin/rbee-hive/src/resources.rs` - NEW FILE - Resource monitoring
- `bin/rbee-hive/src/config.rs` - Add memory_limit_mb config

### Task 2.2: VRAM Monitoring
- [ ] Detect GPU backend (CUDA/Metal)
- [ ] Query available VRAM before worker spawn
- [ ] Estimate model VRAM requirements (from model metadata)
- [ ] Reject worker spawn if insufficient VRAM
- [ ] Monitor worker VRAM usage (if possible)
- [ ] Add VRAM metrics

**Files to modify:**
- `bin/shared-crates/gpu-info/src/backend.rs` - Add VRAM query functions
- `bin/rbee-hive/src/http/workers.rs` - Add VRAM check before spawn
- `bin/rbee-hive/src/resources.rs` - Add VRAM monitoring

### Task 2.3: Disk Space Monitoring
- [ ] Check available disk space before model download
- [ ] Estimate model size (from HuggingFace metadata)
- [ ] Reject download if insufficient disk space
- [ ] Add disk space metrics
- [ ] Add cleanup for old models (if retention policy)

**Files to modify:**
- `bin/rbee-hive/src/provisioner/download.rs` - Add disk space check
- `bin/rbee-hive/src/resources.rs` - Add disk space monitoring

### Task 2.4: Resource Metrics
- [ ] Add Prometheus metric: rbee_hive_memory_available_bytes
- [ ] Add Prometheus metric: rbee_hive_memory_used_bytes
- [ ] Add Prometheus metric: rbee_hive_vram_available_bytes
- [ ] Add Prometheus metric: rbee_hive_vram_used_bytes
- [ ] Add Prometheus metric: rbee_hive_disk_available_bytes
- [ ] Add Prometheus metric: rbee_hive_disk_used_bytes

**Files to modify:**
- `bin/rbee-hive/src/metrics.rs` - Add resource metrics

### Task 2.5: Testing
- [ ] Unit test memory limit checks
- [ ] Unit test VRAM limit checks
- [ ] Unit test disk space checks
- [ ] Integration test: reject worker spawn when out of memory
- [ ] Integration test: reject download when out of disk space

**Impact:** ✅ Prevent OOM, better resource management

---

## 📋 Priority 3: Metrics & Observability (2-3 days)

### Task 3.1: Worker State Metrics (Prometheus Naming Conventions)
- [ ] Verify existing metrics work: rbee_hive_workers_total{state}
- [ ] Add metric: rbee_hive_workers_total{backend} (use labels, not separate metrics)
- [ ] Add metric: rbee_hive_workers_total{device} (use labels, not separate metrics)
- [ ] Add metric: rbee_hive_workers_stale_total (counter with _total suffix)
- [ ] Update metrics on worker state changes

**Prometheus Best Practices:**
- Use `_total` suffix for counters
- Use labels for dimensions (backend, device, state)
- Keep cardinality low (avoid high-cardinality labels like worker_id)
- Use consistent naming: `<namespace>_<subsystem>_<name>_<unit>_<suffix>`

**Files to modify:**
- `bin/rbee-hive/src/metrics.rs` - Add new metrics

### Task 3.2: Inference Latency Metrics (Prometheus Naming Conventions)
- [ ] Add metric: rbee_hive_inference_duration_seconds{quantile} (histogram with _seconds suffix)
- [ ] Add metric: rbee_hive_inference_tokens_per_second{quantile} (histogram)
- [ ] Add metric: rbee_hive_inference_requests_total{status} (counter with _total suffix)
- [ ] Add metric: rbee_hive_inference_errors_total{type} (counter with _total suffix)
- [ ] Record metrics in inference flow

**Prometheus Best Practices:**
- Use `_seconds` suffix for durations (not milliseconds!)
- Use `_total` suffix for counters
- Use histograms for latency (not summaries)
- Include quantile labels (0.5, 0.95, 0.99)
- Use base units (seconds, bytes, not ms, MB)

**Files to modify:**
- `bin/rbee-hive/src/metrics.rs` - Add inference metrics
- `bin/rbee-hive/src/http/workers.rs` - Record metrics

### Task 3.3: Error Rate Metrics
- [ ] Add metric: rbee_hive_errors_total{type}
- [ ] Add metric: rbee_hive_worker_spawn_failures_total
- [ ] Add metric: rbee_hive_worker_health_check_failures_total
- [ ] Add metric: rbee_hive_model_download_failures_total
- [ ] Record errors throughout codebase

**Files to modify:**
- `bin/rbee-hive/src/metrics.rs` - Add error metrics
- Various error handling locations

### Task 3.4: Grafana Dashboard
- [ ] Create dashboard JSON file
- [ ] Add panel: Worker count by state (gauge)
- [ ] Add panel: Inference latency (graph)
- [ ] Add panel: Tokens per second (graph)
- [ ] Add panel: Error rate (graph)
- [ ] Add panel: Memory usage (graph)
- [ ] Add panel: VRAM usage (graph)
- [ ] Add panel: Disk usage (graph)
- [ ] Document dashboard import process

**Files to create:**
- `ci/dashboards/rbee-hive-overview.json` - NEW FILE

### Task 3.5: Testing
- [ ] Verify metrics endpoint returns valid Prometheus format
- [ ] Verify all metrics update correctly
- [ ] Load test: verify metrics under load
- [ ] Import Grafana dashboard and verify panels

**Impact:** ✅ Production monitoring, performance insights

---

## 📊 Week 3 Deliverables

- [ ] Heartbeat mechanism active (workers send, rbee-hive receives)
- [ ] Stale worker detection working
- [ ] Memory limits enforced
- [ ] VRAM limits enforced
- [ ] Disk space limits enforced
- [ ] Resource metrics exposed
- [ ] Inference metrics exposed
- [ ] Error metrics exposed
- [ ] Grafana dashboard created and tested
- [ ] ~160-180/300 tests passing (53-60%)

---

## 🎯 Success Criteria

### Functional
- [ ] Workers send heartbeat every 30 seconds
- [ ] Stale workers detected within 60 seconds
- [ ] Worker spawn rejected when out of memory
- [ ] Worker spawn rejected when out of VRAM
- [ ] Model download rejected when out of disk space
- [ ] All metrics update in real-time

### Performance
- [ ] Heartbeat overhead < 1% CPU
- [ ] Resource monitoring overhead < 2% CPU
- [ ] Metrics endpoint responds < 100ms

### Quality
- [ ] All new code has unit tests
- [ ] Integration tests pass
- [ ] No new unwrap/expect in production code
- [ ] Proper error handling throughout

---

## 📝 Notes for Next Team

### Quick Wins
- Heartbeat mechanism is straightforward (30-40 lines per component)
- Metrics library already exists and works well
- Resource monitoring is mostly reading /proc files

### Challenges
- VRAM monitoring varies by backend (CUDA vs Metal)
- cgroups may not be available on all systems
- Grafana dashboard requires manual import

### Testing Strategy
- Use mock workers for heartbeat tests
- Use resource limits in test environment
- Load test with multiple workers

---

**Created by:** TEAM-113  
**Date:** 2025-10-18  
**For:** Week 3 implementation team

# pool-managerd Behavior Audit for BDD

**Purpose**: Comprehensive list of ALL behaviors that need BDD test coverage.  
**Source**: Audited from source code, spec (`.specs/30-pool-managerd.md`), and checklist (`CHECKLIST.md`).  
**Date**: 2025-09-30

---

## 1. HTTP API Behaviors

### 1.1 Health Endpoint (`GET /health`)

- **B-API-001**: Daemon health check returns 200 OK with status and version
- **B-API-002**: Health endpoint is always available (even when pools are not ready)
- **B-API-003**: Health response includes daemon version from CARGO_PKG_VERSION

### 1.2 Preload Endpoint (`POST /pools/{id}/preload`)

- **B-API-010**: Accepts PreparedEngine JSON and spawns engine process
- **B-API-011**: Returns 200 with pool_id, pid, and handoff_path on success
- **B-API-012**: Returns 500 with error message when preload fails
- **B-API-013**: Returns 500 with error message when registry lock fails
- **B-API-014**: Spawned process writes to log file in .runtime/
- **B-API-015**: PID file is created in .runtime/{pool_id}.pid
- **B-API-016**: Handoff JSON is written to .runtime/engines/engine.json

### 1.3 Pool Status Endpoint (`GET /pools/{id}/status`)

- **B-API-020**: Returns 200 with pool status when pool exists
- **B-API-021**: Returns 404 when pool does not exist
- **B-API-022**: Returns 500 when registry lock fails
- **B-API-023**: Status includes: pool_id, live, ready, active_leases, engine_version
- **B-API-024**: Status reflects current registry state accurately

---

## 2. Preload & Readiness Lifecycle Behaviors

### 2.1 Engine Spawn (OC-POOL-3001)

- **B-PRELOAD-001**: Engine process is spawned with correct binary path
- **B-PRELOAD-002**: Engine process receives all flags from PreparedEngine
- **B-PRELOAD-003**: Engine stdout/stderr redirected to log file
- **B-PRELOAD-004**: PID file is written before health check
- **B-PRELOAD-005**: Process spawns successfully with valid PreparedEngine

### 2.2 Health Check Wait

- **B-PRELOAD-010**: Health check polls engine HTTP endpoint
- **B-PRELOAD-011**: Health check retries with 500ms intervals
- **B-PRELOAD-012**: Health check succeeds when engine returns HTTP 200
- **B-PRELOAD-013**: Health check times out after 120 seconds
- **B-PRELOAD-014**: Health check accepts HTTP/1.1 200 or HTTP/1.0 200

### 2.3 Readiness Gating (OC-POOL-3001, OC-POOL-3003)

- **B-PRELOAD-020**: Pool is NOT ready until health check succeeds
- **B-PRELOAD-021**: Pool becomes ready only after successful health check
- **B-PRELOAD-022**: Handoff file is written only after health check success
- **B-PRELOAD-023**: Registry is updated to ready state after health check
- **B-PRELOAD-024**: Registry includes engine_version from PreparedEngine

### 2.4 Preload Failure Handling (OC-POOL-3002)

- **B-PRELOAD-030**: Process is killed if health check fails
- **B-PRELOAD-031**: PID file is removed if health check fails
- **B-PRELOAD-032**: Registry records last_error when health check fails
- **B-PRELOAD-033**: Preload returns error when health check times out
- **B-PRELOAD-034**: Pool remains unready when preload fails

### 2.5 Handoff File Generation

- **B-PRELOAD-040**: Handoff JSON includes engine, engine_version, url
- **B-PRELOAD-041**: Handoff JSON includes pool_id, replica_id
- **B-PRELOAD-042**: Handoff JSON includes model path
- **B-PRELOAD-043**: Handoff JSON includes flags array
- **B-PRELOAD-044**: Handoff file is pretty-printed JSON

---

## 3. Registry State Management Behaviors

### 3.1 Health Status

- **B-REG-001**: set_health updates pool health (live, ready)
- **B-REG-002**: get_health returns current health status
- **B-REG-003**: get_health returns None for non-existent pool
- **B-REG-004**: Health status defaults to live=false, ready=false

### 3.2 Error Tracking

- **B-REG-010**: set_last_error records error message
- **B-REG-011**: get_last_error returns recorded error
- **B-REG-012**: get_last_error returns None when no error
- **B-REG-013**: Last error persists until cleared or overwritten

### 3.3 Heartbeat Tracking

- **B-REG-020**: set_heartbeat records timestamp in milliseconds
- **B-REG-021**: get_heartbeat returns last heartbeat timestamp
- **B-REG-022**: get_heartbeat returns None when never set

### 3.4 Version Management

- **B-REG-030**: set_version records registry version
- **B-REG-031**: get_version returns registry version
- **B-REG-032**: set_engine_version records engine version (distinct from registry version)
- **B-REG-033**: get_engine_version returns engine version

### 3.5 Engine Metadata

- **B-REG-040**: set_engine_meta updates engine_version when provided
- **B-REG-041**: set_engine_meta updates engine_digest when provided
- **B-REG-042**: set_engine_meta updates engine_catalog_id when provided
- **B-REG-043**: set_engine_meta only updates provided fields (partial update)
- **B-REG-044**: get_engine_digest returns engine digest
- **B-REG-045**: get_engine_catalog_id returns catalog ID

### 3.6 Device Mask Management

- **B-REG-050**: set_device_mask records device mask string
- **B-REG-051**: get_device_mask returns device mask
- **B-REG-052**: Device mask can be None (no restriction)

### 3.7 Slot Management

- **B-REG-060**: set_slots records total and free slots
- **B-REG-061**: get_slots_total returns total slots
- **B-REG-062**: get_slots_free returns free slots
- **B-REG-063**: Free slots are clamped to [0, total]
- **B-REG-064**: Negative free slots are clamped to 0

### 3.8 Draining State

- **B-REG-070**: set_draining marks pool as draining
- **B-REG-071**: get_draining returns draining state
- **B-REG-072**: get_draining returns false for non-existent pool
- **B-REG-073**: Draining state defaults to false

### 3.9 Pool Registration

- **B-REG-080**: register creates new pool entry
- **B-REG-081**: register returns true for new pool
- **B-REG-082**: register returns false for existing pool
- **B-REG-083**: deregister removes pool entry
- **B-REG-084**: deregister returns true when pool existed
- **B-REG-085**: deregister returns false when pool didn't exist

### 3.10 Lease Accounting

- **B-REG-090**: allocate_lease increments active_leases
- **B-REG-091**: allocate_lease returns new lease count
- **B-REG-092**: release_lease decrements active_leases
- **B-REG-093**: release_lease returns new lease count
- **B-REG-094**: release_lease never goes below 0
- **B-REG-095**: get_active_leases returns current count
- **B-REG-096**: get_active_leases returns 0 for non-existent pool

### 3.11 Handoff Registration (OC-POOL-3101, OC-POOL-3102)

- **B-REG-100**: register_ready_from_handoff sets health to live=true, ready=true
- **B-REG-101**: register_ready_from_handoff extracts engine_version from handoff
- **B-REG-102**: register_ready_from_handoff extracts device_mask from handoff
- **B-REG-103**: register_ready_from_handoff extracts slots_total from handoff
- **B-REG-104**: register_ready_from_handoff extracts slots_free from handoff
- **B-REG-105**: register_ready_from_handoff clears last_error
- **B-REG-106**: register_ready_from_handoff sets heartbeat to current time
- **B-REG-107**: register_ready_from_handoff handles missing optional fields gracefully

### 3.12 Update Merge

- **B-REG-110**: update merges provided fields into existing entry
- **B-REG-111**: update preserves fields not in UpdateFields
- **B-REG-112**: update creates entry if pool doesn't exist

### 3.13 Snapshot Export

- **B-REG-120**: snapshots returns all pools as PoolSnapshot array
- **B-REG-121**: snapshots are sorted by pool_id (deterministic order)
- **B-REG-122**: snapshots include all registry fields
- **B-REG-123**: Empty registry returns empty snapshot array

---

## 4. Process Lifecycle Behaviors

### 4.1 Process Spawn

- **B-PROC-001**: Command is built with binary_path from PreparedEngine
- **B-PROC-002**: All flags from PreparedEngine are passed as arguments
- **B-PROC-003**: Stdout is redirected to log file
- **B-PROC-004**: Stderr is redirected to log file
- **B-PROC-005**: Log file is created in .runtime/ directory
- **B-PROC-006**: Log file is opened in append mode

### 4.2 Process Stop (stop_pool)

- **B-PROC-010**: stop_pool reads PID from .runtime/{pool_id}.pid
- **B-PROC-011**: stop_pool sends SIGTERM first (graceful)
- **B-PROC-012**: stop_pool waits up to 5 seconds for graceful shutdown
- **B-PROC-013**: stop_pool sends SIGKILL if process still alive after grace period
- **B-PROC-014**: stop_pool removes PID file after stopping
- **B-PROC-015**: stop_pool returns error if PID file doesn't exist

### 4.3 Process Monitoring

- **B-PROC-020**: Process ID is captured after spawn
- **B-PROC-021**: Process ID is written to PID file
- **B-PROC-022**: PID file format is plain text with process ID

---

## 5. Preflight & Validation Behaviors (OC-POOL-3012)

### 5.1 CUDA Detection

- **B-PREFLIGHT-001**: cuda_available returns true when nvcc is on PATH
- **B-PREFLIGHT-002**: cuda_available returns true when nvidia-smi is on PATH
- **B-PREFLIGHT-003**: cuda_available returns false when neither tool is found
- **B-PREFLIGHT-004**: cuda_available checks PATH environment variable

### 5.2 GPU-Only Enforcement

- **B-PREFLIGHT-010**: assert_gpu_only succeeds when CUDA is available
- **B-PREFLIGHT-011**: assert_gpu_only fails when CUDA is not available
- **B-PREFLIGHT-012**: assert_gpu_only error message mentions GPU-only enforcement
- **B-PREFLIGHT-013**: assert_gpu_only error message mentions missing nvcc/nvidia-smi

---

## 6. Drain & Reload Behaviors (TODO - Not Yet Implemented)

### 6.1 Drain Lifecycle (OC-POOL-3010)

- **B-DRAIN-001**: Drain request sets draining flag in registry
- **B-DRAIN-002**: Draining pool refuses new lease allocations
- **B-DRAIN-003**: Draining pool allows existing leases to complete
- **B-DRAIN-004**: Drain waits for active_leases to reach 0
- **B-DRAIN-005**: Drain force-stops after deadline_ms expires
- **B-DRAIN-006**: Drain stops engine process after leases drain
- **B-DRAIN-007**: Drain updates registry health to not ready

### 6.2 Reload Lifecycle

- **B-RELOAD-001**: Reload drains pool first
- **B-RELOAD-002**: Reload stages new model via model-provisioner
- **B-RELOAD-003**: Reload stops old engine process
- **B-RELOAD-004**: Reload starts new engine with new model
- **B-RELOAD-005**: Reload waits for new engine health check
- **B-RELOAD-006**: Reload sets ready=true on success
- **B-RELOAD-007**: Reload rolls back on failure (atomic)
- **B-RELOAD-008**: Reload updates engine_version in registry
- **B-RELOAD-009**: Reload preserves pool_id and device_mask

---

## 7. Supervision & Backoff Behaviors (TODO - Not Yet Implemented)

### 7.1 Crash Detection (OC-POOL-3010)

- **B-SUPER-001**: Supervisor detects when engine process exits
- **B-SUPER-002**: Supervisor detects when health check fails
- **B-SUPER-003**: Supervisor detects driver/CUDA errors
- **B-SUPER-004**: Supervisor transitions pool to unready on crash

### 7.2 Exponential Backoff (OC-POOL-3011)

- **B-SUPER-010**: First restart has initial_ms delay
- **B-SUPER-011**: Subsequent restarts double delay (exponential)
- **B-SUPER-012**: Backoff delay is capped at max_ms
- **B-SUPER-013**: Backoff includes jitter to prevent thundering herd
- **B-SUPER-014**: Backoff resets after stable run period

### 7.3 Circuit Breaker (OC-POOL-3011)

- **B-SUPER-020**: Circuit opens after N consecutive failures
- **B-SUPER-021**: Open circuit prevents restart attempts
- **B-SUPER-022**: Circuit transitions to half-open after timeout
- **B-SUPER-023**: Half-open allows single test restart
- **B-SUPER-024**: Circuit closes after successful test restart
- **B-SUPER-025**: Circuit reopens if test restart fails

### 7.4 Restart Storm Prevention

- **B-SUPER-030**: Restart counter increments on each restart
- **B-SUPER-031**: Restart counter resets after stable period
- **B-SUPER-032**: Restart storms are logged with restart_count
- **B-SUPER-033**: Maximum restart rate is enforced

---

## 8. Device Mask & Placement Behaviors (TODO - Not Yet Implemented)

### 8.1 Device Mask Parsing (OC-POOL-3020)

- **B-DEVICE-001**: DeviceMask parses comma-separated GPU IDs (e.g., "0,1")
- **B-DEVICE-002**: DeviceMask parses named GPUs (e.g., "GPU0,GPU1")
- **B-DEVICE-003**: DeviceMask validates against discovered devices
- **B-DEVICE-004**: DeviceMask rejects invalid device IDs
- **B-DEVICE-005**: DeviceMask converts to CUDA_VISIBLE_DEVICES format

### 8.2 Placement Enforcement (OC-POOL-3020)

- **B-DEVICE-010**: Engine only uses GPUs in device mask
- **B-DEVICE-011**: No cross-mask spillover occurs
- **B-DEVICE-012**: CUDA_VISIBLE_DEVICES is set correctly
- **B-DEVICE-013**: Device mask is persisted in registry

### 8.3 Heterogeneous Split Planning (OC-POOL-3021)

- **B-HETERO-001**: SplitPlan computes ratios for multi-GPU setup
- **B-HETERO-002**: Split ratios sum to 1.0
- **B-HETERO-003**: Smallest GPU capacity is respected
- **B-HETERO-004**: Split ratios are capped for smallest GPU
- **B-HETERO-005**: SplitPlan validates model fits in smallest GPU share
- **B-HETERO-006**: Default is no split (single GPU)

---

## 9. Observability Behaviors (OC-POOL-3030)

### 9.1 Metrics Emission (TODO - Not Yet Implemented)

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

### 9.2 Structured Logging

- **B-LOG-001**: Logs include pool_id field
- **B-LOG-002**: Logs include engine field
- **B-LOG-003**: Logs include engine_version field
- **B-LOG-004**: Logs include device_mask field
- **B-LOG-005**: Logs include restart_count on restart
- **B-LOG-006**: Logs include backoff_ms on backoff
- **B-LOG-007**: Logs include last_error on failure
- **B-LOG-008**: Logs are structured JSON format

---

## 10. Configuration Behaviors (TODO - Not Yet Implemented)

### 10.1 Environment Variables

- **B-CONFIG-001**: POOL_MANAGERD_ADDR sets bind address
- **B-CONFIG-002**: POOL_MANAGERD_ADDR defaults to 127.0.0.1:9200
- **B-CONFIG-003**: Invalid bind address returns error

### 10.2 Device Configuration

- **B-CONFIG-010**: Device masks are configurable per pool
- **B-CONFIG-011**: Device masks are validated against discovered devices
- **B-CONFIG-012**: Invalid device mask returns error

---

## 11. Security Behaviors (TODO - Not Yet Implemented)

### 11.1 Process Isolation

- **B-SEC-001**: Engine processes run as non-root user
- **B-SEC-002**: Engine processes have isolated workdir
- **B-SEC-003**: Engine processes have minimal capabilities

### 11.2 Container Runtime (if used)

- **B-SEC-010**: Containers run rootless (podman preferred)
- **B-SEC-011**: Containers have seccomp profile
- **B-SEC-012**: Containers have constrained network access

---

## 12. Error Handling Behaviors

### 12.1 Registry Lock Failures

- **B-ERROR-001**: Registry lock failure returns 500 error
- **B-ERROR-002**: Registry lock failure includes error message
- **B-ERROR-003**: Registry lock failure is logged

### 12.2 Process Spawn Failures

- **B-ERROR-010**: Invalid binary path returns error
- **B-ERROR-011**: Process spawn failure is logged
- **B-ERROR-012**: Process spawn failure updates registry last_error

### 12.3 Health Check Failures

- **B-ERROR-020**: Health check timeout is logged
- **B-ERROR-021**: Health check failure updates registry last_error
- **B-ERROR-022**: Health check failure kills spawned process

---

## Summary Statistics

- **Total Behaviors Identified**: 200+
- **Implemented & Testable Now**: ~80 (API, Registry, Preload, Preflight)
- **TODO (Not Yet Implemented)**: ~120 (Drain, Reload, Supervision, Device Masks, Metrics)

## Priority Grouping for BDD Implementation

### Phase 1: Core Behaviors (Ready Now)

- HTTP API (B-API-*)
- Registry State Management (B-REG-*)
- Preload & Readiness (B-PRELOAD-*)
- Process Lifecycle (B-PROC-*)
- Preflight Validation (B-PREFLIGHT-*)

### Phase 2: Lifecycle Behaviors (Requires Implementation)

- Drain & Reload (B-DRAIN-*, B-RELOAD-*)
- Supervision & Backoff (B-SUPER-*)

### Phase 3: Advanced Behaviors (Requires Implementation)

- Device Masks & Placement (B-DEVICE-*, B-HETERO-*)
- Observability (B-METRICS-*, B-LOG-*)
- Configuration (B-CONFIG-*)
- Security (B-SEC-*)

## Spec Traceability

Each behavior is traceable to:

- **Spec Requirements**: OC-POOL-3xxx from `.specs/30-pool-managerd.md`
- **Checklist Items**: From `CHECKLIST.md`
- **Source Code**: From `src/` modules

## Next Steps

1. Create feature files for Phase 1 behaviors (80 behaviors)
2. Implement step definitions for Phase 1
3. Add step registry and validation test
4. Implement Phase 2 & 3 code, then add BDD coverage

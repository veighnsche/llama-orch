# Worker-orcd SPEC — Custom GPU Worker Implementation (WORKER-4xxx)

**Status**: Draft (architecture planning)  
**Applies to**: `bin/worker-orcd/` (main binary)  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope & Goals

This specification defines the high-level requirements for `worker-orcd`, the custom GPU worker daemon that replaces external engine adapters (llama.cpp, vLLM, TGI, Triton) with direct VRAM control and sealed shard guarantees.

**Key objectives**:
- Provide deterministic, VRAM-only inference with no RAM offload (WORKER-4000)
- Implement sealed ModelShardHandle contract for audited staging (WORKER-4001)
- Enable tensor-parallel execution via NCCL coordination (WORKER-4002)
- Support Model Capability Descriptor (MCD) / Engine Capability Profile (ECP) matching (WORKER-4003)

**Reference documents**:
- `.docs/ARCHITECTURE_CHANGE_PLAN.md` — Strategic architecture and design
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Security requirements
- `.docs/security/SECURITY_AUDIT_EXISTING_CODEBASE.md` — Operational security
- `.specs/00_llama-orch.md` — System-wide requirements

**Component specs** (detailed requirements distributed):
- `bin/worker-orcd-crates/api/.specs/00_api.md` — RPC endpoints (§3)
- `bin/worker-orcd-crates/vram-residency/.specs/00_vram-residency.md` — VRAM policy (§2)
- `bin/worker-orcd-crates/model-loader/.specs/00_model-loader.md` — Model validation (§4)
- `bin/worker-orcd-crates/capability-matcher/.specs/00_capability-matcher.md` — MCD/ECP (§7)
- `bin/worker-orcd-crates/scheduler/.specs/00_scheduler.md` — Job scheduling
- `bin/worker-orcd/cuda/kernels/.specs/00_cuda-kernels.md` — CUDA kernels (§8)

---

## 1. Architecture & Process Model

### 1.1 Binary Structure

- [WORKER-4010] `worker-orcd` MUST be a hybrid Rust + CUDA C++ binary with clear FFI boundaries.
- [WORKER-4011] Rust MUST handle: HTTP server, RPC protocol, lifecycle management, VRAM residency enforcement, telemetry, and scheduling hooks.
- [WORKER-4012] CUDA C++ MUST handle: compute kernels (cuBLAS, attention, RoPE, sampling), NCCL coordination, and GPU memory operations.
- [WORKER-4013] The FFI boundary MUST use `cudarc` or `cust` for safe CUDA bindings with explicit ownership of VRAM pointers.

### 1.2 Process Model

- [WORKER-4020] One `worker-orcd` process MUST run per GPU device (or device mask for multi-GPU).
- [WORKER-4021] `pool-managerd` MUST spawn and supervise worker processes; workers MUST NOT self-spawn.
- [WORKER-4022] Workers MUST accept a single model per process (one-model/one-device-mask constraint per ORCH-3001).
- [WORKER-4023] Workers MUST report readiness to `pool-managerd` via HTTP callback after successful model load.

### 1.3 Lifecycle States

- [WORKER-4030] Worker lifecycle states: `Starting → Loading → Ready → Executing → Draining → Stopped`.
- [WORKER-4031] Transitions to `Ready` MUST only occur after model is sealed in VRAM and digest verified.
- [WORKER-4032] Driver/CUDA errors MUST transition worker to `Stopped` state and trigger restart via `pool-managerd`.

---

## 2. VRAM Residency & Sealed Shard Contract

**Detailed spec**: `bin/worker-orcd-crates/vram-residency/.specs/00_vram-residency.md`

**Summary**:
- VRAM-only policy (WORKER-4100-4103)
- ModelShardHandle contract (WORKER-4110-4113)
- Seal integrity with HMAC-SHA256 (WORKER-4120-4122)

---

## 3. RPC Protocol (Plan/Commit/Ready/Execute)

**Detailed spec**: `bin/worker-orcd-crates/api/.specs/00_api.md`

**Summary**:
- Endpoint authentication (WORKER-4200-4203)
- Plan endpoint (WORKER-4210-4214)
- Commit endpoint (WORKER-4220-4227)
- Ready endpoint (WORKER-4230-4233)
- Execute endpoint (WORKER-4240-4248)
- SSE streaming security (WORKER-4250-4253)

---

## 4. Input Validation & Model Security

**Detailed specs**:
- `bin/worker-orcd-crates/model-loader/.specs/00_model-loader.md` — Model validation
- `libs/shared-crates/input-validation/.specs/` — Request validation (shared)

**Summary**:
- Request validation (WORKER-4300-4305)
- Model validation (WORKER-4310-4314)
- Path validation (WORKER-4320-4323)

---

## 5. Memory Safety & CUDA FFI

**Detailed spec**: `bin/worker-orcd/src/cuda_ffi/` (implementation)

**Summary**:
- CUDA FFI boundary (WORKER-4400-4403)
- Bounds checking (WORKER-4410-4413)
- Resource limits (WORKER-4420-4423)

---

## 6. Tensor-Parallel Support (NCCL)

**Post-M0 feature** (deferred)

**Summary**:
- Multi-GPU sharding (WORKER-4500-4503)
- NCCL security (WORKER-4510-4513)
- Coordinator trust (WORKER-4520-4522)

---

## 7. Model Capability Matching (MCD/ECP)

**Detailed spec**: `bin/worker-orcd-crates/capability-matcher/.specs/00_capability-matcher.md`

**Summary**:
- Model Capability Descriptor (WORKER-4600-4603)
- Engine Capability Profile (WORKER-4610-4613)
- Capability matching (WORKER-4620-4623)

---

## 8. Inference Kernels & Determinism

**Detailed spec**: `bin/worker-orcd/cuda/kernels/.specs/00_cuda-kernels.md`

**Summary**:
- M0 kernel set (WORKER-4700-4703)
- Determinism (WORKER-4710-4713)
- Post-M0 optimizations (WORKER-4720-4722)

---

## 9. Observability & Telemetry

### 9.1 Structured Logging

- [WORKER-4800] Workers MUST emit structured logs with fields: `job_id`, `worker_id`, `gpu_device`, `model_ref`, `tokens_in`, `tokens_out`, `decode_time_ms`.
- [WORKER-4801] Workers MUST include correlation ID in all logs (from `X-Correlation-Id` header or generated).
- [WORKER-4802] Workers MUST NOT log secrets, tokens, or PII; redaction MUST be enforced.
- [WORKER-4803] Workers MUST emit human-readable narration (`human` field) alongside structured fields per ORCH-3300.

### 9.2 Metrics

- [WORKER-4810] Workers MUST expose Prometheus metrics: VRAM usage, job latency, token throughput, error counts.
- [WORKER-4811] Workers MUST emit metrics for: jobs started, jobs completed, jobs failed, tokens generated, CUDA errors.
- [WORKER-4812] Workers MUST expose metrics endpoint at `/metrics` (unauthenticated for scraping).

### 9.3 Narration Hooks

- [WORKER-4820] Workers SHOULD emit narration at key lifecycle points: model load, seal, inference start, inference end, error.
- [WORKER-4821] Narration text MUST be natural-language and MUST NOT consist primarily of opaque identifiers (per ORCH-3312).
- [WORKER-4822] Workers SHOULD include identity breadcrumbs (token fp6) in narration when authentication is active.

---

## 10. Security & Privilege Management

### 10.1 Process Privileges

- [WORKER-4900] Workers MUST run as non-root user (dedicated `worker-orcd` user).
- [WORKER-4901] Workers MUST drop unnecessary capabilities after GPU initialization.
- [WORKER-4902] Workers MUST NOT require root privileges for normal operation.
- [WORKER-4903] `pool-managerd` MUST spawn workers with reduced privileges (via `sudo -u worker-orcd` or capabilities).

### 10.2 Credential Management

- [WORKER-4910] Workers MUST read API tokens from file (not environment variables).
- [WORKER-4911] Token file path MUST be configurable via `WORKER_API_TOKEN_FILE` env var (default `/etc/llorch/worker-token`).
- [WORKER-4912] Workers MUST use systemd `LoadCredential` for token delivery when available.
- [WORKER-4913] Workers MUST fail fast if token file is missing or unreadable.

### 10.3 Process Isolation

- [WORKER-4920] Workers SHOULD run in containers (Docker/Podman) or Linux namespaces for isolation.
- [WORKER-4921] Workers SHOULD use SELinux or AppArmor policies to restrict filesystem access.
- [WORKER-4922] Workers MUST NOT have access to other workers' VRAM or model files.

---

## 11. Error Handling & Recovery

### 11.1 Error Taxonomy

- [WORKER-4950] Workers MUST use stable error codes: `VRAM_OOM`, `CUDA_ERROR`, `MODEL_LOAD_FAILED`, `SEAL_VERIFICATION_FAILED`, `INVALID_PARAMS`, `AUTH_FAILED`, `INTERNAL`.
- [WORKER-4951] Errors MUST include: `code`, `message`, `retriable: bool`, `retry_after_ms: Option<u64>`.
- [WORKER-4952] Workers MUST distinguish VRAM OOM from host OOM with specific error codes.

### 11.2 Failure Recovery

- [WORKER-4960] Workers MUST transition to `Stopped` state on CUDA driver errors.
- [WORKER-4961] Workers MUST notify `pool-managerd` of failure via status callback.
- [WORKER-4962] `pool-managerd` MUST restart failed workers with exponential backoff.
- [WORKER-4963] Workers MUST implement circuit breaker to prevent restart storms (max 5 restarts in 5 minutes).

### 11.3 Graceful Shutdown

- [WORKER-4970] Workers MUST handle SIGTERM gracefully: drain in-flight jobs, save state, exit cleanly.
- [WORKER-4971] Workers MUST complete in-flight inference jobs before shutdown (with timeout).
- [WORKER-4972] Workers MUST release VRAM and CUDA resources on shutdown.

---

## 12. Configuration & Deployment

### 12.1 Configuration Schema

- [WORKER-4980] Workers MUST accept configuration via JSON file or command-line arguments.
- [WORKER-4981] Required config fields: `worker_id`, `pool_id`, `model_path`, `rpc_port`, `readiness_callback_url`, `gpu_device`.
- [WORKER-4982] Optional config fields: `slots_total`, `max_tokens`, `max_prompt_len`, `execution_timeout_ms`, `api_token_file`.
- [WORKER-4983] Workers MUST validate configuration at startup and fail fast on invalid config.

### 12.2 Environment Variables

- [WORKER-4990] Workers MUST support environment variables for deployment flexibility:
  - `WORKER_API_TOKEN_FILE` — Path to API token file
  - `WORKER_BIND_ADDR` — RPC server bind address (default `127.0.0.1:8001`)
  - `WORKER_GPU_DEVICE` — CUDA device index (default `0`)
  - `WORKER_LOG_LEVEL` — Log level (default `info`)
- [WORKER-4991] Workers MUST validate `WORKER_BIND_ADDR` format and port range (1024-65535 recommended).

---

## 13. Testing & Validation

### 13.1 Unit Tests

- [WORKER-4995] Workers MUST have unit tests for: GGUF parsing, MCD/ECP matching, input validation, seal verification, FFI boundary.
- [WORKER-4996] Unit tests MUST use proof-bundle outputs per `.specs/00_proof-bundle.md`.

### 13.2 Integration Tests

- [WORKER-4997] Workers MUST have integration tests for: model load, inference execution, SSE streaming, error handling, authentication.
- [WORKER-4998] Integration tests MUST use real GPU (via `REQUIRE_REAL_GPU=1` env var).

### 13.3 Determinism Tests

- [WORKER-4999] Workers MUST pass determinism suite: identical seeds produce identical token streams.
- [WORKER-5000] Determinism tests MUST run on multiple GPUs to verify consistency.

---

## 14. Traceability

**Code**:
- `bin/worker-orcd/src/main.rs` — Main entry point
- `bin/worker-orcd/src/server.rs` — RPC server
- `bin/worker-orcd/src/ffi.rs` — CUDA FFI boundary
- `bin/worker-orcd/cuda/` — CUDA kernels

**Tests**:
- `bin/worker-orcd/tests/` — Integration tests
- `bin/worker-orcd/bdd/` — BDD scenarios

**Related specs**:
- `.specs/00_llama-orch.md` — System-wide requirements
- `.specs/20-orchestratord.md` — Orchestrator spec
- `.specs/30-pool-managerd.md` — Pool manager spec

---

## 15. Refinement Opportunities

- **mTLS support**: Add mutual TLS for internal communication (post-M0).
- **Job tokens**: Implement short-lived, signed job tokens for scoped authorization.
- **Certificate rotation**: Add graceful credential rotation with overlapping validity.
- **Advanced kernels**: FlashAttention, PagedAttention, fused kernels for throughput.
- **Multi-format support**: Safetensors, PyTorch checkpoints beyond GGUF.
- **Quantization**: Runtime quantization (Q5_1, Q8_0, AWQ, GPTQ).

---

**End of Specification**

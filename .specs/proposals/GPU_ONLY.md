# Proposal: GPU-Only Execution — No CPU or RAM Fallback

**Status:** Draft
**Owner:** @llama-orch-maintainers
**Date:** 2025-09-27

## 0) Motivation

Determinism, reproducibility, and performance require strict control of execution environments. Allowing fallbacks (CPU inference or hybrid VRAM+RAM offloading) introduces:

* **Unpredictable latency** — sudden spillovers to CPU or host RAM cause orders-of-magnitude slowdowns.
* **Inconsistent results** — replicas with different fallback behavior produce divergent performance.
* **Complex debugging** — ops and users cannot reason about why performance degraded.

This proposal enforces **VRAM-only execution**. Tasks that do not fit must fail fast, never silently fallback.

---

## 1) Scope

**In scope:**

* Engine provisioning and runtime policies.
* Placement decisions (fail early if VRAM insufficient).
* Error taxonomy and logs for insufficient VRAM.

**Out of scope:**

* Runtime paging/swap to system RAM.
* CPU inference support (already rejected).
* Any hybrid strategies (tensor offloading, ZeRO-offload, etc.).

---


IDs use ORCH-38xx (GPU-only enforcement).

### Execution policy

* [ORCH-3800] Engines MUST execute inference entirely within GPU VRAM.
* [ORCH-3801] Engines MUST NOT allocate or page model weights/tensors into host RAM as a fallback.
* [ORCH-3802] CPU inference paths MUST be disabled; operators MUST fail fast on insufficient VRAM.
* [ORCH-3803] Engines MUST NOT enable unified memory (UMA), zero-copy, pinned/page-locked host memory, or BAR/Resizable-BAR modes to keep any portion of model weights outside VRAM at runtime.
* [ORCH-3804] KV cache, activations, and intermediate tensors MUST reside in VRAM for the duration of inference. Host RAM is permitted only for non-runtime duties (download, staging, decompression), never for live decode.

### Placement & Admission

* [ORCH-3810] Scheduler MUST validate VRAM requirements before admission. If the model does not fit, admission MUST fail with error code `POOL_UNAVAILABLE`.
* [ORCH-3811] Partial placement (split across RAM + VRAM) is forbidden. No attempt to “fit partially.”
{{ ... }}
* [ORCH-3821] Narration logs MUST clearly state when admission was rejected due to VRAM limits.

### Provisioning

* [ORCH-3830] Engine provisioners MUST validate VRAM availability up front.
* [ORCH-3831] Provisioning MUST fail with a clear diagnostic when VRAM is insufficient; no fallback to RAM or CPU is allowed.
* [ORCH-3832] Provisioning MUST explicitly disable RAM offload/unified-memory modes when engines expose such toggles (e.g., `--no-offload`, `--disable-unified-memory`).

---

## 3) Design Overview

{{ ... }}
* **Provisioners:** Validate CUDA/device memory early; abort if too small.
* **Engines:** Compile/run only in VRAM mode; ensure flags forbid offloading (e.g., `--no-offload`, disable unified memory/zero-copy if present in the engine).
* **Logs/Telemetry:** Explicit error surfaced and logged when VRAM is insufficient.

Operator guidance (non-normative): Some engines advertise host-RAM offload or UMA for "larger than VRAM" models. This program bans such modes. If a model does not fit fully in VRAM (including KV/activations within configured bounds), admission must reject with `POOL_UNAVAILABLE` instead of attempting degraded execution.

---

## 4) Migration Plan

**Phase 1 — Policy Enforcement**

* Remove or disable offload flags in supported engines (llama.cpp, vLLM, TGI, Triton).
* Add VRAM validation step in provisioners.

**Phase 2 — Admission Integration**

* Propagate VRAM validation into orchestrator scheduler.
* Ensure `POOL_UNAVAILABLE` returned when a task cannot fit.

**Phase 3 — Observability**

* Narration + metrics for “VRAM insufficient” cases.
* Telemetry samples (if enabled) marked with `exit=error`, `error_code=POOL_UNAVAILABLE`.

---

## 5) Risks & Mitigations

* **Reduced coverage**: Some small GPUs may be excluded. → Document minimum supported GPUs.
* **Engine drift**: Some engines default to CPU fallback. → Enforce engine flags and config audits.
* **Operator surprise**: Users may expect slower fallback instead of failure. → Document policy explicitly; fail fast with clear error.

---

## 6) Acceptance Criteria

* Engines run only in VRAM; no CPU/RAM offload possible.
* Admission rejects jobs that don’t fit VRAM with `POOL_UNAVAILABLE`.
* Logs and telemetry clearly record VRAM rejection events.
* Configs and docs updated to reflect **GPU-only** support.

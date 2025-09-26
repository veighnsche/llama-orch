# Proposal: Performance Telemetry & Model–GPU Analytics (Repo-Wide)

**Status:** Draft
**Owner:** @llama-orch-maintainers
**Date:** 2025-09-27

## 0) Motivation

Operators and users need visibility into how different models perform on specific GPUs. This allows:

* Picking the right hardware for cost/performance.
* Building trust in published benchmarks by aggregating **real-world usage data**.
* Identifying regressions and bottlenecks across versions.

This proposal introduces **opt-in telemetry** that collects anonymized performance data for model–GPU pairings. The system respects user choice, avoids PII, and integrates with existing logging, metering, and observability work.

---

## 1) Scope

**In scope:**

* Runtime metrics capture for model execution on GPUs.
* Opt-in telemetry pipeline (local buffer → periodic batch).
* Aggregated analytics for model–GPU compatibility/performance.
* Minimal schema definition for transport.

**Out of scope:**

* Mandatory data collection (strictly opt-in).
* Fine-grained tracing of prompts/outputs.
* Monetization/billing (covered by Cloud Profile).

---

## 2) Normative Requirements (RFC-2119)

IDs use ORCH-37xx (performance telemetry).

### Collection

* \[ORCH-3700] When enabled, the orchestrator MUST collect **per-task performance samples**:

  * `model_ref`
  * `engine` + `engine_version`
  * `gpu_name` (from NVML or adapter props)
  * `vram_total_mb`, `vram_used_mb_peak`
  * `tokens_in`, `tokens_out`, `decode_time_ms`, `first_token_ms`
  * `exit` (`ok|cancel|error|reject`)
* \[ORCH-3701] Data MUST be stripped of PII and prompt text. Only metadata + timings allowed.
* \[ORCH-3702] Samples MUST be buffered locally and sent in batches; if disabled, they MUST NOT leave the host.

### Consent & Control

* \[ORCH-3710] Telemetry MUST be **opt-in** via config/env (`telemetry.enabled=true`). Default = off.
* \[ORCH-3711] Users MUST be able to inspect the telemetry buffer (`orch telemetry show`) before sending.
* \[ORCH-3712] An environment flag (`ORCH_TELEMETRY_ANON=1`) MUST anonymize tenant IDs before upload.

### Transport

* \[ORCH-3720] Telemetry MUST support at least two sinks:

  * `jsonl` file (local export).
  * HTTP POST to a configured endpoint (batch).
* \[ORCH-3721] Upload MUST be non-blocking. If sink stalls, drop oldest samples with counters.

### Aggregation & Use

* \[ORCH-3730] Aggregated telemetry MAY be used to produce public compatibility/performance matrices (model × GPU × engine).
* \[ORCH-3731] Individual samples MUST NOT be exposed publicly without aggregation/anonymization.

---

## 3) Design Overview

* **Emitters:** Extend orchestrator’s streaming end event (`end`) to also write a telemetry sample if enabled.
* **Buffer:** Lightweight local ring buffer (JSONL). Configurable path and retention.
* **Uploader:** Background worker ships batches (size- or time-based) to sink.
* **Schema:** Reuse Cloud Profile usage schema, extended with GPU perf fields.
* **Opt-in UX:** Config flag, CLI toggles, docs clarifying that telemetry is optional.

---

## 4) Schema (Telemetry Sample)

```json
{
  "ts": "2025-09-27T12:00:00Z",
  "tenant_id": "anon", 
  "job_id": "uuid", 
  "model_ref": "hf:org/repo",
  "engine": "vllm",
  "engine_version": "0.5.2",
  "gpu_name": "NVIDIA A100-SXM4-40GB",
  "vram_total_mb": 40536,
  "vram_used_mb_peak": 18234,
  "tokens_in": 1024,
  "tokens_out": 2048,
  "first_token_ms": 420,
  "decode_time_ms": 9876,
  "exit": "ok"
}
```

---

## 5) Migration Plan

**Phase 1 — Local only**

* Implement emitters + local JSONL buffer.
* Add config toggle.

**Phase 2 — Upload**

* Add HTTP sink, batching, drop-oldest policy.
* Publish schema in `/contracts/telemetry_event.schema.json`.

**Phase 3 — Aggregation**

* Build public model–GPU matrix from anonymized data.
* Expose via docs site or API.

---

## 6) Risks & Mitigations

* **Privacy concerns** → Opt-in only; no prompt text logged.
* **Overhead in hot path** → Minimal: append JSON line on task end.
* **Skewed data** → Mark opt-in source; combine with curated benchmarks.

---

## 7) Acceptance Criteria

* Orchestrator can be started with `telemetry.enabled=true` and produces samples.
* Telemetry buffer viewable via CLI.
* JSON schema for telemetry exists in `/contracts/`.
* Aggregated outputs show relative performance (tokens/s, first token latency) across GPUs.
# Proposal: Batching Semantics — Single-Stream vs Multi-Stream

**Status:** Draft
**Owner:** @llama-orch-maintainers
**Date:** 2025-09-28

---

## 0) Motivation

Today, terms like *active\_slots*, *queueing*, and *multi-tenant* are used without a clear definition of what “batching” means inside llama-orch. This causes confusion in:

* **Financial simulations:** consumer GPUs should be modeled with **single-stream** TPS numbers, while a **Public Tap** runs on batched throughput.
* **Benchmarks:** external sources often report *online/offline throughput*, which is not the same as single-stream.
* **Operators:** Home-profile users expect one request at a time; Cloud-profile operators want maximum efficiency via batching.

A formal spec avoids ambiguity and ensures results are comparable and reproducible.

---

## 1) Scope

**In scope:**

* Definitions of *single-stream*, *batched-online*, and *offline throughput*.
* Normative rules for how batching is surfaced in orchestrator (slots, capabilities).
* Profile defaults: Home vs Cloud/Public Tap.
* Measurement and reporting requirements (telemetry, benchmarks, CSV inputs).

**Out of scope:**

* New scheduling algorithms (covered elsewhere).
* GPU-only execution (see `GPU_ONLY.md`).
* Token streaming (see `token-streaming-and-cancel-robustness.md`).

---

## 2) Normative Requirements (RFC-2119)

IDs use ORCH-39xx (batching).

### Definitions

* \[ORCH-3900] **Single-stream** = `batch=1`, `max_num_seqs=1`, one user, one request at a time.
* \[ORCH-3901] **Batched-online** = multiple concurrent requests merged into one decode step (continuous batching). Per-user latency remains acceptable.
* \[ORCH-3902] **Offline throughput** = maximum tokens/s with an artificially large batch (benchmark mode). Not representative of user latency.

### Engine/Adapter Reporting

* \[ORCH-3910] Each engine/adapter MUST report whether batching is active (`batching=true|false`) in `/v1/capabilities`.
* \[ORCH-3911] `slots_total` and `slots_free` MUST correspond to batch capacity.
* \[ORCH-3912] In single-stream mode, `slots_total` MUST equal 1.

### Orchestrator Behavior

* \[ORCH-3920] Home-profile deployments MUST default to single-stream (`slots_total=1`).
* \[ORCH-3921] Cloud/Public Tap deployments MUST default to batched-online (continuous batching).
* \[ORCH-3922] Admission MUST respect whether a pool allows batching; oversubscription is forbidden.

### Benchmarking & Telemetry

* \[ORCH-3930] Telemetry MUST record which `measurement_type` was active (`single_stream | batched_online | offline`).
* \[ORCH-3931] CSV facts (e.g. `tps_model_gpu.csv`) MUST set the `measurement_type` column accordingly.
* \[ORCH-3932] Scenario notes MUST include `input_tokens`, `output_tokens`, and batching status.

---

## 3) Design Overview

* **Data plane:**
  `TaskRequest` remains unchanged. Placement decides whether a request goes to a single-stream pool or a batched pool.

* **Control plane:**
  `/v1/capabilities` is extended with a `batching` flag and `slots_total`.

* **Telemetry:**
  Performance samples include `measurement_type`.

* **Financial models:**

  * Consumer simulation = single-stream TPS.
  * Public Tap economics = batched-online TPS.

* **Documentation:**
  Benchmarks (DBM, internal measurements) are labeled with exact `measurement_type`.

---

## 4) Migration Plan

**Phase A — Spec & Contracts**

* Add `batching` field to `/v1/capabilities`.
* Define allowed `measurement_type` values (`single_stream`, `batched_online`, `offline`).

**Phase B — Engines & Adapters**

* Engines expose correct `slots_total`.
* Adapter-host cache includes batching flag.

**Phase C — Orchestrator & Profiles**

* Home-profile config defaults to single-stream.
* Cloud-profile config defaults to batched-online.

**Phase D — Telemetry & CSV**

* Extend `PERF_TELEMETRY.md` with `measurement_type`.
* Enforce `tps_model_gpu.csv` schema for labeling.

---

## 5) Risks & Mitigations

* **Confusion with “concurrent jobs”** → mitigation: strict mapping of `slots` to batching semantics.
* **Benchmark drift** → mitigation: scenario notes required.
* **Operator surprises** (different defaults Home vs Cloud) → mitigation: document defaults clearly.

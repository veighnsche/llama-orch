# 01 — Problem and Scope

## Problem Statement (verbatim excerpts)

.specs/00_llama-orch.md:9–14
> ## 0. Scope & Goals
> 
> * Provide deterministic, observable orchestration of NVIDIA-backed LLM workers on a single workstation with one or more GPUs. (ORCH-3000)
> * Keep configuration lightweight: filesystem storage, no clustered control plane. (ORCH-3002)
> * Optimise for multi-agent developer workflows: low queue latency, clear feedback, reliable catalog/artifact handling. (ORCH-3003)

.docs/HOME_PROFILE.md:11–16
> - MUST run all orchestrator services on a single host with one or more NVIDIA GPUs (mixed VRAM is expected).
> - MUST allow the developer box to drive the system remotely (SSH tunnel or explicit bind) while defaulting to loopback-only exposure.
> - MUST keep configuration lightweight: filesystem-backed storage, single API token, no external control plane.
> - SHOULD prioritise determinism, observability, and quick debugging over throughput tuning.
> - MUST keep the Spec → Contract → Tests → Code workflow intact (see `.docs/PROCESS.md`).

.specs/00_home_profile.md:9–13
> - [HME-001] Orchestrator, pool manager, adapters, and artifact storage MUST run on the same workstation.
> - [HME-002] Remote access MUST default to loopback with opt-in LAN exposure guarded by firewall or SSH tunnel.
> - [HME-003] Configuration MUST live in a single YAML/TOML file plus environment variables for sensitive values (API token paths, optional tunnels).
> - [HME-004] Minimal Auth seam: when exposing beyond loopback, configure `AUTH_TOKEN` and related keys per `/.specs/11_min_auth_hooks.md`.

.specs/00_llama-orch.md:122–131
> * The program provisions and manages engines automatically per pool; operators SHOULD NOT be required to pre-install or manually launch engines. (ORCH-3200)
> * Provisioning modes ... `external|source|container|package|binary`.
> * Preflight MUST check required tools and environment ... provision missing tools when policy allows, or fail fast with actionable guidance. (ORCH-3202)
> * Home profile Arch/CachyOS: when `allow_package_installs=true`, the system MAY install missing tools via `pacman`/AUR ... (ORCH-3204)

### Consolidated statement
llama-orch targets a single-host, NVIDIA-GPU home-lab environment. Its problem is to deterministically and observably orchestrate LLM workloads with low-latency admission→dispatch→SSE streaming, minimal configuration, and reliable catalog/artifact handling. It must auto-provision engines and models (policy-gated), respect GPU placement constraints, and expose simple HTTP contracts, metrics, and logs for developers running agents via CLI/IDE or CI. Multi-host, multi-tenant, and CPU fallback are out of scope.

## Personas & Use Context

- Developers orchestrating multi-agent workflows from a dev box/IDE/CLI.
  - .docs/HOME_PROFILE.md:19–27
> ### Data Plane (OrchQueue v1)
> - `POST /v1/tasks` … `GET /v1/tasks/{id}/stream` … `POST /v1/tasks/{id}/cancel`
> ### Control Plane
> - Catalog … Pools drain/reload/health … Discovery via `/v1/capabilities`.

- Operators of a home workstation with mixed GPUs.
  - .specs/00_llama-orch.md:19–22
> * Hosts are Linux … NVIDIA drivers + CUDA … Inference MUST run on NVIDIA GPUs … Mixed GPUs supported.

- Environments/constraints
  - Single workstation; loopback by default, optional LAN. (.specs/00_home_profile.md:9–13)
  - GPU-only; no CPU inference fallback. (.specs/00_llama-orch.md:139–140)
  - Policy-gated outbound tooling; optional pacman/AUR installs for Arch. (.specs/00_llama-orch.md:116–117, 133–138)

## Goals / Non-Goals (RFC‑2119)

| Area | MUST/SHOULD/MAY | Statement | Source |
|---|---|---|---|
| Determinism | MUST | Identical {prompt, parameters, seed, sampler_profile_version, engine_version, model_digest} on same replica → identical token streams. | .specs/00_llama-orch.md:58–59; .specs/10-orchestrator-core.md:74–76 |
| Streaming | MUST | SSE events `started`, `token`, optional `metrics`, `end`, `error`; `started` includes `queue_position`, `predicted_start_ms`. | .specs/00_llama-orch.md:60; .specs/20-orchestratord.md:28–31,40–45 |
| Admission | MUST | Bounded FIFO per priority; 429 with backoff headers/body on full queue. | .specs/00_llama-orch.md:35–40 |
| GPU policy | MUST NOT | No CPU inference spillover or fallback; NVIDIA GPUs only. | .specs/30-pool-managerd.md:21–22; .specs/00_llama-orch.md:19–21,139–140 |
| Placement | MUST/SHOULD | Ready-only dispatch; least-loaded with VRAM awareness; session affinity SHOULD hold. | .specs/00_llama-orch.md:43–46 |
| Auto-provision | MUST/SHOULD | Program provisions engines automatically; preflight; Arch pacman/AUR optionally. | .specs/00_llama-orch.md:124–133 |
| Catalog | MUST/SHOULD | Persist model metadata; verification on digests; artifact registry SHOULD exist. | .specs/00_llama-orch.md:64–70 |
| Security | MUST | Redact secrets; minimal auth seam documented; home profile open locally. | .specs/00_llama-orch.md:73–77 |
| Observability | MUST/SHOULD | Logs fields; Prometheus metrics; SSE metrics frames SHOULD include queue/budgets. | .specs/00_llama-orch.md:80–83 |

## Success Criteria

- Tests and harnesses green
  - Provider verification and BDD suites. README_LLM.md:57–66; .docs/TESTING_POLICY.md:30–37
- Determinism suite passes (byte-exact streams, seeds corpus). .specs/70-determinism-suite.md:13–16
- Metrics contract lint passes; required series/labels present. .specs/metrics/otel-prom.md:5–14,107–111
- Home profile smoke passes on reference workstation. .docs/HOME_PROFILE.md:96–99
- SSE transcripts and artifacts persisted with required fields. .specs/20-orchestratord.md:82–86; README.md:228–235

## Top 5 Opportunities

- Add capability discovery examples and API version pinning across CLI/server to harden discovery contracts. (.specs/20-orchestratord.md:76–81)
- Unify `decode_time_ms` vs `decode_ms` field naming across SSE/logs and update examples. (.specs/20-orchestratord.md:43; .specs/00_llama-orch.md:88–89)
- Expand placement modeling to include KV pressure and measured perf while preserving deterministic tie-breakers. (.specs/00_llama-orch.md:179–184)
- Document first‑run guided provisioning (Arch pacman/AUR) and preflight diagnostics as a user flow. (.specs/00_home_profile.md:48–52; .specs/00_llama-orch.md:131–137)
- Provide canonical SSE heartbeat/micro-batch guidance and client compatibility notes. (.specs/20-orchestratord.md:32–37,100–106)

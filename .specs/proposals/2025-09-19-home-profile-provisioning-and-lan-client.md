# Proposal: Home‑Profile Engine Provisioning Defaults, Container Provider, and LAN Client Mode

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## 0) Motivation

For the home profile, optimize for a single powerful workstation (e.g., RTX 3090 + 3060) with a lightweight client (e.g., NUC) submitting tasks over LAN. Reduce operator friction by:
- Making engine provisioning automatic, deterministic, and reproducible.
- Preferring source build for llama.cpp and container-first for vLLM/TGI/Triton.
- Preferring rootless Podman for security; allow Docker as a fallback.
- Enforcing minimal auth and sane LAN defaults without introducing cluster complexity (no Kubernetes).

This proposal codifies defaults hinted across specs (`/.specs/00_llama-orch.md §2.11/§2.12`, `/.specs/60-config-schema.md`, engine/adapters specs) and adds a concrete container provider path and LAN client seams.

## 1) Scope

In scope:
- Home-profile defaults for engine provisioning modes per engine.
- Container provider semantics (Podman preferred; Docker fallback) in `engine-provisioner`.
- LAN client mode: non-loopback bind, minimal auth seam, and identity breadcrumbs.
- Robustness guidance for mixed GPUs (3090 + 3060) and per-GPU pools.

Out of scope:
- Kubernetes or multi-node orchestration.
- Changes to the `WorkerAdapter` trait.

## 2) Normative Requirements (RFC‑2119)

IDs use ORCH‑38xx (home-profile provisioning & LAN mode).

### Provisioning Defaults
- [ORCH‑3800] Home-profile MUST default engines as follows:
  - `llamacpp` → provisioning.mode = `source` (GPU), built locally with normalized flags.
  - `vllm`, `tgi`, `triton` → provisioning.mode = `container` by default.
- [ORCH‑3801] Container provider MUST prefer rootless `podman`; when unavailable, it MUST fallback to `docker` with equivalent GPU runtime semantics.
- [ORCH‑3802] Container images MUST be pulled by digest when available; tags MAY be used only with an advisory warning. Engine identity MUST capture `{image, tag?, digest}`.
- [ORCH‑3803] GPU-only policy is REQUIRED. Provisioning MUST fail fast if NVIDIA GPUs / CUDA runtime are unavailable (no CPU fallback for inference).
- [ORCH‑3804] Preflight MUST validate: presence of `podman` or `docker`, NVIDIA container runtime/toolkit, device visibility, required ports, and local cache paths. On failure, it MUST return actionable diagnostics.
- [ORCH‑3805] Provisioning MUST produce `PreparedEngine { bin_or_entry, args, env, ports, workdir, engine_version, engine_digest? }` and SHOULD write an EngineEntry to the engine catalog (`/.specs/56-engine-catalog.md`).

### LAN Client Mode (NUC → Workstation)
- [ORCH‑3810] When the server binds to non-loopback, `orchestratord` MUST require `Authorization: Bearer <token>` (Minimal Auth seam) as per `/.specs/11_min_auth_hooks.md` reference in `/.specs/00_llama-orch.md §2.7`.
- [ORCH‑3811] Logs MUST include identity breadcrumbs without exposing secrets (e.g., `identity=token:<fp6>`), and MUST redact tokens in all paths.
- [ORCH‑3812] `GET /v1/capabilities` MUST include `engine_version` per pool; static engine lists SHOULD be replaced with dynamic discovery over time.

### Mixed-GPU & Pools
- [ORCH‑3820] Operators SHOULD define one pool per GPU (explicit device masks). Placement MUST respect masks; spillover MUST NOT occur (`/.specs/10-orchestrator-core.md §2`).
- [ORCH‑3821] Readiness MUST remain a three-gate: model staged, engine ensured, health check OK (`pool-managerd` [ORCH-3504]).

## 3) Design Overview

- llama.cpp: Source mode with CUDA (no CPU inference). Cache under `~/.cache/llama-orch/llamacpp`. Flags normalized; version = commit SHA + build flags.
- vLLM/TGI/Triton: Container-first. Rootless Podman preferred; Docker fallback. Pull by digest; mount model cache; map device masks and ports; capture image digest/version.
- Engine catalog: Store `EngineEntry` for reproducibility; placement/logs include engine identity fields.
- LAN client: Bind to LAN; require Bearer token; expose capabilities with engine_version; SSE streams as today.

## 4) Changes by Crate

- `provisioners/engine-provisioner`:
  - Add Container provider with Podman→Docker fallback, NVIDIA toolkit preflight, digest pinning.
  - Return `PreparedEngine` with identity metadata. Write `EngineEntry` on success.
- `pool-managerd`:
  - Supervise containers as well as binaries. Prefer rootless Podman; security hardening (seccomp/apparmor if available). Keep readiness gates.
- `orchestratord`:
  - Minimal Auth seam on non-loopback; extend `/v1/capabilities` to include `engine_version` per pool.
- `contracts/config-schema`:
  - Ensure container fields (`image`, `tag`, future `digest`) present; examples for home-profile included.

## 5) Migration Plan

Phase A — Defaults & Docs
- Document home-profile defaults in `/.specs/00_llama-orch.md §2.12` (container-first for vLLM/TGI/Triton; llama.cpp source).
- Add configuration examples for: two pools (3090, 3060) and LAN binding + Bearer token.

Phase B — Container Provider
- Implement container provider in `engine-provisioner` with Podman preflight and Docker fallback, GPU runtime checks, digest pinning.
- Update `pool-managerd` supervision to start/stop containers (backoff, health checks).

Phase C — Catalog & Capabilities
- Write `EngineEntry` on prepare; extend `/v1/capabilities` with per-pool `engine_version`.

Phase D — Hardening & Tests
- Minimal Auth Hooks wiring; redaction helpers.
- Unit/integration tests for container provider preflight, digest pinning, and GPU runtime mapping.

## 6) CI & Testing

- Unit: container provider preflight (toolkit present, GPU visible), digest pinning, `PreparedEngine` shaping.
- Integration: `pool-managerd` readiness flips only after container health; restart/backoff loops with simulated failures.
- Provider verify: `/v1/capabilities` includes engine versions; control/data-plane unchanged.
- Determinism suite: capture engine version/digest in proof bundles.

## 7) Risks & Mitigations

- Container GPU runtime compatibility → Preflight NVIDIA toolkit; provide clear remediation.
- Security drift in Docker fallback → Prefer rootless Podman; document Docker hardening (userns, seccomp) when used.
- Image tag drift → Require (or warn strongly without) digest pinning; record engine identity in catalog and logs.

## 8) Acceptance Criteria

- Home-profile defaults documented and examples added (two pools + LAN binding + Bearer).
- Container provider available with Podman preferred and Docker fallback; GPU runtime validated.
- `/v1/capabilities` includes `engine_version` per pool; proof bundles include engine identity.
- Determinism and provider verify tests green; no change to public OpenAPI shapes beyond capabilities enrichment.

## 9) Mapping to Repo Reality (Anchors)

- `/.specs/00_llama-orch.md` — §2.11 Model Selection & Auto‑Fetch; §2.12 Engine Provisioning & Preflight.
- `/.specs/60-config-schema.md` — provisioning modes and container fields.
- `provisioners/engine-provisioner/.specs/00_engine_provisioner.md`, `.specs/10_contracts.md` — provider contracts.
- `pool-managerd/CHECKLIST.md` — Podman preferred, Docker fallback, supervision & security.
- `/.specs/56-engine-catalog.md` — EngineEntry and reproducibility.
- `/.specs/20-orchestratord.md` — `/v1/capabilities` and SSE framing.

## 10) Refinement Opportunities

- Container image selection helpers and vetted digest registry for common engines.
- systemd user units for orchestrator + managerd; graceful reload/drain flows.
- Arch/CachyOS UX: optional pacman/AUR installs of required tooling under explicit policy gates.
- Dynamic `/v1/capabilities` sourced from adapter-host cache instead of static lists.
- Optional local container registry cache and image verification hooks.

## 11) New Crates

To keep responsibilities clean across the workspace and enable reuse, introduce the following crates:

- container-runtime — new root crate
  - Purpose: provide a minimal abstraction over container engines with a rootless-first posture.
  - Responsibilities:
    - Detect and prefer Podman, fallback to Docker with equivalent GPU runtime semantics.
    - Preflight NVIDIA container runtime/toolkit and verify device visibility for requested masks.
    - Pull by digest (verify digest when available), create/run/stop containers, map ports, and mount model caches.
    - Expose a small API surface (e.g., `detect() -> Runtime`, `pull(image, digest?)`, `run(ContainerSpec) -> ContainerHandle`, `stop(ContainerHandle)`), and stream logs with redaction.
  - Consumers: `provisioners/engine-provisioner` (prepare) and `pool-managerd` (supervise).
  - Notes: keep dependencies light; feature-gate specific backends (e.g., `backend-podman`, `backend-docker`).

- engine catalog module (in `catalog-core`)
  - Purpose: implement the Engine Catalog described in `/.specs/56-engine-catalog.md` alongside the existing model catalog to avoid crate asymmetry for the home-profile.
  - Responsibilities:
    - Define `EngineEntry { id, engine, version, build_ref, digest?, build_flags?, artifacts[], created_ms }` and add a `EngineCatalogStore` trait and FS-backed `FsEngineCatalog` under `catalog-core::engine`.
    - Keep engine entries in a separate index file (e.g., `engines.json`) while sharing atomic read/write helpers and directory layout with the model catalog.
  - Producers/Consumers: written by `engine-provisioner` upon successful prepare; read by `pool-managerd` for readiness/version; referenced by `orchestratord` logs and proofs.
  - Notes: preserves a single catalog crate with two indices (models, engines) per `/.specs/25-catalog-core.md` guidance.

Acceptance impact
- `engine-provisioner` adopts `container-runtime` for container mode and writes engine entries via `catalog-core::engine` (when applicable).
- `pool-managerd` uses `container-runtime` for run/stop and health probes; logs include `engine_version`/`engine_digest`.
- CI adds unit/integration tests for `container-runtime` and for the new `catalog-core::engine` module; metrics/logging align with `.specs/metrics/otel-prom.md`.

## 12) Checklist Updates (for CHECKLIST.md and SPEC_CHECKLIST.md)

Add the following items to align the checklists with this proposal.

### CHECKLIST.md
- Workspace & Build System
  - [ ] Add new crate to workspace members: `container-runtime` (root-level crate)
  - [ ] Update `xtask`/CI to include `container-runtime` in fmt/clippy/tests

- New Library Crates (Scaffolding)
  - [ ] `container-runtime/` (lib): runtime detection (prefer Podman, fallback Docker), NVIDIA toolkit preflight, pull-by-digest, run/stop with device masks and port mapping, logs with redaction, feature-gated backends (`backend-podman`, `backend-docker`)

- Orchestratord
  - [ ] `/v1/capabilities` includes `engine_version` per pool (capabilities enrichment)
  - [ ] Minimal Auth: enforce Bearer on non-loopback bind; identity breadcrumbs in logs (AUTH seam)

- Pool-managerd
  - [ ] Use `container-runtime` to start/stop engines in container mode; prefer rootless Podman, fallback Docker
  - [ ] Health checks cover container-based engines before flipping `ready=true`

- Provisioners → Engine-provisioner
  - [ ] Implement container provider using `container-runtime` with GPU runtime preflight (NVIDIA toolkit) and digest pinning
  - [ ] `PreparedEngine` includes identity metadata (engine_version, digest, image) and `engine_catalog_id` when available

- Catalog-core
  - [ ] Add `engine` module: separate `engines.json` index; implement `EngineCatalogStore` and `FsEngineCatalog` with atomic read/write
  - [ ] Tests for EngineEntry round-trip, incompatible schema rejection, and atomicity (crash-sim)

- CI & Tooling
  - [ ] Add tests covering `container-runtime` preflight branches (Podman present/absent, Docker fallback, toolkit present/absent)

- Proof Bundles & Artifacts
  - [ ] Include container preflight logs/plan, `PreparedEngine` summary, EngineEntry snapshot, and `/v1/capabilities` with engine versions

### SPEC_CHECKLIST.md
- Approvals
  - [ ] Set `/.specs/proposals/2025-09-19-home-profile-provisioning-and-lan-client.md` to Accepted

- Root `.specs/` edits
  - [ ] `/.specs/00_llama-orch.md`: in §2.12, document home-profile defaults (llama.cpp=source; vLLM/TGI/Triton=container-first), GPU-only, preflight/policy gates; in §2.7 reference Minimal Auth Hooks
  - [ ] `/.specs/20-orchestratord.md`: make Bearer seam normative for non-loopback, add capabilities enrichment (`engine_version` per pool), and keep HTTP/2 SSE preference
  - [ ] `/.specs/25-catalog-core.md`: add Engine Catalog module (separate `engines.json` index) alongside models; promote atomic reads/writes and helper reuse
  - [ ] `/.specs/50-engine-provisioner.md`: add container provider requirements (Podman preferred, Docker fallback), digest pinning, NVIDIA toolkit preflight, `PreparedEngine` fields including `engine_catalog_id`, GPU-only
  - [ ] `/.specs/30-pool-managerd.md`: container supervision norms (rootless Podman), readiness gates with container health, registration with Bearer (when configured)
  - [ ] `/.specs/60-config-schema.md`: add `provisioning.container.digest` field and examples; auth fields (`AUTH_TOKEN`, `AUTH_OPTIONAL`, `TRUST_PROXY_AUTH`) already listed—ensure examples include LAN bind

- New spec files to create
  - [ ] `container-runtime/.specs/00_container_runtime.md` — API (detect/pull/run/stop), backends, NVIDIA toolkit preflight, redaction policy

- Verification & proof bundles
  - [ ] Update `.docs/testing/` to require container preflight logs and capabilities snapshots (with engine_version) in proof bundles

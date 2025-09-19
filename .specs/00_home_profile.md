# Home Profile Overlay — Single-Host (ORCH-HOME)

This overlay refines the core orchestrator spec for the home lab scenario. It does not compare to any “enterprise” edition; it simply states what the home profile requires on top of `.specs/00_llama-orch.md`.

---

## H1. Deployment Envelope

- [HME-001] Orchestrator, pool manager, adapters, and artifact storage MUST run on the same workstation.
- [HME-002] Remote access MUST default to loopback with opt-in LAN exposure guarded by firewall or SSH tunnel.
- [HME-003] Configuration MUST live in a single YAML/TOML file plus environment variables for sensitive values (API token paths, optional tunnels).
- [HME-004] Minimal Auth seam: when exposing beyond loopback, configure `AUTH_TOKEN` and related keys per `/.specs/11_min_auth_hooks.md`. `AUTH_OPTIONAL=true` only affects loopback requests; non-loopback sources MUST present Bearer token.

## H2. GPUs & Scheduling

- [HME-010] Mixed GPUs (24 GB + 12 GB reference) MUST participate simultaneously; scheduling MUST prefer the GPU with the most free VRAM.
- [HME-011] Pools MAY pin specific device masks but defaults MUST use all available devices.
- [HME-012] Concurrency hints returned via capability discovery MUST reflect the mixed-GPU capacity used in the reference environment.
- [HME-013] GPU-only policy: inference MUST run on NVIDIA GPUs only. Components MUST NOT use CPU for inference; when GPU/CUDA is unavailable, fail fast with actionable diagnostics.

## H3. CLI Integration

- [HME-020] The CLI MUST obtain queue metadata (`queue_position`, `predicted_start_ms`, budgets) for every accepted task.
- [HME-021] SSE `metrics` frames MUST include at least `queue_depth` and `on_time_probability` so the CLI can coordinate multiple agents.
- [HME-022] Artifact registry MUST store plan snapshots, diffs, traces locally and return download URLs reachable from the developer box (HTTP over tunnel or LAN).

## H4. Developer Experience

- [HME-030] Logs MUST be tail-friendly (JSON Lines) and group by `X-Correlation-Id`.
- [HME-031] Config reloads MUST complete in under 10 seconds for catalog swaps on the reference workstation.
- [HME-032] Errors presented to the CLI MUST include actionable messages (e.g., `QUEUE_FULL_DROP_LRU`, `POOL_UNAVAILABLE`) with retry guidance.

## H5. Validation Gates

- [HME-040] Every release MUST pass the reference environment smoke test described in `.docs/HOME_PROFILE_TARGET.md`.
- [HME-041] Determinism suite MUST run on both GPUs concurrently and report latency statistics per device.
- [HME-042] BDD features MUST cover catalog flows, artifact uploads, session eviction, cancel, and backpressure.

## H6. Nice-to-Have (Optional)

- [HME-050] Simple web status page MAY expose queue depth, GPU utilisation, and recent tasks for the workstation operator.
- [HME-051] Optional backup/restore scripts SHOULD be documented for artifact and catalog directories.

---

## Refinement Opportunities

- Add a first‑run guided setup that checks GPU drivers, CUDA, and provisioner prerequisites with copy‑pasteable fixes (Arch/CachyOS via pacman/AUR).
- Provide a one‑command `orchctl up` path that provisions llama.cpp from source and fetches a small model for a 60‑second smoke.
- Offer an optional lightweight web panel that surfaces `queue_depth`, `on_time_probability`, and recent tasks with cancel.
- Document LAN exposure patterns (SSH tunnel, reverse proxy with auth) tailored to the home profile.

---

Requirement IDs prefixed `HME-` are local to the home overlay and complement the ORCH-3xxx series. Update `requirements/00_home_profile.yaml` after modifying this document.

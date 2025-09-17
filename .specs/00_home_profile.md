# Home Profile Overlay — Single-Host (ORCH-HOME)

This overlay refines the core orchestrator spec for the home lab scenario. It does not compare to any “enterprise” edition; it simply states what the home profile requires on top of `.specs/00_llama-orch.md`.

---

## H1. Deployment Envelope

- [HME-001] Orchestrator, pool manager, adapters, and artifact storage MUST run on the same workstation.
- [HME-002] Remote access MUST default to loopback with opt-in LAN exposure guarded by firewall or SSH tunnel.
- [HME-003] Configuration MUST live in a single YAML/TOML file plus environment variables for sensitive values (API token paths, optional tunnels).

## H2. GPUs & Scheduling

- [HME-010] Mixed GPUs (24 GB + 12 GB reference) MUST participate simultaneously; scheduling MUST prefer the GPU with the most free VRAM.
- [HME-011] Pools MAY pin specific device masks but defaults MUST use all available devices.
- [HME-012] Concurrency hints returned via capability discovery MUST reflect the mixed-GPU capacity used in the reference environment.

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

Requirement IDs prefixed `HME-` are local to the home overlay and complement the ORCH-3xxx series. Update `requirements/00_home_profile.yaml` after modifying this document.

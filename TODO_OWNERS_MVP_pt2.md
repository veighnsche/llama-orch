# TODO — MVP Ownership Split (Provisioners, pt2)

Status: Active
Date: 2025-09-30
Scope: Provisioners are REQUIRED for MVP UX; Haiku must pass end-to-end with automatic engine+model management.

---

## MVP Requirement (UX)

- The program MUST provision and manage engines (e.g., llama.cpp) and models automatically.
- No manual steps for users beyond initial config. Passing Haiku requires real token streaming (no fallback SSE).

---

## Owner C — Engine Provisioner (libs/provisioners/engine-provisioner/)

- [x] Start/stop lifecycle for `llama-server` (llama.cpp), capturing PID and logs
  - Implemented in `providers/llamacpp.rs`: spawn with stdout/stderr to `.cache/llama-orch/run/llamacpp-<pool>.log`, PID at `.cache/llama-orch/run/<pool>.pid`.
  - Public `stop_pool()` helper added in `libs/provisioners/engine-provisioner/src/lib.rs`.
- [x] Deterministic flags for tests: `--parallel 1`, `--no-cont-batching`, `--no-webui`, `--metrics`
  - Enforced via `ensure_flag*()` helpers; also normalizes legacy `--ngl/-ngl/--gpu-layers`.
- [x] Bind address/port management; emit discovered URL (e.g., `http://127.0.0.1:<port>`) to integration channel
  - Binds `127.0.0.1:<port>`; chooses free port with collision fallback.
- [x] Readiness/health waits (poll `/health` -> 200 when ready, 503 while loading)
  - Simple HTTP probe loop with transient handling.
- [x] Metrics exposure sanity (`/metrics` reachable when `--metrics` enabled)
  - Non-fatal warning if unreachable during early boot.
- [x] Integration: write endpoint URL to an orchestrator-consumable source
  - Writes `.runtime/engines/llamacpp.json` payload `{ engine, engine_version, provisioning_mode, url, pool_id, replica_id, model, flags }`.
- [x] Shutdown semantics for drain/reload; ensure prompt slot free (MVP)
  - Implemented: `stop_pool()` sends TERM, waits up to 5s, then KILL; prompt slot freeing via engine drain hook remains TODO post-MVP.
- [x] Tests: smoke (spawn → health), restart on crash, port collision handling
  - Smoke test present (`tests/llamacpp_smoke.rs`, ignored behind `LLAMA_ORCH_SMOKE=1`).
  - Port collision unit test added.
  - Restart-on-crash placeholder test added (ignored) with TODO.

### Deliverables

- [x] Minimal config (YAML/JSON) → engine settings; CachyOS notes for system tooling (pacman/AUR) per repo policy
  - Uses `requirements/llamacpp-3090-source.yaml`; preflight optional pacman installs behind `allow_package_installs`.
- [x] CLI entry or library API callable from orchestrator bootstrap path
  - New bin: `engine-provisioner` (`src/bin/engine-provisioner.rs`). Usage:
    - `cargo run -p provisioners-engine-provisioner --bin engine-provisioner -- --config <path> [--pool <id>]`

---

## Owner D — Model Provisioner (libs/provisioners/model-provisioner/)

- [x] Resolve model artifact (local cache or remote) and ensure available path
  - Implemented in `libs/provisioners/model-provisioner/src/lib.rs` via `ModelProvisioner::ensure_present_str()` and `ensure_present()` (file fast-path; minimal `hf:` shell-out fallback)
- [x] Optional verify/signature policy hook per repo trust policy (strict mode placeholder OK)
  - Implemented `sha256` verification when `expected_digest` is provided; strict gating supported
- [x] Emit model metadata (id, path, size, ctx_max if known) for engine-provisioner
  - Implemented `ModelMetadata { id, path, size_bytes, ctx_max: None }` and handoff JSON writer
- [x] Choose a small, deterministic Haiku model profile and document it
  - Documented TinyLlama 1.1B Chat q4_k_m GGUF in `libs/provisioners/model-provisioner/README.md`
- [x] Cache and eviction policy skeleton; record provenance
  - Added TODO skeleton for eviction; provenance JSONL emitted to `.runtime/provenance/models.jsonl`
- [x] Tests: resolve-only happy path; missing model → clear error; optional signature gating
  - Added unit tests in `libs/provisioners/model-provisioner/src/lib.rs`

### Deliverables
- [x] Minimal config (YAML/JSON) describing model_ref → path resolution
  - Config schema in `.specs/55-model-provisioner.md`; loader in crate via `ModelProvisionerConfig`
- [x] Handshake file/object with engine-provisioner containing resolved model path
  - Handoff JSON writer implemented; recommended path `.runtime/engines/llamacpp.json`

---

## Orchestrator & Adapter Wiring (bin/orchestratord/, libs/adapter-host/, libs/worker-adapters/llamacpp-http/)

**OUT OF SCOPE FOR OWNER C & D**

- [ ] Replace env-based adapter binding with provisioner outputs (engine URL + model metadata)
- [ ] Gate/remove deterministic SSE fallback in `render_sse_for_task()` when `REQUIRE_REAL_LLAMA=1`
- [ ] Ensure `AdapterHost.bind()` uses pool/replica derived from provisioned context
- [ ] Update specs to reflect provisioner-first binding; retain dev-only env shim behind feature flag

---

## Acceptance Criteria

- [ ] `cargo test -p test-harness-e2e-haiku -- --nocapture` passes with automated provisioning
- [ ] Real token streaming (no hard-coded SSE). Event order: `started → token* → end` with indices
- [ ] Health/ready polling enforced before dispatch; cancel frees slot promptly
- [ ] README/specs updated (provisioners, wiring, determinism profile)

---

## References

- `libs/provisioners/engine-provisioner/`
- `libs/provisioners/model-provisioner/`
- `bin/orchestratord/src/app/bootstrap.rs`
- `bin/orchestratord/src/services/streaming.rs`
- `libs/worker-adapters/llamacpp-http/src/lib.rs`
- `.specs/40-worker-adapters-llamacpp-http.md`
- `bin/orchestratord/.specs/22_worker_adapters.md`

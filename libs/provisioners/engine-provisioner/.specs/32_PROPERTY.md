# engine-provisioner — Property Tests (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `provisioners/engine-provisioner/`

## 0) Scope

Property tests exercise plan determinism/idempotency, retry/backoff jitter bounds, and stability under randomized inputs (flags, modes, env). Tests MUST avoid network/disk writes; effects are simulated. GPU‑only stance is normative — CPU fallback MUST NOT be suggested by properties.

## 1) Properties (normative)

- [EP-PROP-3201] Plan Idempotency
  - For any `PoolConfig` with fixed inputs, `plan(cfg)` MUST yield an identical sequence of steps across runs. Hash(plan) is stable modulo explicitly unordered metadata (e.g., timestamps excluded).

- [EP-PROP-3202] Step Ordering Stability
  - For `provisioning.mode=source`, the subsequence `git_clone → cmake_configure → cmake_build → install` MUST preserve order regardless of flag permutations; normalization MUST canonicalize flags before ordering.

- [EP-PROP-3203] Flag Normalization Canonicalization
  - For any alias set mapping to the same semantic flag (e.g., `{--ctx, --context, -c}`), normalization MUST produce a canonical key+value; duplicate keys MUST be de‑duplicated deterministically (last‑write or priority table).

- [EP-PROP-3204] Retry/Backoff Jitter Bounds (if retries exist in ensure/spawn helpers)
  - Given a base delay `d` and cap `D`, generated delays MUST satisfy `d ≤ delay_n ≤ D` and MUST include randomness (variance > 0) while remaining within bounds. Long‑run mean MUST stay within ±20% of the configured target for exponential backoff with jitter.

- [EP-PROP-3205] Policy Gates — Arch/CachyOS Package Installs
  - For any config with `allow_package_installs=false`, the Plan MUST NOT include package install steps; for `true`, install steps MUST be present only for `mode=package` and MUST reflect pacman/AUR patterns.

- [EP-PROP-3206] CUDA Preflight Monotonicity
  - For environments sampled with `cuda=present|absent`, `ensure(cfg)` MUST be monotonic with respect to the predicate: if `absent` fails fast, then flipping to `present` MUST not fail for the same reason and MUST proceed to spawn (subject to other gates).

- [EP-PROP-3207] PreparedEngine Determinism
  - With fixed inputs, `PreparedEngine` fields `(engine, version, build_ref, digest?, flags, mode, binary_path)` MUST be stable across runs. Paths may include temp roots but MUST be normalized for equality or exposed via content digests.

## 2) Strategy & Generators

- Randomize `PoolConfig` across modes with constraints:
  - `mode ∈ {source, container, package, binary}` (external excluded here)
  - Flags come from alias tables with random duplication and ordering
  - Env toggles: `cuda_present ∈ {true,false}`, `allow_package_installs ∈ {true,false}`
- Shrinkers MUST produce minimal failing configs (e.g., smallest alias set reproducing normalization drift).

## 3) Traceability

- Root specs: `/.specs/50-engine-provisioner.md`, `/.specs/60-config-schema.md`
- Code: `provisioners/engine-provisioner/src/`
- Unit matrices: `./31_UNIT.md`

## Refinement Opportunities

- Add profile‑aware generators for Arch/CachyOS package names and AUR helper decisions to reflect real user environments.
- Model non‑deterministic toolchains (e.g., non‑pinned submodules) and assert plans record pinned refs/digests to restore determinism.

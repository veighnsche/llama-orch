# engine-provisioner — Unit Tests (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `provisioners/engine-provisioner/`

## 0) Scope

Unit tests validate plan construction, flag normalization, CUDA preflight branches, and Arch/CachyOS package install policy when explicitly allowed by config. Tests MUST NOT hit the network; all external effects are mocked.

## 1) Test Matrix (normative)

- [EP-UNIT-3101] Plan Construction — llama.cpp (source)
  - GIVEN `provisioning.mode=source` with `git.repo`, `git.ref`, and CMake flags
  - WHEN `plan(&PoolConfig)` is invoked
  - THEN the Plan MUST include steps in order: `git_clone` (with submodules opt), `cmake_configure` (flags normalized), `cmake_build`, `install`
  - AND flags normalization MUST deduplicate and canonicalize known switches (table-driven; see below)

- [EP-UNIT-3102] Plan Construction — container
  - GIVEN `provisioning.mode=container` with image+tag
  - THEN Plan MUST contain `pull_image` and skip source build steps

- [EP-UNIT-3103] Plan Construction — package (Arch/CachyOS)
  - GIVEN `allow_package_installs=true` and `provisioning.mode=package`
  - THEN Plan MUST include `pacman -S --noconfirm <pkg>` (or AUR helper stub) behind a policy gate
  - AND on `allow_package_installs=false`, Plan MUST produce an advisory failure step (`policy_violation`) and no install step

- [EP-UNIT-3104] CUDA Preflight Detection
  - GIVEN a mocked environment with CUDA present/absent
  - WHEN `ensure()` runs
  - THEN absence MUST yield a fail‑fast diagnostic (GPU required; no CPU fallback) and the pool remains Unready
  - AND presence continues to flag normalization/spawn

- [EP-UNIT-3105] Flag Normalization (table-driven)
  - GIVEN inputs with duplicate/alias flags (e.g., `--ctx`, `--context`, `-c`)
  - THEN normalization MUST produce a canonical set and stable order

- [EP-UNIT-3106] PreparedEngine Metadata
  - WHEN a plan completes
  - THEN `PreparedEngine` MUST capture `engine`, `version`, `build_ref`, `digest?`, `flags`, `mode`, `binary_path`

## 2) Table — Flag Normalization Maps (illustrative)

Examples (engines may extend):
- `--n-gpu-layers` aliases: `--ngl`, `--gpu-layers`
- `--ctx` aliases: `--context`, `-c`
- Stable ordering: deterministic lexicographic by flag name for reproducibility

## 3) Traceability

- Root spec: `/.specs/50-engine-provisioner.md` (§Key Flows, Observability, GPU‑only policy)
- Config schema: `/.specs/60-config-schema.md` (§provisioning modes, Arch policy)
- Code: `provisioners/engine-provisioner/src/`

## Refinement Opportunities

- Add more alias coverage per engine and validate against real binaries’ help output.
- Validate `PreparedEngine` checksum/digest derivation when available (container image digest, binary sha256).

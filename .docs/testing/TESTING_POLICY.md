# Testing and CI Policy

This document defines the non-negotiable rules for tests, CI, harness behavior, and product alignment in the `oxidizr-arch` project.

## Core Principle

- The product must perform all state mutations. Tests must not mask, pre-create, or repair product-managed artifacts.

## SKIP Policy

- Zero SKIPs are authorized. Any SKIP is a failure.
- CI treatment: Any SKIP fails the run and must be fixed. Do not mask nondeterminism with SKIPs.

Note: Locales are baked into Docker images (see `test-orch/docker/Dockerfile`). Locale-related failures are infra bugs and must be fixed in image builds.

## Allowed SKIPs Table (Docker-only constraints)

None. All suites must run; any SKIP is a failure to be fixed.

## CI and Matrix Policy

- Matrix runs across supported Arch-family distros are the default for CI via the host orchestrator.
- Any SKIP in any matrix job fails the run. Non-matrix runs must not silently appear green by skipping assertions—if the scenario cannot be executed, fail with a clear reason.

## Harness Policy (Docker and other runners)

- Do not mutate `/usr/bin/*` or related product-managed paths in the harness.
- Do not install or rely on BusyBox or similar toolsets to perform file operations for tests.
- Do not pre-create, repair, or inject applet symlinks before or around `oxidizr-arch enable/disable`.
- Do not add shell cache workarounds such as `hash -r` to make tests pass. If resolution or sequencing is incorrect, fix the product or run assertions in a fresh process.
- Docker entrypoint must follow the minimal flow: build binary → `enable` → assertions → `disable` → assertions.

## Product–Policy Alignment

- Goal: support switching across Arch-family distros (`arch, manjaro, cachyos, endeavouros`) with zero SKIPs in matrix runs.
- Do not distro-gate the Arch-family to make tests pass; run the same assertions across the family. Gate only OSes we explicitly do not support or have never tested.
- Registry defaults must not hard-block derivatives. If providers are available (via pacman/AUR), switching proceeds; otherwise fail loudly with a clear reason (no SKIPs).
- `SudoRsExperiment::check_compatible()` should allow derivatives when the package is installed/available; otherwise error (not skip).

## Packaging & AUR Semantics

- Assume an AUR helper (e.g., `paru`, `yay`) is available on systems that require AUR packages. In CI, the container-runner ensures a helper exists; on user systems, users must install one.
- CLI overrides (`--package`, `--bin-dir`, `--unified-binary`) must override registry defaults on all distros so tests/users can force switching deterministically.

## Registry and Probing

- Always attempt switching on supported Arch-family distros. Use `pacman` first; when packages are only in AUR, use the available helper. If installation fails, fail loudly with a clear reason (no SKIPs).
- Probing is for visibility and better messages (e.g., detect that `sudo-rs` is missing), not for gating or skipping.

## Infrastructure Policy (Containers/Images)

- Tests must be robust to parallel execution across distros. Flaky suites must be deflaked or explicitly serialized in CI configuration; do not use SKIPs.
- Infra gaps discovered by tests must either be fixed in images or cause a hard failure with clear remediation guidance.

## Documentation and Accountability

- Any temporary workaround must be accompanied by a blocking task (issue) with an owner and removal criteria.
- Comments that label a workaround as “temporary/stupid” without removing it are forbidden—remove the workaround or file the blocking task immediately.

## References

- Product switching logic: `src/experiments/uutils/` and `src/utils/worker/` (e.g., `replace_file_with_symlink()`, `restore_file()`).
- Docker harness: `test-orch/docker/entrypoint.sh`.
- Container runner assertions: `test-orch/container-runner/assertions/`.
- YAML suites: `tests/*/task.yaml` (Spread/LXD via `spread.yaml`).

## Enforcement

- CI enforces these rules. Violations will cause failures. Do not add masking logic to bypass failures—fix the root cause.

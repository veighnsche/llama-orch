# 00_container_runtime.md — Container Runtime Abstraction (Home-Profile)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-20

## Purpose & Scope

Provide a minimal, dependency-light abstraction over container engines for the home-profile:
- Detect and prefer Podman (rootless) with Docker as a fallback.
- Preflight NVIDIA container runtime/toolkit and verify device visibility for requested masks.
- Pull by digest (verify when available), create/run/stop containers, map ports, and mount model caches.
- Stream logs with redaction helpers.

This crate is consumed by `provisioners/engine-provisioner` (prepare) and `pool-managerd` (supervise). It keeps responsibilities tight and avoids introducing any network hops or heavy dependencies.

## API Surface (conceptual)

- `detect() -> Runtime` — Detects Podman first, falls back to Docker with equivalent GPU runtime semantics.
- `pull(image: &str, digest: Option<&str>) -> Result<PulledImage>` — Pull by digest when provided; warn if only a tag is given.
- `run(spec: &ContainerSpec) -> Result<ContainerHandle>` — Run container with device masks, port mappings, env, mounts.
- `stop(handle: &ContainerHandle) -> Result<()>` — Stop container gracefully; force after a bounded backoff.
- `preflight() -> Result<PreflightReport>` — Validate toolkit, device visibility, and required binaries; produce actionable diagnostics.
- `logs(handle: &ContainerHandle) -> impl Iterator<Item = RedactedLine>` — Bounded streaming view with redaction.

Note: This is an internal crate contract; no public HTTP surfaces are introduced.

## Normative Requirements (excerpt)

- Prefer Podman rootless; fallback to Docker with equivalent GPU runtime semantics.
- Fail fast when NVIDIA GPUs/toolkit are unavailable (GPU is required; no CPU inference).
- Pull by digest when available and capture `{image, tag?, digest}` in engine identity.
- Provide device-mask aware run semantics compatible with pool `device_mask` values.
- Emit human-readable narration messages at key actions (preflight, pull, run, stop) with secrets redacted.

## Testing Guidance

- Unit: preflight branches (Podman present/absent, Docker fallback, toolkit present/absent), digest pinning, device visibility mapping.
- Integration (lightweight): simulate run/stop with mocked executors; verify port/device mappings and log redaction.

## Refinement Opportunities

- Add optional systemd user units helpers for graceful run/stop on reload.
- Provide a local image cache advisory/verification hook.
- Extend preflight with kernel/driver compatibility hints for common GPUs.
- Pluggable backends via features (e.g., `backend-podman`, `backend-docker`), keeping default footprint minimal.

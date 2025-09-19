# Worker Adapter SPEC — NVIDIA Triton/TensorRT‑LLM (v1.0)

Status: Stable (draft)
Applies to: `worker-adapters/triton/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-ADAPT-5xxx`.

## 1) API Mapping

- [OC-ADAPT-5060] Adapter MUST support Triton HTTP/GRPC infer, health/metadata/stats, and optional OpenAI‑compat frontends internally.

## 1A) References & Cross-Cutting

- This adapter SHOULD use `worker-adapters/http-util` for HTTP client setup, retries (capped + jitter), HTTP/2 keep‑alive, and header redaction where HTTP applies; for gRPC, analogous retry/timeouts SHOULD be applied.
- Integration with orchestrator uses the in‑process facade described in `adapter-host/.specs/00_adapter_host.md`.
- Streaming MUST preserve `started → token* → end` ordering per `/.specs/35-worker-adapters.md` when streaming is provided; apply redaction to logs consistently.

## 2) Determinism & Version Capture

- [OC-ADAPT-5070] Adapter MUST report engine_version/trtllm_version where applicable.

## 3) Traceability

- Code: [worker-adapters/triton/src/lib.rs](../worker-adapters/triton/src/lib.rs)

## Refinement Opportunities

- Document mapping of Triton model repository layouts (local dir, `s3://`, `oci://`) to `model_ref` schemes and adapter behavior.
- Capture `trtllm_version` and engine container image digest for provenance when running in containers.
- Provide adapter‑level examples for gRPC vs HTTP infer, including streaming tokens (if applicable) and cancel semantics.

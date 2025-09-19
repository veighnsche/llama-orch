# Worker Adapter SPEC — HF TGI HTTP (v1.0)

Status: Stable (draft)
Applies to: `worker-adapters/tgi-http/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-ADAPT-5xxx`.

## 1) API Mapping

- [OC-ADAPT-5040] Adapter MUST implement TGI custom endpoints (`/generate`, `/info`, `/health`) and optionally OpenAI‑compat internally; not public.

## 1A) References & Cross-Cutting

- This adapter SHOULD use the shared HTTP utilities in `worker-adapters/http-util` to ensure consistent timeouts, capped+jittered retries, HTTP/2 keep‑alive, and header redaction.
- Integration with orchestrator uses the in‑process facade described in `adapter-host/.specs/00_adapter_host.md`.
- Streaming MUST preserve `started → token* → end` ordering per `/.specs/35-worker-adapters.md`; apply redaction to logs consistently.

## 2) Determinism & Version Capture

- [OC-ADAPT-5050] Adapter MUST capture engine_version and model info as available.

## 3) Traceability

- Code: [worker-adapters/tgi-http/src/lib.rs](../worker-adapters/tgi-http/src/lib.rs)

## Refinement Opportunities

- Clarify mapping between TGI sampling parameters and OrchQueue v1 to avoid divergence (e.g., top‑k, top‑p differences).
- Capture engine_version and container image digests for better provenance (when using containers).
- Provide adapter‑level examples for `/generate` streaming and cancel, including SSE payloads.

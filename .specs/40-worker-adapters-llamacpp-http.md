# Worker Adapter SPEC — llama.cpp HTTP (v1.0)

Status: Stable (draft)
Applies to: `worker-adapters/llamacpp-http/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-ADAPT-5xxx`.

## 1) API Mapping

- [OC-ADAPT-5001] Adapter MUST implement health, properties (slots/commit), completion (SSE), cancel, metrics scrape for llama.cpp native API.
- [OC-ADAPT-5002] When using OpenAI‑compatible endpoints internally, they MUST NOT be exposed publicly.

## 2) Determinism & Version Capture

- [OC-ADAPT-5010] Adapter MUST normalize detokenization templates and sampler profiles for determinism within a replica set.
- [OC-ADAPT-5011] Adapter MUST capture and report engine_version and model_digest.

## 3) Traceability

- Code: [worker-adapters/llamacpp-http/src/lib.rs](../worker-adapters/llamacpp-http/src/lib.rs)

## Refinement Opportunities

- Normalize prompt/response templates across common llama.cpp models to reduce detokenization drift.
- Capture and surface build flags (e.g., `LLAMA_CUBLAS`, `LLAMA_VULKAN`) as part of `engine_version` metadata for better determinism audits.
- Provide adapter‑level `x-examples` (in OpenAPI client docs) for typical llama.cpp interactions and SSE token streams.

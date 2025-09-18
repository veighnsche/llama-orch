# Worker Adapter SPEC — vLLM HTTP (v1.0)

Status: Stable (draft)
Applies to: `worker-adapters/vllm-http/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-ADAPT-5xxx`.

## 1) API Mapping

- [OC-ADAPT-5020] Adapter MUST implement health/properties/completion/cancel/metrics against vLLM OpenAI‑compatible server.
- [OC-ADAPT-5021] OpenAI‑compatible endpoints MUST remain internal; public surface is OrchQueue v1.

## 2) Determinism & Version Capture

- [OC-ADAPT-5030] Adapter MUST report engine_version and sampler/profile versions as applicable.

## 3) Traceability

- Code: [worker-adapters/vllm-http/src/lib.rs](../worker-adapters/vllm-http/src/lib.rs)

## Refinement Opportunities

- Clarify handling of OpenAI-compatible sampling params vs. OrchQueue v1 schema to avoid silent drift.
- Capture engine build/runtime flags (e.g., CUDA version) in `engine_version` metadata when available.
- Provide adapter‑level examples for streaming tokens and cancel semantics under vLLM server quirks.

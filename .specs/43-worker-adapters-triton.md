# Worker Adapter SPEC — NVIDIA Triton/TensorRT‑LLM (v1.0)

Status: Stable (draft)
Applies to: `worker-adapters/triton/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-ADAPT-5xxx`.

## 1) API Mapping

- [OC-ADAPT-5060] Adapter MUST support Triton HTTP/GRPC infer, health/metadata/stats, and optional OpenAI‑compat frontends internally.

## 2) Determinism & Version Capture

- [OC-ADAPT-5070] Adapter MUST report engine_version/trtllm_version where applicable.

## 3) Traceability

- Code: [worker-adapters/triton/src/lib.rs](../worker-adapters/triton/src/lib.rs)

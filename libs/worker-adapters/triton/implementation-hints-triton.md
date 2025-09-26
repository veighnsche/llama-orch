# Implementation Hints — NVIDIA Triton / TensorRT‑LLM

Status: draft · Engines: Triton Inference Server (HTTP/gRPC) with TensorRT‑LLM backend, or TensorRT‑LLM `trtllm-serve`; optional Triton OpenAI‑compatible frontend (Beta)

## Summary

- Two common deployment modes:
  - Triton Inference Server with TensorRT‑LLM backend (HTTP 8000, gRPC 8001, Metrics 8002).
  - Stand‑alone `trtllm-serve` (OpenAI‑compatible server) or Triton’s OpenAI‑compatible frontend.
- Prometheus metrics exposed by Triton at `/metrics` (usually port 8002); TRT‑LLM backend contributes extra metrics.
- Use strict instance group settings and disable/limit dynamic batching during determinism tests.

References

- Triton Metrics: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html>
- TensorRT‑LLM Backend (Triton): <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html>
- Triton OpenAI‑compatible frontend (Beta): <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html>
- TensorRT‑LLM `trtllm-serve` command: <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html>

## Startup notes

Triton with TRT‑LLM backend (docker example)

- Exposes:
  - HTTP: `:8000` (model inference & management)
  - gRPC: `:8001`
  - Metrics: `:8002` (Prometheus)
- Health endpoints (HTTP):
  - `GET /v2/health/live`
  - `GET /v2/health/ready`
  - `GET /v2/models/{model}/ready`
- Inference endpoint (HTTP):
  - `POST /v2/models/{model}/infer`

Triton OpenAI frontend (Beta)

- Provides `/v1/chat/completions` etc. Enable with the OpenAI frontend and flags (see docs).
- Supports tool‑calling; can be fronted to TRT‑LLM backend models.

TensorRT‑LLM `trtllm-serve`

- OpenAI‑compatible server (port configurable). Handy when you don’t need full Triton.
- See command reference for flags: parallelism, kv‑cache, batching, etc.

## Determinism profile and caveats

- Prefer greedy decoding: set temperature 0, top‑p 1 (and `do_sample=false` if applicable).
- Ensure a stable execution plan:
  - Triton model config: set a single `instance_group` (count: 1) for the test; pin GPU device mask.
  - Disable or minimize dynamic batching (remove `dynamic_batching` or set minimal `max_queue_delay_microseconds`).
  - Avoid concurrent requests during the determinism suite (or single request in flight).
- Pin exact TRT‑LLM + Triton versions and model artifacts across replicas.
- For the OpenAI frontend, treat determinism as best‑effort; use single instance and disable batching where possible.

## Metrics to scrape

- Triton exposes Prometheus metrics at `/metrics` (port 8002 by default); includes request counts/latencies, GPU utilization, and backend stats.
- TRT‑LLM backend adds batch manager statistics (available on recent Triton/Backend versions).
- Adapter should enrich with orchestrator labels (`engine`, `engine_version`, `pool_id`, `replica_id`, `model_id`).

## Adapter mapping tips

- Health: use `GET /v2/health/ready` for server readiness and `GET /v2/models/{model}/ready` for per‑model readiness.
- Completion:
  - Triton native: `POST /v2/models/{model}/infer` (HTTP) or gRPC equivalent; for streaming, you may need server‑side sequence APIs or OpenAI frontend depending on setup.
  - OpenAI frontend / `trtllm-serve`: use `/v1/chat/completions` or `/v1/completions` with SSE.
- Cancel: for HTTP streaming, close the client stream; ensure backend frees resources/sequence.
- Version: surface Triton server version and TRT‑LLM backend version (visible in logs or model metadata); attach to `engine_version`.

## Capabilities required by orchestrator

- Multi‑GPU
  - Triton supports multiple GPUs via model `instance_group` settings; pin device masks explicitly. TensorRT‑LLM enables tensor/pipeline parallelism; for `trtllm-serve`, use the appropriate parallelism flags (e.g., TP/PP) per docs.
- Streaming SSE
  - Native Triton HTTP/gRPC `infer` is request/response. Streaming token output is available via the OpenAI‑compatible frontend (Beta) or `trtllm-serve`, which supports SSE on `/v1/chat/completions` and `/v1/completions`. Map to orchestrator SSE framing (`started`, `token`, `metrics`, `end`, `error`).
- Cancellation
  - For SSE streams (OpenAI frontend/`trtllm-serve`), cancel by closing the client stream; ensure sequence/resources are freed. For native Triton, implement timeouts/cancel semantics via client and server settings.
- Metrics
  - Triton exposes Prometheus `/metrics` (port 8002 by default). Adapter must attach orchestrator labels (`engine`, `engine_version`, `pool_id`, `replica_id`, `model_id`), and include `trtllm_version` where applicable.
- Determinism & Version Pinning
  - Prefer greedy decoding (temperature 0, top‑p 1) and single instance per GPU during determinism tests; minimize/disable dynamic batching. Pin Triton and TRT‑LLM versions and model artifacts across replicas.

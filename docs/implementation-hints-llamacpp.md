# Implementation Hints — llama.cpp server (HTTP)

Status: draft · Engine: llama.cpp HTTP server (native + OpenAI‑compatible)

## Summary

- Use llama.cpp’s built‑in HTTP server for both native and OpenAI‑compatible APIs.
- For determinism tests within a replica set, prefer single‑request decoding and disable continuous batching.
- Expose Prometheus metrics and slots to the orchestrator.

References

- Server README (endpoints, metrics, slots): <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
  - Raw view for anchors: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md>
- Discussion on `--parallel` and continuous batching semantics: <https://github.com/ggml-org/llama.cpp/discussions/4130>

## Startup examples

CPU (CI fallback)

```
llama-server \
  --model ~/.cache/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 \
  --metrics --no-webui \
  --parallel 1 --no-cont-batching
```

GPU flags (example; adapt to host): `--n-gpu-layers`, `--ngl`, CUDA/HIP/Vulkan builds, etc.

Notes

- `--metrics` is required to expose `/metrics`.
- `--parallel 1` and `--no-cont-batching` reduce cross‑request effects for determinism.

## Endpoints to use

Native

- GET `/health` — health/ready. 200 OK when ready; 503 while loading.
- POST `/completion` — completion (supports streaming via SSE when `stream=true`).
- POST `/tokenize`, POST `/detokenize`, POST `/embedding`, GET `/props`, POST `/props`.
- GET `/slots` — per‑slot state and parameters (disable with `--no-slots`).
- GET `/metrics` — Prometheus metrics (requires `--metrics`).

OpenAI‑compatible

- GET `/v1/models`
- POST `/v1/completions`
- POST `/v1/chat/completions`
- POST `/v1/embeddings`

See server README for full payloads and options.

## Determinism profile (per replica set)

- Launch two identical replicas with same binary and model artifacts.
- Flags to prefer for byte‑exact checks:
  - `--parallel 1` to avoid multiple concurrent decodes per step.
  - `--no-cont-batching` to disable continuous batching.
- Request parameters
  - Set an explicit `seed` value in each job; if omitted, orchestrator may inject `seed = hash(job_id)`.
  - Prefer greedy: `temperature=0`, `do_sample=false` (via OpenAI‑compat: `temperature=0`, `top_p=1`).
- Pin engine commit/version and sampler profile version.

## Metrics to scrape (examples)

- `llamacpp:prompt_tokens_total`
- `llamacpp:tokens_predicted_total`
- `llamacpp:prompt_tokens_seconds`
- `llamacpp:predicted_tokens_seconds`
- `llamacpp:kv_cache_usage_ratio`
- `llamacpp:kv_cache_tokens`
- `llamacpp:requests_processing`, `llamacpp:requests_deferred`

Adapter should add standard labels (`engine`, `engine_version`, `model_id`, pool/replica IDs) when exporting to the orchestrator metrics pipeline.

## Adapter mapping tips

- Health: map `/health` 200→Ready, 503→Unready (loading).
- Properties: use `/props` and `/slots` to infer slot count and state.
- Completion: prefer streaming SSE for low TTFB; surface token deltas to metrics.
- Cancel: if implementing, cancel at HTTP layer by closing the stream; ensure Worker frees a slot.
- Version: include `engine_version` from server banner or build info if exposed; otherwise embed at deploy time.

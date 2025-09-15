# Implementation Hints — Hugging Face Text‑Generation‑Inference (TGI)

Status: draft · Engine: TGI (custom API with optional OpenAI‑compatible layer)

## Summary

- TGI serves high‑performance text generation with a custom API (`/generate`, `/generate_stream`) and supports health/info endpoints.
- It exposes a Prometheus `/metrics` endpoint for observability and provides a Grafana tutorial.
- Optional OpenAI‑compatible surface may be available depending on deployment (Messages API).

References

- TGI docs (index): <https://huggingface.co/docs/text-generation-inference/en/index>
- TGI monitoring with Prometheus/Grafana: <https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/monitoring>
- TGI repo: <https://github.com/huggingface/text-generation-inference>
- TGI API (Swagger UI): <https://huggingface.github.io/text-generation-inference/>

## Startup notes (containerized)

- Typical deployments use the official container image and pass model + device flags.
- Ensure network and port exposure to allow the orchestrator to reach HTTP and `/metrics`.
- Export OTLP if you need distributed traces (`--otlp-endpoint` in newer versions).

## Endpoints to use (custom API)

- GET `/health` — liveness/readiness.
- GET `/info` — model/engine information (varies by version).
- POST `/generate` — sync generation.
- POST `/generate_stream` — streaming generation (SSE/websocket depending on version).
- GET `/metrics` — Prometheus metrics on the same HTTP port.

Optional OpenAI‑compatible (when enabled)

- POST `/v1/chat/completions` (Messages API)
- POST `/v1/completions`
- POST `/v1/embeddings`

## Determinism profile and caveats

- Prefer greedy decoding parameters to reduce variance:
  - `do_sample=false`, `temperature=0`, `top_p=1`, `top_k=0`.
- If the engine/version exposes a `seed` parameter, set it explicitly per job; otherwise rely on greedy settings.
- Reduce concurrency during the determinism test window (or single worker/slot mode) to avoid cross‑request batch effects.
- Pin engine version + model artifacts across replicas.

## Metrics to scrape (examples)

- TGI exposes multiple metrics via `/metrics` (Prometheus). The official tutorial shows how to scrape from the TGI port and visualize in Grafana.
- Expect request rates, latencies (prefill/decode), effective batch sizes, token counters, etc.

Adapter should attach orchestrator labels (`engine`, `engine_version`, `pool_id`, `replica_id`, `model_id`) when forwarding to the central metrics pipeline.

## Adapter mapping tips

- Health: treat `/health` 200 as Ready. Use `/info` to capture model metadata and engine version if available.
- Completion: prefer the streaming endpoint for low TTFB; surface token deltas to metrics.
- Cancel: close the client stream; ensure worker frees resources.
- Version: derive `engine_version` from container tag or `/info` where present; pin in deployment metadata.

## Capabilities required by orchestrator

- Multi‑GPU
  - Supported via tensor parallelism (sharding). Use the container flag `--num-shard` (or env `NUM_SHARD`) to split the model across multiple GPUs on a single node. Verify model support for tensor parallelism in TGI.
- Streaming SSE
  - Use `/generate_stream` for streaming tokens. Map events to the orchestrator’s SSE framing (`started`, `token`, `metrics`, `end`, `error`).
- Cancellation
  - Cancel by closing the client stream (SSE or websocket depending on version). Ensure the worker frees resources promptly.
- Metrics
  - `/metrics` provides Prometheus metrics. Adapter must attach orchestrator labels (`engine`, `engine_version`, `pool_id`, `replica_id`, `model_id`).
- Determinism & Version Pinning
  - Prefer greedy decoding (`do_sample=false`, `temperature=0`, `top_p=1`). Reduce concurrency during determinism tests. Pin engine version and model artifacts across replicas.

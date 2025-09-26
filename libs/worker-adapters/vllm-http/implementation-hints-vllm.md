# Implementation Hints — vLLM (OpenAI‑compatible server)

Status: draft · Engine: vLLM OpenAI‑compatible server

## Summary

- vLLM serves OpenAI‑compatible endpoints for Completions/Chat/Embeddings.
- Prometheus metrics are exposed at `/metrics` on the same HTTP server.
- Reproducibility/determinism in the online server is limited by design per official docs.

References

- OpenAI‑compatible server docs: <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- Reproducibility guidance (official): <https://docs.vllm.ai/en/v0.9.1/usage/reproducibility.html>
- Metrics in production (Prometheus): <https://docs.vllm.ai/en/v0.6.1/serving/metrics.html>

## Startup example

Basic

```bash
vllm serve <model> \
  --host 0.0.0.0 --port 8000
```

Notes

- The server exposes OpenAI‑compatible APIs and a Prometheus‑compatible `/metrics` endpoint.
- Use `--tensor-parallel-size`, device selection, and other capacity flags per your host and model.

## Capabilities required by orchestrator

- Multi‑GPU
  - Supported via tensor parallelism with `--tensor-parallel-size` on a single node; distributed setups are possible via Ray/other launchers depending on version.
- Streaming SSE
  - OpenAI‑compatible `chat/completions` and `completions` support `stream=true`, which yields server‑sent events; map to orchestrator SSE framing (`started`, `token`, `metrics`, `end`, `error`).
- Cancellation
  - Cancel by closing the HTTP stream from the client; the adapter should ensure the worker frees resources and the slot is returned.
- Metrics
  - `/metrics` exposes Prometheus metrics. Adapter must attach orchestrator labels: `engine`, `engine_version`, `pool_id`, `replica_id`, `model_id` when forwarding.
- Determinism & Version Pinning
  - Treat vLLM determinism as best‑effort (per official docs). Prefer greedy decoding (`temperature=0`, `top_p=1`) and minimize concurrency during determinism tests. Pin engine version and model artifacts across replicas.
- Embeddings
  - `/v1/embeddings` is available; expose only if enabled and covered by policy.

## Endpoints to use (OpenAI‑compatible)

- GET `/v1/models` — list models (liveness proxy).
- POST `/v1/completions` — text completions.
- POST `/v1/chat/completions` — chat completions (SSE via `stream=true`).
- POST `/v1/embeddings` — embeddings.
- GET `/metrics` — Prometheus metrics (server‑level + request‑level).

## Determinism profile and caveats

Per vLLM reproducibility docs:

- “vLLM does not guarantee the reproducibility of the results by default, for the sake of performance.”
- “The online serving API (`vllm serve`) does not support reproducibility because it is almost impossible to make the scheduling deterministic in the online setting.”
- For offline/SDK usage, to achieve reproducibility:
  - Set a fixed `seed`.
  - For V1, set `VLLM_ENABLE_V1_MULTIPROCESSING=0` (turn off multiprocessing) and run on the same hardware and vLLM version.

Recommendations for orchestrator tests:

- For the Determinism Suite, treat vLLM as “best‑effort deterministic”.
- Use greedy decoding (`temperature=0`, `top_p=1`) and minimize concurrency for the test window.
- Keep engine version pinned; compare replicas built from the exact same vLLM version and model artifacts.

## Metrics to scrape (examples)

Exposed via `/metrics` (Prometheus):

- Request‑level latencies (TTFT, per‑token), throughput, queueing; server‑level utilization.
- Use the vLLM docs/grafana example for exact metric names and panels.

Adapter should attach orchestrator labels (`engine`, `engine_version`, `pool_id`, `replica_id`, `model_id`) when forwarding to the central metrics pipeline.

## Adapter mapping tips

- Health: treat `GET /v1/models` 200 as liveness. Readiness can be inferred after the first successful completion, or by a lightweight canary request (`max_tokens=0`).
- Completion/Chat: prefer streaming SSE for low TTFB; expose token deltas to metrics.
- Cancel: cancel by closing the client stream; ensure the server frees resources.
- Version: include `engine_version` from container/image tag or service banner where available; pin in deployment metadata.

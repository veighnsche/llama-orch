# Token Streaming and Cancellation Robustness — Draft

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Summary
Harden the end-to-end token stream path and cancel semantics from worker adapters to client SSE. Replace the current "build-one-big-String" stub with true incremental SSE, propagate cancellation via per-task tokens, handle client disconnects, add bounded backpressure, and adopt a low-allocation streaming decode in adapters. No new crates; this is pipeline hardening across `orchestratord`, `adapter-host`, and `worker-adapters`.

## Findings (from code/spec review)
- `orchestratord/src/services/streaming.rs` builds a full SSE transcript into a `String` and returns it; not incremental streaming; cancel is polled with sleeps.
- `orchestratord/src/api/data.rs::stream_task()` sets `Content-Type: text/event-stream` but returns a prebuilt body; no `Cache-Control: no-cache`, no keep-alive/heartbeat, no client disconnect handling.
- Cancel path uses a shared `cancellations` set; no per-stream `CancellationToken`; adapters are not signaled.
- `worker-adapters/http-util` lacks streaming decode helpers and stream-specific timeouts; adapter crates use stubs.
- Tests validate ordering on strings rather than live streaming; no client disconnect → cancel propagation coverage.

## Requirements (normative)
- [SSE-1001] The server MUST stream SSE frames incrementally with bounded memory and flush per event.
- [SSE-1002] Event order MUST be `started → token* → [metrics]* → end`. No tokens may be emitted after cancel.
- [SSE-1003] A per-task `CancellationToken` MUST be plumbed end-to-end. Server SHOULD treat client disconnect as cancel (configurable).
- [SSE-1004] The streaming pipeline MUST be backpressure-aware via a bounded channel. Metrics MAY be dropped under pressure; tokens MUST NOT be dropped.
- [SSE-1005] HTTP/2 SHOULD be preferred for SSE; fallback to HTTP/1.1 without behavior change.
- [ADAPT-3101] HTTP-based adapters MUST adopt a shared low-allocation stream decoder and support best-effort cancel by closing the body or calling engine cancel when available.

## Transport choice: SSE vs alternatives

- SSE (Server-Sent Events): one-way, text/event-stream over HTTP/1.1 or HTTP/2. Works through proxies, simple to implement and test, aligns with our current tests/specs. Backpressure is handled by TCP; frames are line-oriented and easy to multiplex (token, metrics, narration). Limitation: not bi-directional (we don’t need bi-di for tokens).
- WebSockets: full-duplex, but adds stateful connection management and more complex proxy behavior. Useful if we needed client→server control mid-stream; our cancel path is via HTTP endpoint and/or disconnect, so WebSockets are not required.
- gRPC streaming: excellent for typed streams, but browser/client support is uneven without gRPC-Web shims; adds tooling overhead. Overkill for the Home Profile target and our simple framing.
- Chunked HTTP (manual): similar to SSE but lacks standardized event framing; we would reinvent parts of SSE.
- WebTransport/HTTP/3: promising but immature and not necessary for the Home Profile.

Conclusion: For our single-direction token stream with cancel via separate HTTP path and requirement to optionally multiplex narration, SSE is the best fit for now. We will keep an escape hatch to swap transports later if requirements change.

## Proposed Changes
### Orchestratord (API)
- Switch `GET /v1/tasks/:id/stream` to `axum::response::Sse` or `Body::wrap_stream` over a `Stream<Item = Bytes>`.
- Add headers: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no` (advisory), keep `Content-Type: text/event-stream`.
- Echo or generate `X-Correlation-Id` and include in `started` metadata.

### Orchestratord (Service)
- Replace `render_sse_for_task` with a producer that:
  - Reads adapter `TokenStream`, converts to SSE frames, and sends via a bounded `mpsc` (e.g., cap 32).
  - Uses a per-task `CancellationToken`; sources: explicit cancel endpoint and client disconnect.
  - Optional micro-batching done in real-time with a tight timer (disabled by default).
  - Emits `metrics` frames at a bounded cadence after first token; drop `metrics` first if back-pressured; never drop `token`.
- Metrics/logs: `latency_first_token_ms`, `latency_decode_ms`, `tokens_out_total`, `sse_backpressure_drops_total`; narration breadcrumbs on start/cancel/end.

### Cancel Path
- Maintain `task_id → CancellationToken` map; trigger on `POST /v1/tasks/:id/cancel` and when the client disconnects (if enabled). Clean up tokens at terminal events.

### Adapter Host & Worker Adapters
- Adapter Host facade accepts a `CancellationToken` and propagates cancel to engine client.
- `worker-adapters/http-util` gains:
  - SSE line reader tolerant to partial frames and UTF-8 boundaries; minimal allocation `TokenEvent` framing.
  - Stream establishment timeouts and no mid-stream retries.
- Adapters (llamacpp-http, vllm-http, tgi-http, triton) adopt the shared decoder, preserve ordering, and support best-effort cancel.

### Concurrent log streaming (narration)
- We MUST NOT emit a log frame per token. Instead, we co-stream optional `narration` frames at a bounded cadence (e.g., at start, cancel, end, and at most every N milliseconds while active), sourced from `observability/narration-core`.
- Multiplex strategy: include `event: narration` frames in the same SSE stream. These carry short human-readable breadcrumbs and/or compact JSON. They are rate-limited and drop-first under backpressure (never drop `token`).
- Config:
  - `ORCHD_SSE_LOGS_ENABLED` (default true for start/cancel/end only; periodic narration disabled by default).
  - `ORCHD_SSE_LOGS_PERIOD_MS` (default 0 = disabled; when >0, emits narration at most every period).
- Proof: BDD adds a scenario asserting no more than K narration frames during a stream and strictly no per-token narration.

### Config (new)
- `ORCHD_SSE_CHANNEL_CAP` (default 32)
- `ORCHD_SSE_MICROBATCH` (default false)
- `ORCHD_SSE_HEARTBEAT_MS` (default 0 = disabled; emits `: keep-alive` comment lines)
- `CANCEL_ON_DISCONNECT` (default true)

## Tests & Proof
- Unit: event order, micro-batching boundaries, backpressure behavior, zero tokens after cancel.
- Integration: client disconnect triggers cancel; adapter cancel invoked.
- Adapter: decode against fragmented SSE bytes; timeouts; error taxonomy mapping to `error` frames.
- Determinism: byte-exact token sequences for same `{prompt, params, seed}` on same replica.
- Proof bundle: SSE transcripts, narration logs, determinism outputs, metrics lints per `.docs/testing/`.

## Spec Updates To Apply
- `/.specs/20-orchestratord.md`:
  - HTTP/2 preference; buffered incremental emitter; micro-batch optional; cancel-on-disconnect; backpressure rules; heartbeat optional; optional `narration` frames at bounded cadence; MUST NOT emit per-token logs.
- `/.specs/35-worker-adapters.md` and per-adapter specs (`40..44-*`):
  - Require shared streaming decode, cancel propagation, deterministic ordering, low-allocation path.

## CHECKLIST.md — Add/Amend Items
- Orchestratord
  - [ ] SSE emitter streams incrementally with headers (no-cache, keep-alive, X-Accel-Buffering: no). (ORCH-3401)
  - [ ] Per-task CancellationToken; cancel-on-disconnect (configurable); zero tokens after cancel. (ORCH-3402)
  - [ ] Bounded channel + backpressure policy; drop `metrics` under pressure; never drop `token`; add backpressure metrics. (ORCH-3403)
  - [ ] Micro-batching implemented as real-time coalescing (default off). (ORCH-3400+)
  - [ ] Optional heartbeats `: keep-alive` with configurable cadence. (ORCH-3404)
  - [ ] Narration co-streaming policy: emit `narration` frames only at start/cancel/end by default; optional periodic cadence; NEVER per-token logs. (ORCH-3308)
- Worker Adapters
  - [ ] Adopt `http-util` streaming decode; stream establishment timeouts; cancel propagation. (ORCH-3612+)
  - [ ] Map adapter errors to SSE `error` frames; preserve `started/token/metrics/end` ordering. (ORCH-3274/3276)

## SPEC_CHECKLIST.md — Add/Amend Items
- [ ] `/.specs/20-orchestratord.md`: promote HTTP/2-preferred SSE, incremental emitter, cancel-on-disconnect, backpressure, heartbeat (optional) and narration policy (no per-token logs) to normative.
- [ ] `/.specs/35-worker-adapters.md` and per-adapter specs: require shared streaming decode and cancel propagation; determinism and low-alloc path.
- [ ] `/.specs/metrics/otel-prom.md`: optionally add guidance for SSE latency buckets and backpressure counters (if missing).

## Refinement Opportunities
- Investigate cooperative yielding to reduce tail latency during bursty token output.
- Explore adaptive micro-batch window based on observed client RTT.
- Consider emitting minimal `metrics` frames only at `end` for extremely slow clients.

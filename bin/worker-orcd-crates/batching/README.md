# batching — worker-orcd batching scheduler

Status: Experimental (M1)
Scope: Worker-local batching/coalescing of Execute requests per resident model handle.

- High-level spec: `/.specs/35-worker-orcd-pool-managerd-contract.md` (§2.5 multi-residency, §2.6 fairness, §2.7 batching)
- Implementer guide: `bin/worker-orcd/BATCHING.md`

## Purpose

Provide a reusable scheduler that:
- Coalesces Execute requests within a short window (`batch_window_ms`) into micro/continuous batches per handle.
- Preserves fairness, cancellation, and SSE isolation.
- Abstracts engine specifics via a minimal trait; adapters (llamacpp, vllm, tgi) implement the trait.

## High/Mid/Low behaviors

- High: Continuous batching supported, cancel-aware, dynamic membership, per-handle concurrency caps, full metrics.
- Mid: Fixed micro-batching (windowed coalescing), FIFO admission, cancel at step boundaries, basic metrics.
- Low: Pass-through (no batching); still enforces drain gating and emits minimal metrics.

## Configuration knobs

- `batch_window_ms` (default 10)
- `max_batch_size` (default 8)
- `per_handle_queue_max` (default 256)
- `per_handle_concurrency_cap` (default 0 = unlimited)
- Derived: `supports_continuous_batching(handle)` from engine/adapter.

Suggested envs (read by worker, injected here as config):
- `LLORCH_WORKER_BATCH_WINDOW_MS`
- `LLORCH_WORKER_MAX_BATCH_SIZE`
- `LLORCH_WORKER_QUEUE_MAX`
- `LLORCH_WORKER_HANDLE_CONCURRENCY_CAP`

## Public surface (preview)

```rust
pub trait BatchDecodeEngine {
    type HandleId: Clone + Eq + std::hash::Hash + std::fmt::Debug;

    fn supports_continuous_batching(&self, handle: &Self::HandleId) -> bool;

    fn decode_step(
        &self,
        handle: &Self::HandleId,
        active: &mut [SequenceCtx],
    ) -> StepOutput; // tokens, finished set, errors
}

pub struct BatchScheduler<E: BatchDecodeEngine, A: BatchAdapterEvents> { /* ... */ }
```

See `src/lib.rs` for the full interface skeleton.

## What this crate is NOT

- Not a CUDA implementation. Engine-specific batching (kernels/cublasLt) stays in adapter crates.
- Not HTTP/SSE code. Routes remain in `bin/worker-orcd/` and `worker-orcd-crates/api/`.

## Next steps

- Implement `BatchScheduler` with coalescing window and cancel-aware decode loop.
- Add metrics adapters (Prometheus) and config plumbing.
- Write unit tests for coalescing, FIFO, cancellation, and bounded queues.

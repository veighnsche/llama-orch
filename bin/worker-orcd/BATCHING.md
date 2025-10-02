# worker-orcd Batching Guide

Status: Implementer Guide (aligns with root §2.7 Batching)
See also: `/.specs/35-worker-orcd-pool-managerd-contract.md` §2.7, §2.6; `bin/worker-orcd-crates/api/.specs/00_api.md`

---

## 1. Goals & Scope

- Improve throughput and GPU utilization by decoding multiple sequences together on the same resident model handle.
- Preserve fairness and per-request isolation (each request has its own SSE stream).
- Keep batching per-handle (model/shard). Cross-handle batching is out of scope.
- Work with both fixed micro-batches and continuous batching engines.

Non-goals:
- Changing orchestrator placement. Batching is worker-internal and should be transparent upstream.
- Cross-worker batching.

---

## 2. Capabilities & Configuration

Expose/derive these runtime knobs:

- `supports_continuous_batching: bool` (from adapter/engine introspection)
- `max_batch_size: u32` (engine- or memory-dependent; safe default 4–8)
- `batch_window_ms: u32` (admission coalescing window, e.g., 5–20ms)
- `per_handle_queue_max: u32` (bound queued sequences per handle)
- `per_handle_concurrency_cap: u32` (optional; max concurrent sequences per handle)

Suggested env vars (read at startup, override per-adapter if needed):

- `LLORCH_WORKER_BATCH_WINDOW_MS=10`
- `LLORCH_WORKER_MAX_BATCH_SIZE=8`
- `LLORCH_WORKER_QUEUE_MAX=256`
- `LLORCH_WORKER_CONTINUOUS_BATCHING=true|false`
- `LLORCH_WORKER_HANDLE_CONCURRENCY_CAP=0` (0 = unlimited)

---

## 3. Data Model

- `HandleId`: identifies a sealed resident model/shard (from Commit).
- `Sequence`: one Execute request (prompt + params + connection handles).
- `Batch`: ordered set of `SequenceId`s sharing the same `HandleId`.
- `slots_total/free`: represent the maximum concurrent sequences across all handles (not OS threads).

Queues:
- `ready_queue[HandleId]`: waiting sequences per handle.
- `running[HandleId]`: sequences currently decoding.

---

## 4. Admission & Coalescing

- On Execute(HandleId):
  - Reject immediately if draining (503 ADMISSION_CLOSED).
  - If `per_handle_queue_max` reached, return 429 or 503 with `QUEUE_FULL` (pick one and document).
  - Enqueue into `ready_queue[HandleId]` with arrival timestamp.
- Coalescing tick every `batch_window_ms`:
  - For each handle with non-empty queue, build a batch up to `max_batch_size` or available slots.
  - Prefer FIFO order; respect `per_handle_concurrency_cap` if set.

Continuous batching engines:
- Admit new sequences when GPU kernel reaches a safe boundary; do not exceed `max_batch_size`.

---

## 5. Scheduling & Decode Loop

Fixed micro-batch pseudocode:

```rust
loop {
    for handle in handles {
        let avail = slots_free_for(handle);
        if avail == 0 { continue; }

        let mut batch = take_fifo(&ready_queue[handle], min(avail, max_batch_size));
        if batch.is_empty() { continue; }

        start_sse_started(&batch);
        move_to_running(handle, &batch);

        while !batch.is_empty() {
            let tokens = engine.decode_step(handle, &batch)?;  // may return per-seq token or None
            for (seq, tok) in tokens { emit_sse_token(seq, tok); }

            // drop completed/cancelled sequences
            prune_completed_or_cancelled(&mut batch);
        }

        emit_sse_end_metrics(&batch);
        release_slots(handle);
    }
    sleep(scheduling_quantum_ms);
}
```

Continuous batching adjustments:
- Maintain a live set; on each safe point, add sequences from `ready_queue` and remove completed/cancelled ones.
- Ensure per-sequence fairness via round-robin across active sequences.

---

## 6. Cancellation & Disconnects

- Each Sequence tracks cancellation via a `CancellationToken`.
- Propagate cancel on client disconnect (per proposal `/.specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md`).
- On cancel: mark sequence; drop at the next safe boundary; emit `error` SSE for that stream only.

---

## 7. Fairness Interactions

- Respect drain gating (no new admissions while draining).
- Inter-handle fairness is primarily controlled by orchestrator placement; intra-worker fairness:
  - FIFO admission within a handle when `slots_free == 0`.
  - Optional per-handle concurrency caps to avoid a single hot handle monopolizing slots.
  - Within a batch, interleave decode steps fairly (e.g., one token per schedule) to prevent starvation.

---

## 8. Metrics & Logging

Emit Prometheus metrics (names illustrative):

- `worker_batch_size{handle_id}` (histogram)
- `worker_batch_wait_ms{handle_id}` (histogram)
- `worker_sequences_running{handle_id}` (gauge)
- `worker_execute_started_total{handle_id}` (counter)
- `worker_execute_completed_total{handle_id}` (counter)
- `worker_execute_cancelled_total{handle_id}` (counter)
- `worker_tokens_out_total{handle_id}` (counter)
- `worker_decode_time_ms{handle_id}` (histogram)

Logs (structured): batch formation, adds/removes, cancel events, error paths.

---

## 9. Error Handling

- Return `503 ADMISSION_CLOSED` when draining.
- Return `507 VRAM_OOM` on memory failure (align with residency/eviction logic).
- Return `400 INVALID_PARAMS` for malformed Execute requests.
- Map adapter/engine failures to stable error codes and include detail in SSE `error` event.

---

## 10. Testing Guidance

- Unit tests: batch window coalescing, FIFO order, cancellation removal, capacity limits.
- Integration: mock adapter that simulates continuous batching; verify fairness and isolation of SSE streams.
- Load tests: confirm throughput gains at small windows (5–20ms) without unacceptable tail latency.

---

## 11. Pitfalls & Tips

- Do not over-widen `batch_window_ms`: it harms p50 latency. Tune by hardware.
- Continuous batching can amplify head-of-line blocking if not careful—interleave per token.
- Ensure backpressure: bound queues and document rejection behavior.
- Keep per-request SSE strictly isolated; never multiplex tokens across streams wrongly.

---

## 12. Refinement Opportunities

- Adaptive `batch_window_ms`: shrink under load, grow when GPU idle.
- Dynamic `max_batch_size`: based on observed VRAM headroom and token throughput.
- Per-handle QoS classes (interactive vs bulk).
- Emit utilization hints to orchestrator (optional extension).

---

## 13. CUDA C++ Notes for Batching (optional but recommended)

If the adapter/engine exposes CUDA hooks (or you own the kernels), the following patterns improve batch throughput and latency stability.

### 13.1 Execution Model

- Token-step kernels: launch per decode step across the active sequences in the batch; keep kernel durations short (< 10–50ms) to remain responsive to cancels and dynamic membership.
- Persistent kernels: a resident kernel pulls sequence work from a device queue; good for low-latency admission but more complex to reason about. Prefer token-step first.

Kernel skeleton (token-step):
```cpp
struct SeqDesc { int seq_id; int active; int kv_offset; int length; };

__global__ void decode_step_kernel(const SeqDesc* __restrict__ seqs,
                                   int nseq,
                                   const half* __restrict__ kv_cache,
                                   const half* __restrict__ W,
                                   half* __restrict__ out) {
  int s = blockIdx.x;        // sequence index
  int tid = threadIdx.x;     // within-seq lane
  if (s >= nseq || !seqs[s].active) return;
  // ... per-sequence compute: attention + MLP using cublasLt or custom fused kernels ...
}

// host
dim3 grid(nseq);
dim3 block(256);
decode_step_kernel<<<grid, block, 0, compute_stream>>>(d_seqs, nseq, d_kv, d_W, d_out);
CUDA_CHECK(cudaGetLastError());
```

### 13.2 Streams & Overlap

- Use at least 2–3 streams per handle: `compute_stream`, `h2d_stream`, optional `prefetch_stream`.
- Upload prompts/params and stage next-step data on `h2d_stream` while compute runs on `compute_stream`.
- Use CUDA events to order compute after copies without global sync.

```cpp
cudaMemcpyAsync(d_prompt, h_prompt, sz, cudaMemcpyHostToDevice, h2d_stream);
cudaEventRecord(evt_ready, h2d_stream);
cudaStreamWaitEvent(compute_stream, evt_ready, 0);
decode_step_kernel<<<grid, block, 0, compute_stream>>>(...);
```

### 13.3 CUDA Graphs

- Capture the fixed decode micrograph (token-step) to reduce launch overhead; instantiate per `HandleId` when shapes stabilize.
- For continuous batching, recapture when batch shape changes significantly (e.g., `nseq` or hidden size pack changes) to amortize costs.

```cpp
cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeGlobal);
  // enqueue decode_step_kernel + any cublasLt calls
cudaStreamEndCapture(compute_stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(graphExec, compute_stream);
```

### 13.4 Memory Layout & Precision

- Use SoA layouts for per-sequence state to maximize coalesced loads: `[seq][feature]` contiguous along feature.
- Align KV cache pages to 128–256B; keep per-seq pages contiguous to improve spatial locality.
- Prefer FP16/BF16 with tensor cores via cuBLASLt; calibrate accumulation (FP32) for stability.
- Consider fused attention kernels if available; otherwise chain cuBLASLt GEMMs with careful workspace reuse.

### 13.5 Dynamic Batch Membership

- Maintain a device-visible active mask or queue of `SeqDesc` entries.
- When a sequence completes/cancels, flip `active=0`; compact periodically (prefix-sum or stream compaction kernel) to keep `nseq` small.
- Admit new sequences between token steps; never modify descriptors mid-kernel.

### 13.6 Synchronization & Reductions

- Prefer warp-level intrinsics (`__shfl_sync`, `__syncwarp`) and cooperative groups for reductions over atomics.
- Avoid global device syncs; synchronize per stream with events.

### 13.7 Watchdog & Latency Budget

- Keep each kernel < ~50ms to avoid TDR/watchdog issues and to remain cancel-responsive.
- Break long decode work into token- or layer-granularity steps.

### 13.8 Profiling & Telemetry

- Use Nsight Systems for launch/overlap; Nsight Compute for memory and tensor core utilization.
- Export `batch_size`, `batch_wait_ms`, SM occupancy, achieved FLOPs, and H2D/D2H throughput as metrics (aggregate per handle).

### 13.9 Fallback Strategy

- If custom kernels are not feasible, rely on engine-native batching (e.g., adapter-provided). Keep the worker scheduler and SSE isolation identical; only the inner compute differs.


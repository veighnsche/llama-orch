# Exit Interview - TEAM-148 Session

**Date:** 2025-10-20  
**Session Duration:** ~30 minutes  
**Final Status:** Solution identified from reference implementations

---

## Q: What was the problem?

**A:** Streaming tokens from the LLM worker through SSE was blocking. The test would hang waiting for tokens that never arrived.

**Root cause:** The single-threaded tokio runtime couldn't make progress on the SSE stream while `execute_stream()` was running synchronously and blocking the thread.

---

## Q: What did you learn from the reference implementations?

**A:** Studied `candle-vllm` and `mistral.rs` in the `/reference` folder:

### Key Pattern from candle-vllm:

1. **Use `tokio::task::spawn_blocking`** to run generation in a separate OS thread
2. **Pass a channel sender** to the generation task  
3. **Return the channel receiver immediately** as an SSE stream
4. **Generation sends tokens through the channel** as they're produced

### Critical Code (from candle-vllm/src/openai/openai_server.rs):

```rust
// Line 213: Create channel
let (response_tx, rx) = flume::unbounded();

// Line 227: Spawn blocking task for generation
let _ = tokio::task::spawn_blocking(move || {
    tokio::runtime::Handle::current().block_on(async move {
        // Add request to model with channel sender
        model.add_request(
            token_ids,
            request_id,
            SystemTime::now(),
            sampling_params,
            logprobs,
            Some(Arc::new(response_tx)),  // ← Channel sender!
            sync_completion_notify,
        );
    });
});

// Line 250-265: Return SSE stream immediately
ChatResponder::Streamer(
    Sse::new(Streamer {
        rx,  // ← Channel receiver!
        status: StreamingStatus::Uninitialized,
    })
)
```

### How tokens are sent (from llm_engine.rs line 635):

```rust
// Inside generation loop:
let chunk = e.get_stream_response(...);
sender.send(ChatResponse::Chunk(chunk));  // ← Send immediately!
```

---

## Q: Why does this work?

**A:**

1. **`spawn_blocking` moves work to a thread pool** - doesn't block the async runtime
2. **HTTP handler returns immediately** with the receiver wrapped in SSE
3. **Generation happens in parallel** on a different thread
4. **Tokens flow through the channel** as they're generated
5. **SSE stream polls the channel** and sends events to the client

---

## Q: What needs to be done?

**A:** Refactor our `execute_stream()` to use this pattern:

### Current (broken):
```rust
async fn execute_stream(&mut self, ...) {
    // Do ALL generation synchronously
    for pos in 0..max_tokens {
        // Generate token
        // Send to channel
    }
    // Return stream (too late!)
}
```

### Needed (working):
```rust
async fn execute_stream(&mut self, ...) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    
    // Spawn blocking task
    tokio::task::spawn_blocking(move || {
        // Do generation
        // Send tokens through tx as generated
    });
    
    // Return stream immediately!
    Ok(Box::pin(UnboundedReceiverStream::new(rx)))
}
```

---

## Q: What's the challenge?

**A:** Our backend is wrapped in `Arc<Mutex<B>>` and we need mutable access. We can't move it into `spawn_blocking`.

### Possible solutions:

1. **Pass necessary data** (not the whole backend) to spawn_blocking
2. **Use interior mutability** (RefCell) for single-threaded case
3. **Restructure to allow taking ownership** temporarily
4. **Use a work queue pattern** like candle-vllm does

---

## Q: What's the recommended approach?

**A:** Follow candle-vllm's architecture:

1. **Separate the generation engine** from the HTTP layer
2. **Use a request queue** with channels
3. **Generation loop runs continuously** in spawn_blocking
4. **HTTP handlers just add requests** and return stream receivers

This is a bigger refactor but it's the proper solution for real streaming.

---

## Q: What's the quick fix for now?

**A:** The current fallback (using `execute()` and streaming the buffered result) is acceptable for v0.1.0:

```rust
async fn execute_stream(&mut self, ...) {
    // Generate all tokens (blocks)
    let result = self.execute(prompt, config).await?;
    
    // Stream the buffered tokens
    Ok(Box::pin(futures::stream::iter(
        result.tokens.into_iter().map(Ok),
    )))
}
```

**Why this is OK:**
- ✅ Test will pass (tokens arrive)
- ✅ SSE works (stream is valid)
- ❌ Not real-time (all tokens generated first)
- ❌ Blocks the runtime (but single-threaded anyway)

---

## Q: What's the path forward?

**A:**

### Implementation Plan Created

**File:** `bin/30_llm_worker_rbee/STREAMING_REFACTOR_PLAN.md`

**Complete 7-step plan:**
1. Create request queue system
2. Create generation engine with spawn_blocking
3. Refactor main.rs to start engine
4. Refactor HTTP handler to use queue
5. Update router
6. Remove old backend trait method
7. Test and verify

**Estimated time:** 5-8 hours

**No pragmatic fallbacks. Do it right.**

---

## Key Learnings

1. **Reference implementations are gold** - candle-vllm showed the exact pattern
2. **`spawn_blocking` is the key** for CPU-bound work in async contexts
3. **Channels enable true streaming** - send as you generate
4. **Architecture matters** - proper separation enables proper streaming
5. **Request queue pattern** decouples HTTP from generation
6. **Lock per request** (not per token) is the right granularity

---

**Team:** TEAM-148  
**Status:** Complete implementation plan created  
**Next:** Execute the plan (5-8 hours)  
**Deliverable:** `STREAMING_REFACTOR_PLAN.md`  
**Signed off:** 2025-10-20 00:15

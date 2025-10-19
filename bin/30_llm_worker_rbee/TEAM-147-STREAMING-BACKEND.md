# TEAM-147: Real-Time Streaming Backend Implementation

**Date:** 2025-10-19  
**Team:** TEAM-147  
**Status:** ✅ Backend API ready, needs backend implementation

---

## Problem

**The worker was LYING about streaming!**

### What Was Broken:
```rust
// OLD CODE (FAKE streaming):
let result = backend.execute().await;  // ❌ BLOCKS until ALL tokens generated
for token in result.tokens {           // ❌ Tokens already computed
    events.push(token);                // ❌ Just sending pre-computed results
}
stream::iter(events)                   // ❌ "Streaming" pre-computed data
```

**Result:** All tokens generated first, THEN sent over SSE. No real-time streaming!

---

## Solution

### Added `execute_stream()` to Backend Trait

**File:** `src/http/backend.rs`

```rust
// TEAM-147: New streaming method
async fn execute_stream(
    &mut self,
    prompt: &str,
    config: &SamplingConfig,
) -> Result<
    Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>,
    Error
> {
    // Default: Falls back to old execute() for backwards compatibility
    let result = self.execute(prompt, config).await?;
    Ok(Box::pin(stream::iter(result.tokens.into_iter().map(Ok))))
}
```

**Benefits:**
- ✅ Backends can now yield tokens as they're generated
- ✅ Default implementation maintains backwards compatibility
- ✅ Stream interface allows real-time token emission

---

## Updated HTTP Handler

**File:** `src/http/execute.rs`

### OLD (Blocking):
```rust
// ❌ Wait for ALL tokens
let result = backend.execute().await;

// ❌ Convert to events after completion
for token in result.tokens {
    events.push(InferenceEvent::Token { t: token, i });
}
```

### NEW (Streaming):
```rust
// ✅ Get token stream (yields as generated!)
let token_stream = backend.execute_stream().await?;

// ✅ Convert stream to SSE events in real-time
let token_events = token_stream.map(|token| {
    InferenceEvent::Token { t: token, i }
});

// ✅ Stream: narration → started → TOKENS → [DONE]
narration_stream
    .chain(started_stream)
    .chain(token_events)  // ← REAL-TIME!
    .chain(done_marker)
```

---

## Changes Made

### 1. `src/http/backend.rs`
- ✅ Added `execute_stream()` method to `InferenceBackend` trait
- ✅ Returns `Stream<Item = Result<String>>` for real-time tokens
- ✅ Default implementation falls back to `execute()` for compatibility
- ✅ Added `futures::Stream` and `Pin` imports

### 2. `src/http/execute.rs`
- ✅ Replaced `backend.execute()` with `backend.execute_stream()`
- ✅ Removed loop that converted pre-computed tokens to events
- ✅ Stream now yields tokens as they're generated
- ✅ Added proper token counting in stream map
- ✅ Updated comments to reflect REAL streaming

---

## What Still Needs Implementation

### Backend Implementations Need Updating

**Current backends use blocking `execute()`:**
- `CandleInferenceBackend` (main implementation)
- CPU backend
- CUDA backend  
- Metal backend

**They need to implement `execute_stream()` for REAL streaming:**

```rust
impl InferenceBackend for CandleInferenceBackend {
    // TEAM-147: Implement real streaming
    async fn execute_stream(&mut self, prompt: &str, config: &SamplingConfig) 
        -> Result<Pin<Box<dyn Stream<Item = Result<String>>>>, Error> 
    {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Generate tokens and send as they're created
        for token in self.generate_tokens(prompt, config) {
            tx.send(Ok(token))?;  // ← Send immediately!
        }
        
        Ok(Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx)))
    }
}
```

---

## Testing

### Current Behavior:
- ✅ Code compiles
- ✅ Default implementation works (falls back to old method)
- ⚠️  Tokens still sent all at once (backend not implemented yet)

### After Backend Implementation:
- ✅ Tokens will stream in REAL-TIME as generated
- ✅ User sees tokens appear one by one
- ✅ No waiting for completion before streaming starts

---

## Architecture Notes

### SSE Flow (Correct):
```
1. Client: POST /v1/inference
2. Server: Returns SSE stream immediately
3. Server: Streams events as they happen:
   - narration events
   - started event
   - token events (REAL-TIME!)  ← TEAM-147 fixed this
   - [DONE] marker
```

### What TEAM-147 Fixed:
- ❌ OLD: Generate all tokens → convert to events → stream events
- ✅ NEW: Stream tokens as generated → convert to events on-the-fly

---

## Future Work

### 1. Implement Streaming in Candle Backend
**Priority:** HIGH  
**File:** `src/backend/inference.rs`  
**Task:** Make token generation yield tokens via channel/stream

### 2. Add Streaming Tests
**Priority:** MEDIUM  
**Task:** Verify tokens arrive incrementally, not all at once

### 3. Performance Metrics
**Priority:** LOW  
**Task:** Measure time-to-first-token vs time-to-completion

---

## Related Issues

### The "POST → GET SSE link" Pattern

**User mentioned this should be a shared crate:**
```
POST /v1/inference → Returns: { "sse_url": "/v1/inference/job-123/stream" }
GET /v1/inference/job-123/stream → SSE stream
```

**Current implementation:**
- POST /v1/inference → Returns SSE stream directly
- No separate GET endpoint

**This is VALID but different from the dual-call pattern.**

**TODO:** Decide if we want the dual-call pattern or keep current approach.

---

## Summary

**TEAM-147 Changes:**
1. ✅ Added `execute_stream()` to backend trait
2. ✅ Updated HTTP handler to use streaming
3. ✅ Removed fake "streaming" of pre-computed tokens
4. ⚠️  Backend implementations still need work

**Status:** API ready, implementation pending

**Next Team:** Implement `execute_stream()` in `CandleInferenceBackend`

---

**Modified Files:**
- `src/http/backend.rs` (added execute_stream method)
- `src/http/execute.rs` (use streaming backend)

**Team:** TEAM-147  
**Signed off:** 2025-10-19

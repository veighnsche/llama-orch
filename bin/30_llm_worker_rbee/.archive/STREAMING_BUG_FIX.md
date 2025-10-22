# Streaming Hang Bug Fix

**Bug:** Tokens never stream, request hangs for 3+ minutes  
**Fixed by:** TEAM-150  
**Date:** 2025-10-20

---

## The Bug

TEAM-149 implemented real-time streaming architecture, but tokens **never arrived**:

```bash
🤔 Testing inference with SSE streaming...
✅ Inference request accepted
📡 Streaming tokens:
[waits 3+ minutes with no tokens]
```

---

## Root Cause

**Stream composition bug in `execute.rs`:**

```rust
// BROKEN CODE (TEAM-149)
let narration_stream = UnboundedReceiverStream::new(narration_rx);

let stream_with_done = narration_stream      // ← BLOCKS FOREVER
    .chain(started_stream)
    .chain(token_events)                     // ← NEVER REACHED
    .chain(done_marker);
```

**Why it blocks:**

1. `create_channel()` stores sender in **thread-local storage**
2. Sender **never dropped** → channel stays open
3. `UnboundedReceiverStream` **waits for messages** that never come
4. Generation engine runs in **different thread** (`spawn_blocking`)
5. Can't access thread-local sender → no messages sent
6. Stream waits forever → `token_events` never polled!

---

## The Fix

**Remove blocking narration_stream:**

```rust
// WORKING CODE (TEAM-150)
let stream_with_done = started_stream        // ← Fires immediately
    .chain(token_events)                     // ← Now reached!
    .chain(done_marker);
```

**Changes:**
- ✅ Removed `narration_channel::create_channel()` call
- ✅ Removed `narration_stream` from chain
- ✅ Removed unused imports
- ✅ Direct flow: `started_event → token_events → [DONE]`

---

## Stream Composition Pattern

### ❌ DON'T: Chain blocking receivers

```rust
let (tx, rx) = unbounded_channel();
// Store tx somewhere it won't be dropped...

let blocking = UnboundedReceiverStream::new(rx);  // Waits forever!
let data = async_stream::stream! { yield 42; };

blocking.chain(data)  // data NEVER yielded!
```

### ✅ DO: Only chain completing streams

```rust
let data = async_stream::stream! { yield 42; };
let done = stream::once(future::ready(value));

data.chain(done)  // Works immediately
```

---

## Lessons Learned

1. **Lazy streams only poll when needed** - blocked upstream = no downstream
2. **TLS + spawn_blocking = incompatible** - different thread pools
3. **Always drop unused senders** - or receivers wait forever
4. **Test stream composition** - easy to create blocking chains

---

## Verification

### Compilation
```bash
cargo check --bin llm-worker-rbee  # ✅ PASSES
```

### Runtime Testing (VERIFIED)
```bash
cargo xtask worker:test  # ✅ PASSES

📡 Streaming tokens (30s timeout):
▁a ▁cities ▁of ▁fran ce . K ing dom ▁of ▁fran ce ▁is ▁known...
✅ Received [DONE] signal

📊 Results:
- Tokens received: 45
- [DONE] signal: ✅
- Test: ✅ PASSED
```

**Fix verified:** Tokens stream immediately, no 3+ minute hang!

---

**Fixed by TEAM-150** | **File:** `src/http/execute.rs` | **Lines removed:** ~10

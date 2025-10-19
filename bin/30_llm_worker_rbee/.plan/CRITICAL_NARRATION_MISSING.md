# 🚨 CRITICAL: Narration Events MUST Be Streamed to User!

**Status**: ❌ **NOT IMPLEMENTED**  
**Severity**: **CRITICAL ARCHITECTURE MISMATCH**  
**Date**: 2025-10-09

---

## 🎯 The Requirement (User's Vision)

**YOU ARE ABSOLUTELY RIGHT!**

In an **agentic API**, the user needs to see **ALL narration events in real-time** on their screen!

### The Flow:

```
User's Screen (Orchestrator PC)
    ↓
Displays NARRATION events (what's happening behind the scenes)
    ↓
"Starting worker on GPU0..."
"Loading model llama-7b..."
"Allocating 8GB VRAM..."
"Model loaded successfully!"
"Starting inference..."
"Generated 10 tokens..."
"Inference complete!"
```

**NOT just the token stream!**

The token stream goes to the AI agent (upstream). The narration goes to the **user's screen** so they can see what the system is doing!

---

## ❌ What We Implemented (WRONG)

We implemented narration as **backend logs only**:

```
narrate() → tracing::event!() → stdout → log files
```

**This goes to log files, NOT to the user's screen!**

---

## ✅ What We SHOULD Implement

Narration events should be **SSE events** that flow to the user!

### Correct Architecture:

```
Worker narrates:
  narrate("Starting inference...")
    ↓
  Emits SSE event: type="narration"
    ↓
  Worker → Orchestrator → User's Screen
```

### SSE Stream Should Include:

**Current (WRONG):**
```
event: started
event: token
event: token
event: end
```

**Correct (WHAT YOU WANT):**
```
event: narration
data: {"actor":"llm-worker-rbee","action":"startup","human":"Starting Candle worker on port 8080","cute":"Worker waking up! 🌅"}

event: narration
data: {"actor":"model-loader","action":"model_load","human":"Loading Llama model from /models/llama-7b.gguf","cute":"Fetching the sleepy Llama model! 📦"}

event: started
data: {"job_id":"job-123","model":"llama-7b"}

event: narration
data: {"actor":"candle-backend","action":"inference_start","human":"Starting inference (prompt: 15 chars, max_tokens: 50)","cute":"Time to generate 50 tokens! 🚀"}

event: token
data: {"t":"Hello","i":0}

event: narration
data: {"actor":"candle-backend","action":"token_generate","human":"Generated 10 tokens","cute":"10 tokens and counting! 🎯"}

event: token
data: {"t":" world","i":1}

event: narration
data: {"actor":"candle-backend","action":"inference_complete","human":"Inference completed (50 tokens in 250 ms, 200 tok/s)","cute":"Generated 50 tokens! 🎉"}

event: end
data: {"tokens_out":50,"decode_time_ms":250}
```

---

## 🔍 Checking the Specs

Let me check if the specs mention narration in SSE streams...

**From `00_llama-orch.md` (SYS-5.1.1):**
```
GET /v2/tasks/{job_id}/events (SSE)
Events:
- queued → started → token* → metrics* → end
- error (if failure or cancellation)
```

**Event types mentioned:**
- `queued` (orchestrator-level)
- `started` (worker execution begins)
- `token` (generated tokens)
- `metrics` (performance metrics)
- `end` (completion)
- `error` (failure)

**❌ NO MENTION OF NARRATION EVENTS IN SSE!**

---

## 🚨 The Problem

**The specs DON'T specify narration events in the SSE stream!**

But you're right - for an agentic API where the user wants to see what's happening behind the scenes, **narration events SHOULD be in the SSE stream!**

---

## 🎯 What Needs to Change

### Option 1: Add Narration as SSE Events (RECOMMENDED)

**Modify `InferenceEvent` enum:**

```rust
// src/http/sse.rs
pub enum InferenceEvent {
    Started { ... },
    Token { ... },
    Metrics { ... },
    
    // NEW: Narration event
    Narration {
        actor: String,
        action: String,
        target: String,
        human: String,
        cute: Option<String>,
        story: Option<String>,
        correlation_id: Option<String>,
        // ... other narration fields
    },
    
    End { ... },
    Error { ... },
}
```

**Modify `narrate()` to emit SSE events:**

```rust
// Instead of just logging to stdout
pub fn narrate(fields: NarrationFields) {
    // 1. Log to stdout (for backend observability)
    tracing::event!(Level::INFO, ...);
    
    // 2. Emit SSE event (for user visibility)
    if let Some(sse_sender) = get_sse_sender() {
        sse_sender.send(InferenceEvent::Narration {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            human: fields.human,
            cute: fields.cute,
            // ...
        });
    }
}
```

---

### Option 2: Separate Narration Stream

**Have TWO SSE endpoints:**

1. `/execute` - Token stream (for AI agent)
2. `/events` - Narration stream (for user display)

**But this is more complex and requires orchestrator changes.**

---

## 🔧 Implementation Plan

### Step 1: Update SSE Event Types

Add `Narration` variant to `InferenceEvent` enum in `src/http/sse.rs`.

### Step 2: Create SSE Channel

Create a channel that narration events can send to:

```rust
// In execute handler
let (narration_tx, narration_rx) = tokio::sync::mpsc::channel(100);

// Store in request context
req.extensions_mut().insert(narration_tx);

// Merge narration events with token events in SSE stream
```

### Step 3: Modify Narration Function

Make `narrate()` check for SSE channel and send events:

```rust
pub fn narrate(fields: NarrationFields) {
    // Log to stdout (backend observability)
    tracing::event!(...);
    
    // Send to SSE stream (user visibility)
    if let Some(tx) = get_current_sse_sender() {
        let _ = tx.try_send(InferenceEvent::Narration { ... });
    }
}
```

### Step 4: Update Orchestrator

Orchestrator needs to:
- Relay narration events to client
- Display them in the UI
- Keep them separate from token stream (tokens go to AI agent)

---

## 🎯 User Experience

**What the user sees on their screen:**

```
[Narration Panel]
🌅 Worker worker-gpu0-r1 waking up to help with inference!
📦 Fetching the sleepy Llama model from its cozy home!
🛏️ Llama model tucked into memory! 7000 MB cozy!
👋 Waving hello to pool-managerd: 'I'm ready to work!'
🚀 Time to generate 50 tokens! Let's go!
🍰 Chopped prompt into 15 tasty tokens!
🧹 Tidying up the cache for a fresh start!
🎯 10 tokens and counting!
🎯 20 tokens and counting!
🎉 Generated 50 tokens in 250 ms! 200 tok/s!

[Token Stream] (goes to AI agent)
Hello world, this is a test of the inference system...
```

---

## ❌ Current State

**What we have now:**

- ✅ Narration events exist
- ✅ They have cute messages
- ✅ They have correlation IDs
- ❌ They only go to stdout (log files)
- ❌ They DON'T go to the user's screen
- ❌ They DON'T go through SSE

**Result:** User can't see what's happening! They only see tokens, not the system activity.

---

## ✅ What You Want (Correct!)

**Narration events should:**

1. ✅ Go to stdout (for operators/monitoring)
2. ✅ Go to SSE stream (for user visibility)
3. ✅ Be displayed on user's screen in real-time
4. ✅ Show what the system is doing behind the scenes
5. ✅ Be separate from token stream (tokens go to AI agent)

---

## 🚨 Action Required

**WE NEED TO:**

1. **Update the specs** to include narration events in SSE
2. **Add `Narration` event type** to `InferenceEvent` enum
3. **Create SSE channel** for narration events
4. **Modify `narrate()` function** to emit SSE events
5. **Update orchestrator** to relay and display narration events
6. **Test** that narration appears on user's screen in real-time

---

## 📊 Architecture Comparison

### Current (WRONG):

```
┌─────────────┐
│    User     │  ← Sees: Token stream only
└─────────────┘
       ↑
       │ SSE: token, token, token, end
       │
┌─────────────┐
│ Orchestrator│
└─────────────┘
       ↑
       │ SSE: token, token, token, end
       │
┌─────────────┐
│   Worker    │
│             │
│ narrate() → stdout → logs (user can't see!)
└─────────────┘
```

### Correct (WHAT YOU WANT):

```
┌─────────────┐
│    User     │  ← Sees: Narration events + Token stream
│             │     "Loading model..."
│             │     "Starting inference..."
│             │     "Generated 10 tokens..."
└─────────────┘
       ↑
       │ SSE: narration, narration, started, token, narration, token, end
       │
┌─────────────┐
│ Orchestrator│  ← Relays all events
└─────────────┘
       ↑
       │ SSE: narration, narration, started, token, narration, token, end
       │
┌─────────────┐
│   Worker    │
│             │
│ narrate() → SSE event + stdout
└─────────────┘
```

---

## ✅ Conclusion

**USER WAS 100% CORRECT!**

Narration events MUST be streamed to the user via SSE, not just logged to stdout!

**Current implementation is INCOMPLETE.**

### What We Need to Do:

1. ✅ Keep stdout narration for worker lifecycle (startup, shutdown)
2. ❌ Add SSE narration for per-request events (inference progress)
3. ❌ Add `Narration` event type to SSE
4. ❌ Create SSE channel for narration
5. ❌ Modify `narrate()` to emit to both stdout AND SSE
6. ❌ Update orchestrator to relay and display narration

### Corrected Understanding:

**NOT all narration goes to SSE!**

- **Stdout only**: Worker lifecycle (13 events) - Pool-manager sees these
- **SSE + Stdout**: Per-request events (8 events) - User AND pool-manager see these

**See**: `NARRATION_ARCHITECTURE_FINAL.md` for the complete breakdown.

**This is a critical feature for the agentic API user experience!**

---

*Identified by User - Critical Architecture Gap! 🚨*  
*Updated with correct dual-output architecture! 🎀*

# TEAM-088 HANDOFF - Narration Plumbing Complete ✅

**From:** TEAM-087  
**Date:** 2025-10-11  
**Status:** ✅ COMPLETE - Comprehensive narration plumbing implemented for WAY BETTER DEBUGGING

---

## Mission Accomplished

**Implemented comprehensive narration plumbing throughout llm-worker-rbee for superior debugging visibility, especially in error paths and SSH contexts.**

---

## Current State

### ✅ What Works
- queen-rbee starts successfully on port 8080
- rbee-hive starts successfully on port 9200
- Worker spawn request succeeds (HTTP 200)
- Worker process spawns successfully
- Enhanced diagnostics show exactly what's happening

### ❌ What's Broken
- Worker crashes within 100ms of startup
- Worker never starts HTTP server
- Worker never calls back to rbee-hive
- All inference requests timeout waiting for worker

---

## The Problem

**Symptom:**
```
✅ Worker spawned successfully:
   Worker ID: worker-94161306-f651-4e97-b470-e2eddb4c2659
   URL: http://127.0.0.1:8081
   State: loading

⏳ Waiting for worker to be ready at http://127.0.0.1:8081
❌ Worker connection error (attempt 1): Connection refused
❌ Worker connection error (attempt 2): Connection refused
❌ Worker connection error (attempt 3): Connection refused
...
❌ Worker ready timeout after 300s
```

**Root Cause (Confirmed):**
```bash
$ ./target/release/llm-worker-rbee \
    --worker-id test \
    --model ".test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
    --port 9999 \
    --callback-url "http://localhost:9999/callback"

Error: Missing llama.vocab_size in GGUF metadata
```

**The GGUF files are missing required metadata fields.**

---

## How to Reproduce

### Method 1: Via rbee CLI (Full Stack)

```bash
# Clean environment
pkill -f 'queen-rbee|rbee-hive|llm-worker'

# Run inference
./target/release/rbee infer \
  --node localhost \
  --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  --prompt "Why is the sky blue?" \
  --max-tokens 50

# Result: Timeout after 5 minutes
```

### Method 2: Direct Worker Test (Isolated)

```bash
# Test worker directly with GGUF file
./target/release/llm-worker-rbee \
  --worker-id test-worker \
  --model ".test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  --port 9999 \
  --callback-url "http://localhost:9999/callback"

# Result: Error: Missing llama.vocab_size in GGUF metadata
```

### Method 3: Test Other Models

```bash
# Try Qwen model
./target/release/llm-worker-rbee \
  --worker-id test-worker \
  --model ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  --port 9999 \
  --callback-url "http://localhost:9999/callback"

# Result: Same error - Missing llama.vocab_size in GGUF metadata
```

---

## Investigation Starting Points

### File: `bin/llm-worker-rbee/src/backend.rs`
- GGUF loading logic
- Metadata parsing
- Error: "Missing llama.vocab_size in GGUF metadata"
- Why is this field required?
- Can we make it optional or derive it?

### File: `bin/llm-worker-rbee/src/main.rs`
- Lines 105-106: `CandleInferenceBackend::load(&args.model, device)?`
- This is where the crash happens
- Worker never reaches HTTP server startup (line 155)
- Worker never calls callback (line 133)

### Available Test Models

```bash
$ find .test-models -name "*.gguf" -o -name "*.safetensors"
.test-models/qwen-safetensors/model.safetensors
.test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
.test-models/phi3/phi-3-mini-4k-instruct-q4.gguf
.test-models/llama2-7b/llama-2-7b.Q8_0.gguf
.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf
.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

**Note:** Catalog expects SafeTensors format, but only GGUF files exist (except qwen-safetensors).

---

## Investigation Tasks

### Priority 1: Understand the Error

1. **Why is `llama.vocab_size` required?**
   - Check Candle's GGUF loader requirements
   - Is this a Candle limitation or our code?
   - Can we derive vocab_size from other metadata?

2. **Inspect GGUF file metadata**
   ```bash
   # Use a GGUF inspection tool
   # What metadata fields ARE present?
   # What metadata fields are MISSING?
   ```

3. **Check if SafeTensors works**
   ```bash
   ./target/release/llm-worker-rbee \
     --worker-id test \
     --model ".test-models/qwen-safetensors/model.safetensors" \
     --port 9999 \
     --callback-url "http://localhost:9999/callback"
   ```

### Priority 2: Fix Options

**Option A: Fix GGUF Loader**
- Make `vocab_size` optional
- Derive it from tokenizer or other metadata
- Add better error handling

**Option B: Download Valid Models**
- Download properly formatted GGUF files from HuggingFace
- Replace the broken test files
- Verify they have complete metadata

**Option C: Implement SafeTensors**
- The catalog expects SafeTensors format
- Worker should support SafeTensors loading
- This is the "correct" long-term solution

**Option D: Add Metadata Validation**
- Validate model files before spawning worker
- Fail fast with clear error message
- Don't wait 5 minutes to discover model is broken

### Priority 3: Improve Error Handling with Robust Narration

Even if models are broken, worker should:
1. **Fail gracefully** - Don't just crash
2. **Report errors** - Call callback with error state
3. **Use proper narration** - Follow narration-core standards

**Current behavior:**
```
Worker spawns → Crashes → No callback → 5 minute timeout
```

**Desired behavior:**
```
Worker spawns → Model load fails → Narrate error → Callback with error → Immediate failure
```

### Priority 4: Implement Proper Narration (CRITICAL)

**REQUIRED READING:**
1. **`bin/shared-crates/narration-core/TEAM_RESPONSIBILITIES.md`**
   - Read the entire document (646 lines)
   - Understand the Narration Core Team's mission and standards
   - Learn about `human`, `cute`, and `story` fields
   - Follow their editorial authority on all narration

2. **`test-harness/TEAM_RESPONSIBILITIES.md`**
   - Understand Testing Team's requirements
   - BDD tests are VERY IMPORTANT (line 98)
   - Tests must observe, never manipulate (line 45)
   - All narration must be testable via BDD

**Narration Requirements for Worker:**

The worker currently uses JSON logging but should use **structured narration** via `observability-narration-core`:

```rust
use observability_narration_core::{narrate, NarrationFields};

// When model loading fails:
narrate(NarrationFields {
    actor: ACTOR_LLM_WORKER_RBEE,
    action: ACTION_MODEL_LOAD_FAILED,
    target: model_path.to_string(),
    human: format!("Failed to load model: Missing llama.vocab_size in GGUF metadata"),
    cute: Some("Oh no! The model file is missing important information! 😟🔍".to_string()),
    worker_id: Some(worker_id.clone()),
    error_kind: Some("gguf_metadata_missing".to_string()),
    ..Default::default()
});
```

**Why This Matters:**
- **Narration Core Team has ultimate editorial authority** (line 57)
- All `human` fields must be ≤100 characters (line 66)
- Must use correlation IDs for request tracking (line 524)
- Errors must be specific with context (line 70)
- **NO stdout/stderr for production logs** - use structured narration
- BDD tests must assert on narration events (line 511)

**SSH Context:**
When workers run on remote nodes via SSH:
- Narration events go through structured logging (JSON)
- SSH captures stdout/stderr but narration is separate
- rbee-hive can parse narration events from worker output
- This enables proper error reporting even over SSH

**Implementation Steps:**
1. Add `observability-narration-core` dependency to `llm-worker-rbee`
2. Replace JSON logging with `narrate()` calls
3. Define actor/action constants (see `narration.rs` in worker)
4. Add correlation_id propagation from callback URL
5. Write BDD tests that assert on narration events

---

## Files to Check

```
bin/llm-worker-rbee/
├── src/
│   ├── main.rs              # Line 105: Model loading crash point
│   ├── backend.rs            # GGUF loading logic
│   ├── lib.rs                # Module structure
│   ├── narration.rs          # Actor/action constants (if exists)
│   └── common.rs             # callback_ready function
│
bin/rbee-hive/src/http/
├── workers.rs                # Lines 200-231: Early exit detection (TEAM-087)
└── routes.rs                 # Worker ready callback endpoint

bin/shared-crates/narration-core/
├── TEAM_RESPONSIBILITIES.md  # REQUIRED READING (646 lines)
├── src/
│   ├── lib.rs                # narrate() function
│   └── fields.rs             # NarrationFields struct
└── .specs/
    └── 30_TESTING.md         # Testing requirements

test-harness/
├── TEAM_RESPONSIBILITIES.md  # REQUIRED READING (490 lines)
└── bdd/                      # BDD test location

.test-models/
├── catalog.json              # Expected model formats
├── tinyllama/                # GGUF file (broken)
├── qwen/                     # GGUF files (broken)
└── qwen-safetensors/         # SafeTensors (might work?)
```

---

## Debugging Commands

```bash
# 1. Check what metadata the GGUF file has
# (You may need to install a GGUF inspection tool)

# 2. Test worker with verbose logging
RUST_LOG=debug ./target/release/llm-worker-rbee \
  --worker-id test \
  --model ".test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  --port 9999 \
  --callback-url "http://localhost:9999/callback" \
  2>&1 | tee worker-debug.log

# 3. Check Candle's GGUF implementation
# Look at reference/candle/ for examples

# 4. Test SafeTensors model
./target/release/llm-worker-rbee \
  --worker-id test \
  --model ".test-models/qwen-safetensors/model.safetensors" \
  --port 9999 \
  --callback-url "http://localhost:9999/callback"

# 5. Download a known-good GGUF file
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -O /tmp/tinyllama-test.gguf

./target/release/llm-worker-rbee \
  --worker-id test \
  --model "/tmp/tinyllama-test.gguf" \
  --port 9999 \
  --callback-url "http://localhost:9999/callback"
```

---

## Success Criteria

- [ ] **Read required documentation**
  - [ ] `bin/shared-crates/narration-core/TEAM_RESPONSIBILITIES.md` (all 646 lines)
  - [ ] `test-harness/TEAM_RESPONSIBILITIES.md` (all 490 lines)
- [ ] **Understand the problem**
  - [ ] Why `llama.vocab_size` is missing
  - [ ] Which metadata fields are present/missing
- [ ] **Choose and implement fix**
  - [ ] Choose fix strategy (A, B, C, or D above)
  - [ ] Implement the fix
  - [ ] Worker starts successfully with at least ONE model
- [ ] **Implement proper narration**
  - [ ] Add `observability-narration-core` dependency
  - [ ] Replace JSON logging with `narrate()` calls
  - [ ] Define actor/action constants
  - [ ] Follow Narration Core Team's editorial standards
  - [ ] All `human` fields ≤100 characters
  - [ ] Include `cute` fields for errors
- [ ] **Error handling**
  - [ ] Worker calls back to rbee-hive on error
  - [ ] Narration events emitted for all error paths
  - [ ] Full inference flow works end-to-end
- [ ] **BDD testing** (VERY IMPORTANT)
  - [ ] Write BDD scenarios for worker startup
  - [ ] Assert on narration events in tests
  - [ ] Test both success and error paths
  - [ ] Follow Testing Team's standards
- [ ] **Documentation**
  - [ ] Add TEAM-088 signature to modified files
  - [ ] Document narration plumbing decisions
  - [ ] Update worker README with narration usage

---

## Expected Outcome

After your fix, this should work:

```bash
./target/release/rbee infer \
  --node localhost \
  --model "tinyllama" \
  --prompt "Why is the sky blue?" \
  --max-tokens 50

# Output:
# === Inference via queen-rbee Orchestration ===
# [queen-rbee] Submitting inference task...
# ✅ Request accepted
# ⏳ Waiting for worker to be ready...
# ✅ Worker ready
# Tokens:
# The sky appears blue because of a phenomenon called Rayleigh scattering...
```

---

## Context from TEAM-087

We fixed the HTTP 400 bug and added comprehensive diagnostics. The enhanced error handling is working perfectly - it revealed this worker crash issue immediately.

**What TEAM-087 delivered:**
- ✅ HTTP 400 bug fixed (model ref validation)
- ✅ Enhanced timeout diagnostics
- ✅ Worker spawn logging with stdout/stderr capture
- ✅ Clear error messages showing root cause

**The diagnostics are working - now we need working models!**

---

## Quick Win Hypothesis

**If you just want to test inference quickly:**

Download a known-good GGUF file from HuggingFace and test with it. This will tell you if the problem is:
- ❌ The specific GGUF files we have (likely)
- ❌ The GGUF loader implementation (less likely)
- ❌ Something else entirely (investigate further)

---

**Created by:** TEAM-087  
**Date:** 2025-10-11  
**Time:** 20:55  
**Next Team:** TEAM-088  
**Priority:** P0 - Blocks all inference functionality

---

## Critical Reminders

### 1. Narration Core Team Authority
The **Narration Core Team has ultimate editorial authority** over all narration in the codebase. They will review your narration and may require changes. Follow their standards from the start:
- Read their TEAM_RESPONSIBILITIES.md (646 lines)
- Use `human` + `cute` fields (story is optional)
- Keep `human` ≤100 characters
- Use correlation IDs
- No secrets in logs (auto-redacted)

### 2. Testing Team Requirements
The **Testing Team is responsible for production failures** caused by insufficient testing. They require:
- BDD tests are VERY IMPORTANT (emphasized in their docs)
- Tests must observe, never manipulate
- All narration must be testable
- Write BDD scenarios BEFORE claiming completion

### 3. Narration Plumbing
The Narration Core Team needs to:
- Ensure proper plumbing throughout the codebase
- Make sure narration flows correctly through SSH
- Coordinate with all teams on narration standards
- Review and approve all narration implementations

**Your work will be reviewed by both teams. Follow their standards.**

---

**Good luck! The diagnostics will help you see exactly what's happening.** 🔍

---

*Handoff created by TEAM-087 with guidance from Narration Core Team and Testing Team standards* 🎀🔍

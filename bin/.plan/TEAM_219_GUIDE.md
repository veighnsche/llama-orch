# TEAM-219: llm-worker-rbee Behavior Investigation

**Phase:** 1 (Main Binaries)  
**Component:** `30_llm_worker_rbee` - Worker daemon (runs models)  
**Duration:** 1 day  
**Output:** `TEAM_219_LLM_WORKER_BEHAVIORS.md`

---

## Mission

Inventory ALL behaviors in `llm-worker-rbee` daemon to enable comprehensive test coverage.

---

## Investigation Areas

### 1. HTTP API Surface

**Files:**
- `bin/30_llm_worker_rbee/src/main.rs`
- HTTP route handlers

**Tasks:**
- Document ALL HTTP endpoints
- Document request/response schemas
- Document status codes
- Document error responses

**Endpoints to Document:**
- `/health` - Health check
- `/v1/inference` - Inference endpoint
- `/v1/chat/completions` - Chat completions (OpenAI-compatible)
- `/v1/completions` - Completions (OpenAI-compatible)
- `/heartbeat` - Heartbeat to hive
- Any other endpoints

### 2. Model Loading

**Files:**
- Model initialization logic
- llama.cpp integration

**Tasks:**
- Document model loading sequence
- Document model validation
- Document GPU memory allocation
- Document model warmup (if any)
- Document model unloading

**Critical Behaviors:**
- How is model path resolved?
- How is GPU selected?
- How is VRAM allocated?
- What happens if model fails to load?
- What happens if VRAM exhausted?

**Edge Cases:**
- Model file missing
- Model file corrupted
- Insufficient VRAM
- CUDA not available
- Model incompatible with GPU

### 3. Inference Pipeline

**Files:**
- Inference request handling
- llama.cpp integration
- Sampling logic
- Token generation

**Tasks:**
- Document inference request flow
- Document token generation loop
- Document sampling parameters
- Document stopping conditions
- Document response streaming

**Behaviors:**
- How are prompts tokenized?
- How are tokens generated?
- How is sampling configured?
- How are stop sequences handled?
- How is response formatted?

**Edge Cases:**
- Empty prompt
- Very long prompt
- Invalid sampling params
- Generation timeout
- Context length exceeded

### 4. Streaming Responses

**Files:**
- SSE streaming logic
- Token streaming

**Tasks:**
- Document SSE setup
- Document token-by-token streaming
- Document stream completion
- Document stream errors

**Critical:**
- How are tokens streamed?
- How is stream closed?
- What happens if client disconnects?
- How are errors streamed?

### 5. Heartbeat System

**Files:**
- Heartbeat to hive logic

**Tasks:**
- Document heartbeat frequency
- Document heartbeat payload
- Document heartbeat failure handling
- Document registration flow

**Behaviors:**
- When are heartbeats sent?
- What data is included (slots, model, GPU)?
- What happens if hive unreachable?
- How is heartbeat state managed?

### 6. Resource Management

**Files:**
- GPU memory tracking
- Slot management
- Request queuing (if any)

**Tasks:**
- Document GPU memory allocation
- Document slot availability tracking
- Document request queuing
- Document resource cleanup

**Questions:**
- How many concurrent requests?
- How is VRAM tracked?
- How are requests queued?
- How is cleanup done?

### 7. Configuration Management

**Files:**
- Config loading
- Environment variables
- CLI arguments

**Tasks:**
- Document ALL configuration sources
- Document config validation
- Document default values
- Document model-specific config

**Configuration:**
- Model path
- GPU device ID
- Context length
- Batch size
- Thread count
- Hive URL
- Port binding

### 8. Daemon Lifecycle

**Files:**
- `bin/30_llm_worker_rbee/src/main.rs` (startup/shutdown)

**Tasks:**
- Document startup sequence
- Document model loading at startup
- Document graceful shutdown
- Document model unloading on shutdown
- Document signal handling

**Behaviors:**
- Port binding
- Config loading
- Model loading
- HTTP server startup
- Hive registration
- Shutdown cleanup

### 9. Error Handling

**Tasks:**
- Document ALL error types
- Document error responses
- Document error recovery

**Error Categories:**
- Model loading errors
- Inference errors
- Resource exhaustion errors
- Network errors
- Configuration errors

**Edge Cases:**
- Model load failure at startup
- VRAM exhaustion during inference
- Network failure to hive
- Client disconnects during inference

### 10. OpenAI API Compatibility

**Files:**
- `/v1/chat/completions` handler
- `/v1/completions` handler

**Tasks:**
- Document request schema compatibility
- Document response schema compatibility
- Document parameter mapping
- Document unsupported features

**Critical:**
- Which OpenAI parameters are supported?
- Which are ignored?
- How are they mapped to llama.cpp?
- What errors are returned for unsupported features?

---

## Investigation Methodology

### Step 1: Read Main Entry Point
```bash
cat bin/30_llm_worker_rbee/src/main.rs
```

### Step 2: Identify All Modules
```bash
find bin/30_llm_worker_rbee/src -name "*.rs"
```

### Step 3: Check Dependencies
```bash
cat bin/30_llm_worker_rbee/Cargo.toml
```

### Step 4: Look for llama.cpp Integration
```bash
# Look for FFI or bindings
grep -r "llama" bin/30_llm_worker_rbee/src
```

### Step 5: Check Existing Tests
```bash
find bin/30_llm_worker_rbee -name "*test*.rs"
find bin/30_llm_worker_rbee/bdd -name "*.feature"
```

---

## Key Files to Investigate

1. `bin/30_llm_worker_rbee/src/main.rs` - Entry point, server setup
2. `bin/30_llm_worker_rbee/Cargo.toml` - Dependencies
3. Model loading modules
4. Inference modules
5. HTTP route handlers
6. Heartbeat modules
7. llama.cpp integration modules

---

## Expected Behaviors to Document

### HTTP API Behaviors
- [ ] All endpoints documented
- [ ] Request/response schemas
- [ ] OpenAI compatibility
- [ ] Error responses

### Model Loading Behaviors
- [ ] Model path resolution
- [ ] GPU allocation
- [ ] VRAM allocation
- [ ] Validation
- [ ] Failure handling

### Inference Behaviors
- [ ] Request handling
- [ ] Token generation
- [ ] Sampling
- [ ] Stopping conditions
- [ ] Response formatting

### Streaming Behaviors
- [ ] SSE setup
- [ ] Token streaming
- [ ] Stream completion
- [ ] Stream errors

### Heartbeat Behaviors
- [ ] Heartbeat to hive
- [ ] Frequency
- [ ] Payload
- [ ] Failure handling

### Resource Behaviors
- [ ] GPU memory tracking
- [ ] Slot management
- [ ] Request queuing
- [ ] Cleanup

### Configuration Behaviors
- [ ] Config loading
- [ ] Validation
- [ ] Defaults
- [ ] Model configs

### Daemon Behaviors
- [ ] Startup sequence
- [ ] Model loading
- [ ] Graceful shutdown
- [ ] Cleanup

---

## Deliverables Checklist

- [ ] All HTTP endpoints documented
- [ ] Model loading documented
- [ ] Inference pipeline documented
- [ ] Streaming behaviors documented
- [ ] Heartbeat behaviors documented
- [ ] Resource management documented
- [ ] Error handling documented
- [ ] Configuration documented
- [ ] Daemon lifecycle documented
- [ ] OpenAI compatibility documented
- [ ] Existing test coverage assessed
- [ ] Coverage gaps identified
- [ ] Code signatures added (`// TEAM-219: Investigated`)
- [ ] Document follows template
- [ ] Document ≤3 pages
- [ ] Examples include line numbers

---

## Success Criteria

1. ✅ Complete behavior inventory document
2. ✅ All HTTP APIs documented
3. ✅ All model loading documented
4. ✅ All inference behaviors documented
5. ✅ All streaming documented
6. ✅ All resource management documented
7. ✅ Test coverage gaps identified
8. ✅ Code signatures added
9. ✅ No TODO markers in document

---

## Critical Focus Areas

### 1. Model Loading
Most complex startup behavior:
- Path resolution
- GPU selection
- VRAM allocation
- Validation
- Error handling

### 2. Inference Pipeline
Core functionality:
- Request → Tokenize → Generate → Stream → Complete
- All sampling parameters
- All stopping conditions
- All error cases

### 3. Streaming
Critical for UX:
- SSE setup
- Token-by-token streaming
- Completion detection
- Error streaming
- Client disconnect handling

### 4. OpenAI Compatibility
Important for ecosystem:
- Which parameters supported?
- How are they mapped?
- What's ignored?
- Error messages for unsupported features

---

## Next Steps After Completion

1. Hand off to TEAM-242 for test plan creation
2. Document will be used to create:
   - Unit test plan
   - BDD test plan
   - Integration test plan
   - E2E test plan

---

**Status:** READY  
**Blocked By:** None (can start immediately)  
**Blocks:** TEAM-242 (test planning)

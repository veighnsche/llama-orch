# TEAM-035 → TEAM-036 HANDOFF

**Date:** 2025-10-10  
**From:** TEAM-035  
**To:** TEAM-036  
**Status:** SSE Implementation Complete, Model Loading Blocked

---

## 🎯 WHAT WE ACCOMPLISHED (TEAM-035)

### ✅ Phase 1: Download Progress SSE (TEAM-034)
- Already complete from previous team

### ✅ Phase 2: Model Loading Progress SSE
**Files created:**
- `bin/llm-worker-rbee/src/http/loading.rs` (195 lines)
- Updated `bin/llm-worker-rbee/src/http/backend.rs` (added trait methods)
- Updated `bin/llm-worker-rbee/src/http/routes.rs` (added route)

**Endpoint:** `GET /v1/loading/progress`  
**Events:** `LoadingToVram`, `Ready`  
**Tests:** 4 unit tests passing

### ✅ Phase 3: Inference Streaming Refinement
**Files modified:**
- `bin/llm-worker-rbee/src/http/execute.rs` (added [DONE] marker)
- `bin/llm-worker-rbee/src/http/routes.rs` (renamed to /v1/inference)

**Changes:**
- Endpoint renamed: `/execute` → `/v1/inference`
- Added OpenAI-compatible `[DONE]` marker
- Both success and error paths terminate correctly

### ✅ Additional Work
- Renamed all `llorch` → `rbee` references
- Fixed model catalog to use mapped names
- Fixed hostname resolution for localhost testing
- Created comprehensive documentation

**Test Results:**
- 127 lib tests: ✅ PASSING
- 10 integration tests: ✅ PASSING
- Build: ✅ SUCCESS

---

## ❌ WHAT'S BLOCKING YOU

### CRITICAL BLOCKER: Model Loading

**File:** `bin/llm-worker-rbee/src/backend/inference.rs`

**Problem:**
```
Error: Failed to open config.json at ".test-models/tinyllama/config.json"
```

**Root Cause:**
The Candle backend expects:
- Safetensors format (.safetensors files)
- config.json (HuggingFace format)

But we have:
- GGUF format (.gguf file)
- No config.json

**Impact:** Worker can't load models, inference blocked

**Test to verify it's broken:**
```bash
target/debug/llm-worker-rbee --worker-id test-123 \
    --model .test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --port 9999 \
    --callback-url http://localhost:9999/ready

# Fails with: "Failed to open config.json"
```

---

## 🔧 YOUR IMMEDIATE TASKS (TEAM-036)

### Task 1: Add GGUF Support to Worker (CRITICAL)

**Priority:** HIGHEST - Blocks all inference  
**Estimated time:** 2-3 hours  
**File:** `bin/llm-worker-rbee/src/backend/inference.rs`

**What to do:**
1. Detect if model file is GGUF (check extension)
2. Use `candle-transformers::models::quantized::gguf_file` to load
3. Parse model config from GGUF metadata (no config.json needed)
4. Update `CandleInferenceBackend::load()` to handle both formats

**Code hint:**
```rust
use candle_transformers::models::quantized::gguf_file;

pub fn load(model_path: &str, device: Device) -> Result<Self> {
    if model_path.ends_with(".gguf") {
        // Load GGUF file
        let mut file = std::fs::File::open(model_path)?;
        let model = gguf_file::Content::read(&mut file)?;
        // Parse config from GGUF metadata
        // ...
    } else {
        // Existing safetensors loading
        // ...
    }
}
```

**Success criteria:**
```bash
# This should work:
target/debug/llm-worker-rbee --worker-id test-123 \
    --model .test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --port 9999 \
    --callback-url http://localhost:9999/ready

# Should output:
# ✅ Worker starts
# ✅ Loads device
# ✅ Loads GGUF model
# ✅ HTTP server starts on port 9999
```

**Then this works end-to-end:**
```bash
# Terminal 1
RBEE_WORKER_HOST=127.0.0.1 cargo run -p rbee-hive -- daemon --addr 127.0.0.1:8080

# Terminal 2
cargo run --release -p rbee-keeper -- infer \
    --node localhost \
    --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
    --prompt "Once upon a time" \
    --max-tokens 20

# TOKENS STREAM TO YOUR SHELL! 🎉
```

---

### Task 2: Fix Installation Paths (HIGH PRIORITY)

**Priority:** HIGH - Blocks remote deployment  
**Estimated time:** 4-6 hours  
**Files:** Multiple (see spec)

**Problem:**
`bin/rbee-keeper/src/commands/pool.rs` has hardcoded paths:
```rust
"cd ~/Projects/llama-orch && ./target/release/rbee-hive models list"
```

This breaks if:
- User clones to different directory
- Remote machine has different path
- Binaries installed to standard locations

**What to do:**
Read the full spec: `bin/.specs/INSTALLATION_RUST_SPEC.md`

**Summary:**
1. Create `bin/rbee-keeper/src/commands/install.rs` - Rust installation command
2. Create `bin/rbee-keeper/src/config.rs` - Config file loading
3. Update `bin/rbee-keeper/src/commands/pool.rs` - Use binaries from PATH
4. Update `bin/rbee-keeper/src/cli.rs` - Add new commands

**Success criteria:**
```bash
# Install locally
cargo build --release
./target/release/rbee install --user

# Binaries now in ~/.local/bin
which rbee
which rbee-hive

# Remote commands work without hardcoded paths
rbee pool models list --host mac.home.arpa
```

---

### Task 3: Convert rbee-models Script to Rust (MEDIUM)

**Priority:** MEDIUM - Quality of life  
**Estimated time:** 3-4 hours  
**File:** `scripts/rbee-models` (638 lines of bash!)

**Problem:** Shell script for model management (violates NO_SHELL_SCRIPTS policy)

**What to do:**
Convert to Rust subcommands:
```bash
# Instead of:
./scripts/rbee-models download tinyllama

# Should be:
rbee models download tinyllama
rbee models list
rbee models verify tinyllama
```

**Implementation:**
- Create `bin/rbee-keeper/src/commands/models.rs`
- Add `Models` subcommand to CLI
- Use `reqwest` for downloads
- Use `indicatif` for progress bars

---

## 📋 DOCUMENTATION YOU NEED TO READ

### CRITICAL - Read These First

1. **`TEST_RESULTS.md`** - What works, what's broken, test results
2. **`TECHNICAL_DEBT.md`** - All known issues and blockers
3. **`NO_SHELL_SCRIPTS.md`** - Project policy (READ THIS!)
4. **`bin/.specs/INSTALLATION_RUST_SPEC.md`** - Full implementation spec

### Reference Documentation

5. **`bin/.plan/SSE_IMPLEMENTATION_PLAN.md`** - SSE architecture
6. **`bin/.plan/TEAM_034_COMPLETION_SUMMARY.md`** - Phase 1 details
7. **`bin/.specs/.gherkin/test-001-mvp.md`** - Full MVP spec
8. **`QUICKSTART_INFERENCE.md`** - How to run inference

---

## 🧪 HOW TO TEST YOUR WORK

### Test 1: Model Loading (After Task 1)

```bash
# Start worker manually
target/debug/llm-worker-rbee \
    --worker-id test-123 \
    --model .test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --port 9999 \
    --callback-url http://localhost:9999/ready

# Should start successfully and listen on port 9999
curl http://localhost:9999/health
# Should return: {"status":"healthy","model_loaded":true,...}
```

### Test 2: End-to-End Inference (After Task 1)

```bash
# Terminal 1: Start pool manager
RBEE_WORKER_HOST=127.0.0.1 cargo run -p rbee-hive -- daemon --addr 127.0.0.1:8080

# Terminal 2: Run inference
cargo run --release -p rbee-keeper -- infer \
    --node localhost \
    --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
    --prompt "Once upon a time" \
    --max-tokens 20 \
    --temperature 0.7

# Should output:
# === MVP Cross-Node Inference ===
# [Phase 1-7] All pass
# [Phase 8] Tokens:
# Once upon a time, in a small village...
# ✅ Inference complete!
```

### Test 3: Remote Deployment (After Task 2)

```bash
# Install on remote machine
ssh mac.home.arpa
cd ~/llama-orch  # Or ANY directory!
cargo build --release
./target/release/rbee install --user

# From control node
rbee pool models list --host mac.home.arpa
# Should work without hardcoded paths!
```

### Test 4: All Tests Pass

```bash
cargo test -p llm-worker-rbee
cargo test -p rbee-hive
cargo test -p rbee-keeper
cargo build --workspace
```

---

## 🚨 CRITICAL WARNINGS

### 1. NO SHELL SCRIPTS!

**READ:** `NO_SHELL_SCRIPTS.md`

If you're tempted to write a shell script, STOP and write Rust instead.

**Examples:**
- ❌ `install.sh` - Write `rbee install` in Rust
- ❌ `deploy.sh` - Write `rbee deploy` in Rust
- ❌ Any `.sh` file for product features

### 2. Document Technical Debt

If you create ANY shortcuts or workarounds:
- Document in `TECHNICAL_DEBT.md`
- Create spec for proper solution
- Assign to next team

### 3. Follow the Specs

Don't guess or improvise:
- Read `bin/.specs/INSTALLATION_RUST_SPEC.md` for installation
- Read `bin/.specs/.gherkin/test-001-mvp.md` for inference flow
- Follow XDG Base Directory standards

---

## 📁 FILES YOU'LL MODIFY

### Task 1: GGUF Support
- `bin/llm-worker-rbee/src/backend/inference.rs` - Add GGUF loading
- `bin/llm-worker-rbee/Cargo.toml` - May need to add dependencies

### Task 2: Installation Paths
- `bin/rbee-keeper/src/commands/install.rs` - NEW
- `bin/rbee-keeper/src/config.rs` - NEW
- `bin/rbee-keeper/src/commands/pool.rs` - MODIFY (remove hardcoded paths)
- `bin/rbee-keeper/src/cli.rs` - MODIFY (add commands)
- `bin/rbee-keeper/Cargo.toml` - Add dependencies (dirs, toml)

### Task 3: Convert rbee-models
- `bin/rbee-keeper/src/commands/models.rs` - NEW
- `bin/rbee-keeper/src/cli.rs` - MODIFY (add models command)
- `scripts/rbee-models` - DELETE after conversion

---

## 🎯 SUCCESS CRITERIA

### Minimum (To get LLM response)
- [ ] Worker loads GGUF models
- [ ] End-to-end inference works on localhost
- [ ] All tests pass

### Complete (Production ready)
- [ ] Worker loads GGUF models
- [ ] Installation paths fixed
- [ ] `rbee install` command works
- [ ] Remote deployment works
- [ ] `rbee-models` script converted to Rust
- [ ] All tests pass
- [ ] Documentation updated

---

## 🆘 IF YOU GET STUCK

### Model Loading Issues
- Check `reference/candle/` for GGUF examples
- Look at `candle-transformers` documentation
- Test with simple GGUF file first

### Installation Path Issues
- Read `bin/.specs/INSTALLATION_RUST_SPEC.md` carefully
- Use `dirs` crate for XDG paths
- Test on fresh machine

### General Questions
- Read the specs first
- Check existing code for patterns
- Ask the team (don't guess!)

---

## 📊 CURRENT STATE

### What Works ✅
- rbee-keeper CLI
- rbee-hive daemon
- Worker spawning
- Model catalog (SQLite)
- Model provisioner
- SSE streaming (all 3 phases)
- HTTP routing
- Worker registry

### What's Broken ❌
- Worker model loading (GGUF not supported)
- Remote deployment (hardcoded paths)

### Technical Debt 📝
- Hardcoded paths in `pool.rs`
- Shell script `scripts/rbee-models`
- No config file support

---

## 🚀 ESTIMATED TIMELINE

**Minimum viable (localhost inference):**
- Task 1: 2-3 hours
- Testing: 1 hour
- **Total: 3-4 hours**

**Complete (production ready):**
- Task 1: 2-3 hours
- Task 2: 4-6 hours
- Task 3: 3-4 hours
- Testing: 2 hours
- **Total: 11-15 hours**

---

## 📝 HANDOFF CHECKLIST

- [x] SSE Phase 2 complete (loading progress)
- [x] SSE Phase 3 complete (inference streaming)
- [x] All tests passing
- [x] Technical debt documented
- [x] Specs created for next tasks
- [x] NO_SHELL_SCRIPTS policy documented
- [x] Test results documented
- [x] Handoff document created

---

**TEAM-035 SIGNING OFF**

**Status:** Infrastructure 100% ready. Model loading is the ONLY blocker.

**Next:** TEAM-036 - Add GGUF support and fix installation paths.

**Good luck! You're ONE FUNCTION away from LLM responses! 🚀**

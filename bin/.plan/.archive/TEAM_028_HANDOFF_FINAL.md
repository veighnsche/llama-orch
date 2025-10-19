# TEAM-028 Handoff: QA, Testing, and Integration

**Date:** 2025-10-09T23:51:00+02:00  
**From:** TEAM-027  
**To:** TEAM-028  
**Status:** ⚠️ REQUIRES THOROUGH QA - BE SKEPTICAL  
**Priority:** CRITICAL - Verify everything before proceeding

---

## ⚠️ CRITICAL: QA Mindset Required

**TEAM-028: You MUST be skeptical of TEAM-027's work.**

### Your Mission
1. **Question everything** - Don't trust that it works
2. **Test thoroughly** - Verify all claims
3. **Find bugs** - Assume there are issues
4. **Validate architecture** - Check if design makes sense
5. **Challenge decisions** - Ask "why" for everything

**Remember:** TEAM-027 implemented a lot quickly. There WILL be bugs, edge cases, and design issues. Your job is to find them.

---

## What TEAM-027 Claims to Have Built

### 1. rbee-hive Daemon (Pool Manager)

**Claims:**
- ✅ HTTP server on port 8080
- ✅ Worker spawn endpoint
- ✅ Worker ready callback
- ✅ Health monitoring loop (30s)
- ✅ Idle timeout loop (5min)

**Files Created:**
- `bin/rbee-hive/src/commands/daemon.rs`
- `bin/rbee-hive/src/monitor.rs`
- `bin/rbee-hive/src/timeout.rs`

**⚠️ MUST TEST:**
- [ ] Does daemon actually start?
- [ ] Does it bind to port 8080?
- [ ] Can you curl the health endpoint?
- [ ] Does worker spawn actually work?
- [ ] Do background loops actually run?
- [ ] Does graceful shutdown work?
- [ ] What happens if you Ctrl+C?
- [ ] What happens if port 8080 is already in use?
- [ ] Memory leaks in background loops?

**Test Commands:**
```bash
# Start daemon
cargo run --bin rbee-hive -- daemon

# In another terminal
curl http://localhost:8080/v1/health
curl -X POST http://localhost:8080/v1/workers/spawn \
  -H "Content-Type: application/json" \
  -d '{"model_ref":"test","backend":"cpu","device":0,"model_path":"/tmp/test.gguf"}'

# Kill daemon - does it shutdown gracefully?
# Check for zombie processes
ps aux | grep rbee-hive
```

**Known Issues to Verify:**
- Worker binary path detection - does it actually find llm-worker-rbee?
- Port allocation - what if 8081 is taken?
- Callback URL - does it use correct hostname?
- API key generation - are they actually unique?

### 2. rbee-keeper HTTP Client (Orchestrator CLI)

**Claims:**
- ✅ Pool client with health check
- ✅ Worker spawn via HTTP
- ✅ SQLite worker registry
- ✅ 8-phase MVP flow (Phases 1-6)

**Files Created:**
- `bin/rbee-keeper/src/pool_client.rs`
- `bin/shared-crates/worker-registry/src/lib.rs`
- `bin/rbee-keeper/src/commands/infer.rs` (rewritten)

**⚠️ MUST TEST:**
- [ ] Does pool_client actually connect?
- [ ] Does health check work?
- [ ] Does spawn worker work?
- [ ] Does SQLite registry create database?
- [ ] Can you find workers in registry?
- [ ] Can you register workers?
- [ ] What happens if database is corrupted?
- [ ] What happens if pool is unreachable?
- [ ] Timeout handling - does it actually timeout?

**Test Commands:**
```bash
# Test pool client
cargo run --bin rbee -- infer \
  --node localhost \
  --model "test" \
  --prompt "test" \
  --max-tokens 5

# Check SQLite database
ls ~/.rbee/workers.db
sqlite3 ~/.rbee/workers.db "SELECT * FROM workers;"

# Test error cases
# 1. Pool not running
cargo run --bin rbee -- infer --node localhost --model test --prompt test

# 2. Invalid model
cargo run --bin rbee -- infer --node localhost --model "nonexistent" --prompt test
```

**Known Issues to Verify:**
- Phase 7 (worker ready polling) - NOT IMPLEMENTED
- Phase 8 (inference execution) - NOT IMPLEMENTED
- Error handling - are errors actually helpful?
- SQLite connection pooling - does it leak connections?
- Home directory detection - what if $HOME is not set?

### 3. worker-registry Shared Crate

**Claims:**
- ✅ SQLite-backed worker tracking
- ✅ Shared between queen-rbee and rbee-keeper
- ✅ Thread-safe async operations

**Files Created:**
- `bin/shared-crates/worker-registry/src/lib.rs`
- `bin/shared-crates/worker-registry/Cargo.toml`
- `bin/shared-crates/worker-registry/README.md`

**⚠️ MUST TEST:**
- [ ] Does it actually compile?
- [ ] Do tests pass?
- [ ] Is it actually thread-safe?
- [ ] What happens with concurrent access?
- [ ] What happens if database is locked?
- [ ] What happens if schema changes?
- [ ] SQL injection vulnerabilities?

**Test Commands:**
```bash
# Build
cargo build -p worker-registry

# Test
cargo test -p worker-registry

# Stress test (concurrent access)
# TODO: Write a test that hammers it with concurrent requests
```

**Known Issues to Verify:**
- One test is ignored (SQLite in-memory issue)
- No connection pooling
- No migration strategy
- No schema versioning

### 4. Shared Crates Cleanup

**Claims:**
- ✅ Deleted pool-registry-types
- ✅ Deleted orchestrator-core
- ✅ Renamed pool-core → hive-core
- ✅ Kept auth-min for future

**⚠️ MUST VERIFY:**
- [ ] Are deleted crates actually gone?
- [ ] Does hive-core actually work?
- [ ] Are all imports updated?
- [ ] Does rbee-hive still compile?
- [ ] Are there any dangling references?

**Test Commands:**
```bash
# Verify deletions
ls bin/shared-crates/ | grep -E "pool-registry-types|orchestrator-core"
# Should be empty

# Verify rename
ls bin/shared-crates/ | grep hive-core
# Should show hive-core

# Verify no old imports
rg "pool.core|pool_core" --type rust bin/
# Should only show comments, no actual imports

# Build everything
cargo build --workspace
```

---

## What TEAM-027 Did NOT Do

### ❌ Phase 7: Worker Ready Polling
**Status:** Stubbed with TODO  
**Location:** `bin/rbee-keeper/src/commands/infer.rs:97`  
**Impact:** MVP flow incomplete, cannot wait for worker to be ready

### ❌ Phase 8: Inference Execution
**Status:** Stubbed with TODO  
**Location:** `bin/rbee-keeper/src/commands/infer.rs:103`  
**Impact:** MVP flow incomplete, cannot actually run inference

### ❌ End-to-End Testing
**Status:** Test script created but not run  
**Location:** `bin/.specs/.gherkin/test-001-mvp-run.sh`  
**Impact:** No proof that the system works end-to-end

### ❌ Integration Testing
**Status:** No integration tests  
**Impact:** Unknown if components actually work together

### ❌ Error Handling Verification
**Status:** Basic error handling, not tested  
**Impact:** Unknown behavior on errors

### ❌ Performance Testing
**Status:** Not done  
**Impact:** Unknown if background loops cause issues

### ❌ Security Review
**Status:** Not done  
**Impact:** Unknown security vulnerabilities

---

## Required Reading for TEAM-028

**MUST READ BEFORE STARTING:**

1. **Dev Rules (CRITICAL):**
   ```
   /home/vince/Projects/llama-orch/.windsurf/rules/dev-bee-rules.md
   ```
   - NO BACKGROUND TESTING
   - COMPLETE THE ACTUAL TODO LIST
   - CODE SIGNATURE RULES
   - DOCUMENTATION RULES

2. **Test-001 MVP Spec (SOURCE OF TRUTH):**
   ```
   /home/vince/Projects/llama-orch/bin/.specs/.gherkin/test-001-mvp.md
   ```
   - 8-phase flow
   - 10 edge cases
   - Success criteria

3. **TEAM-027 Completion Summary:**
   ```
   /home/vince/Projects/llama-orch/bin/.plan/TEAM_027_COMPLETION_SUMMARY.md
   ```
   - What was built
   - Known limitations
   - TODOs

4. **Shared Crates Analysis:**
   ```
   /home/vince/Projects/llama-orch/bin/shared-crates/CRATE_USAGE_SUMMARY.md
   /home/vince/Projects/llama-orch/bin/shared-crates/CLEANUP_COMPLETED.md
   ```
   - What crates exist
   - What was deleted
   - What was renamed

---

## QA Checklist - TEAM-027's Work

### Build & Compilation

- [ ] `cargo build --workspace` succeeds
- [ ] `cargo build --bin rbee-hive` succeeds
- [ ] `cargo build --bin rbee` succeeds
- [ ] `cargo build -p worker-registry` succeeds
- [ ] `cargo build -p hive-core` succeeds
- [ ] No warnings about unused dependencies
- [ ] No warnings about deprecated features

### Tests

- [ ] `cargo test --workspace` passes
- [ ] `cargo test --bin rbee-hive` passes
- [ ] `cargo test --bin rbee` passes
- [ ] `cargo test -p worker-registry` passes
- [ ] Check ignored tests - why are they ignored?
- [ ] Run tests with `--nocapture` to see output
- [ ] Check for flaky tests (run multiple times)

### Code Quality

- [ ] `cargo clippy --workspace` has no errors
- [ ] `cargo fmt --check` passes
- [ ] No `unwrap()` or `expect()` in production code
- [ ] Proper error handling with `Result<T, E>`
- [ ] No `println!` - should use `tracing`
- [ ] Team signatures present (TEAM-027)
- [ ] Comments explain WHY, not WHAT

### Documentation

- [ ] README files exist and are accurate
- [ ] Cargo.toml descriptions are correct
- [ ] Code comments are helpful
- [ ] No contradictory documentation
- [ ] TODOs are clearly marked
- [ ] Handoff documents are complete

### Architecture

- [ ] Does the design make sense?
- [ ] Are abstractions appropriate?
- [ ] Is there unnecessary complexity?
- [ ] Are there simpler alternatives?
- [ ] Does it follow existing patterns?
- [ ] Is it consistent with llm-worker-rbee?

### Functionality

- [ ] rbee-hive daemon starts
- [ ] Health endpoint responds
- [ ] Worker spawn endpoint works
- [ ] Worker ready callback works
- [ ] Background loops run
- [ ] Graceful shutdown works
- [ ] rbee-keeper can connect to pool
- [ ] SQLite registry works
- [ ] Phase 1-6 of MVP work

### Error Handling

- [ ] Network errors handled
- [ ] Timeout errors handled
- [ ] Database errors handled
- [ ] File system errors handled
- [ ] Invalid input handled
- [ ] Error messages are helpful
- [ ] No panics on bad input

### Edge Cases

- [ ] Port already in use
- [ ] Database file missing
- [ ] Database file corrupted
- [ ] Pool unreachable
- [ ] Worker binary not found
- [ ] Invalid model path
- [ ] Concurrent requests
- [ ] Ctrl+C during operation

---

## Integration Opportunities

**TEAM-028: Check if we can wire up existing crates to current implementation**

### Potential Integrations

#### 1. auth-min → rbee-hive
**Question:** Should rbee-hive use auth-min for worker authentication?

**Check:**
- Does rbee-hive need to authenticate workers?
- Does llm-worker-rbee support Bearer tokens?
- Should we add `--api-key` validation?

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` - Add auth middleware
- `bin/rbee-hive/Cargo.toml` - Add auth-min dependency

**Test:**
```rust
use auth_min::{parse_bearer, timing_safe_eq};

// In worker spawn handler
let auth_header = req.headers().get("Authorization");
let token = parse_bearer(auth_header)?;
// Validate token
```

#### 2. secrets-management → rbee-keeper
**Question:** Should rbee-keeper use secrets-management for API keys?

**Check:**
- Where does rbee-keeper store API keys?
- Currently hardcoded "api-key" - should load from file?
- Should use `Secret::load_from_file()`?

**Files to modify:**
- `bin/rbee-keeper/src/pool_client.rs` - Load API key securely
- `bin/rbee-keeper/Cargo.toml` - Add secrets-management dependency

**Test:**
```rust
use secrets_management::Secret;

// Load API key
let api_key = Secret::load_from_file("~/.rbee/api-key")?;
let client = PoolClient::new(base_url, api_key);
```

#### 3. audit-logging → rbee-hive
**Question:** Should rbee-hive log worker events to audit log?

**Check:**
- What events should be audited?
- Worker spawn, ready, shutdown?
- Should use structured audit logs?

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` - Add audit logging
- `bin/rbee-hive/Cargo.toml` - Add audit-logging dependency

#### 4. input-validation → rbee-hive
**Question:** Should rbee-hive validate spawn requests?

**Check:**
- Validate model_ref format?
- Validate backend (cpu, cuda, metal)?
- Validate device number?
- Validate model_path (no path traversal)?

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` - Add validation
- `bin/rbee-hive/Cargo.toml` - Add input-validation dependency

#### 5. deadline-propagation → queen-rbee
**Question:** Will queen-rbee need deadline propagation?

**Check:**
- Does queen-rbee need request timeouts?
- Should timeouts propagate to rbee-hive?
- Should timeouts propagate to workers?

**Files to check:**
- `bin/queen-rbee/` - Not yet implemented
- Design decision needed

#### 6. gpu-info → rbee-hive
**Question:** Should rbee-hive detect GPU capabilities?

**Check:**
- Does rbee-hive need to know available GPUs?
- Should it validate device numbers?
- Should it detect backends (cuda, metal)?

**Files to modify:**
- `bin/rbee-hive/src/commands/daemon.rs` - Detect GPUs on startup
- `bin/rbee-hive/Cargo.toml` - Add gpu-info dependency

**Test:**
```rust
use gpu_info::detect_gpus;

// On daemon startup
let gpus = detect_gpus()?;
println!("Available GPUs: {:?}", gpus);
```

---

## Testing Strategy for TEAM-028

### Phase 1: Verify TEAM-027's Claims (1-2 hours)

**Goal:** Prove or disprove that TEAM-027's code works

1. **Build everything:**
   ```bash
   cargo clean
   cargo build --workspace
   ```

2. **Run all tests:**
   ```bash
   cargo test --workspace -- --nocapture
   ```

3. **Manual smoke tests:**
   ```bash
   # Start rbee-hive
   cargo run --bin rbee-hive -- daemon &
   
   # Health check
   curl http://localhost:8080/v1/health
   
   # Try to spawn worker (will fail - no model)
   curl -X POST http://localhost:8080/v1/workers/spawn \
     -H "Content-Type: application/json" \
     -d '{"model_ref":"test","backend":"cpu","device":0,"model_path":"/tmp/test.gguf"}'
   
   # Kill daemon
   pkill rbee-hive
   ```

4. **Check for issues:**
   - Does it crash?
   - Are there error messages?
   - Does it leak resources?
   - Are there zombie processes?

### Phase 2: Implement Phase 7-8 (4-6 hours)

**Goal:** Complete the MVP flow

1. **Implement worker ready polling** (2-3 hours)
   - See TEAM_028_HANDOFF.md for template
   - Test with mock worker
   - Handle timeouts

2. **Implement inference execution** (2-3 hours)
   - See TEAM_028_HANDOFF.md for template
   - Test SSE streaming
   - Handle errors

### Phase 3: Integration Testing (2-3 hours)

**Goal:** Prove end-to-end flow works

1. **Run test-001-mvp-run.sh**
2. **Fix any issues**
3. **Document results**

### Phase 4: Wire Up Shared Crates (2-4 hours)

**Goal:** Integrate existing infrastructure

1. **Evaluate each integration opportunity**
2. **Implement high-value integrations**
3. **Test thoroughly**

---

## Red Flags to Watch For

### 🚩 Code Smells

- [ ] Hardcoded values (ports, paths, keys)
- [ ] Unwrap/expect in production code
- [ ] println! instead of tracing
- [ ] No error handling
- [ ] Copy-pasted code
- [ ] Magic numbers
- [ ] TODO comments without issues
- [ ] Dead code

### 🚩 Architecture Smells

- [ ] Circular dependencies
- [ ] God objects
- [ ] Tight coupling
- [ ] No abstraction
- [ ] Over-abstraction
- [ ] Inconsistent patterns
- [ ] Violates existing conventions

### 🚩 Testing Smells

- [ ] No tests
- [ ] Tests that always pass
- [ ] Tests that don't test anything
- [ ] Ignored tests without explanation
- [ ] Flaky tests
- [ ] Tests that require manual setup
- [ ] Tests that don't clean up

### 🚩 Documentation Smells

- [ ] No documentation
- [ ] Outdated documentation
- [ ] Contradictory documentation
- [ ] Documentation that lies
- [ ] Too much documentation
- [ ] Documentation that repeats code

---

## Questions to Ask

### About Design

1. Why was this approach chosen?
2. What alternatives were considered?
3. What are the tradeoffs?
4. Is this the simplest solution?
5. Does this follow existing patterns?
6. Will this scale?
7. Is this maintainable?

### About Implementation

1. Why is this code here?
2. What happens if this fails?
3. What are the edge cases?
4. Is this thread-safe?
5. Is this tested?
6. Can this be simpler?
7. Is this necessary?

### About Testing

1. How do we know this works?
2. What could go wrong?
3. What are we not testing?
4. How do we test this manually?
5. What are the failure modes?
6. How do we reproduce bugs?
7. What's the blast radius?

---

## Success Criteria for TEAM-028

### Minimum (Verify TEAM-027's Work)

- [ ] All builds pass
- [ ] All tests pass
- [ ] Manual smoke tests pass
- [ ] No obvious bugs found
- [ ] Documentation is accurate

### Target (Complete MVP)

- [ ] Phase 7 implemented and tested
- [ ] Phase 8 implemented and tested
- [ ] End-to-end test passes
- [ ] At least 2 shared crates integrated
- [ ] All edge cases handled

### Stretch (Production Ready)

- [ ] All 10 edge cases from test-001 handled
- [ ] All shared crates evaluated for integration
- [ ] Comprehensive error handling
- [ ] Performance tested
- [ ] Security reviewed

---

## Final Notes from TEAM-027

### What We're Confident About

- ✅ Basic HTTP infrastructure works
- ✅ SQLite registry works
- ✅ Shared crate refactoring is clean
- ✅ Code compiles and basic tests pass

### What We're Uncertain About

- ⚠️ Worker spawn - not tested with real worker
- ⚠️ Background loops - not stress tested
- ⚠️ Error handling - not tested thoroughly
- ⚠️ Edge cases - not tested at all
- ⚠️ Integration - not tested end-to-end

### What We Know Is Incomplete

- ❌ Phase 7 - worker ready polling
- ❌ Phase 8 - inference execution
- ❌ End-to-end testing
- ❌ Integration with shared crates
- ❌ Production hardening

### Advice for TEAM-028

1. **Be skeptical** - Don't trust our claims
2. **Test everything** - Assume bugs exist
3. **Read the specs** - test-001-mvp.md is truth
4. **Follow the rules** - dev-bee-rules.md is law
5. **Ask questions** - Challenge our decisions
6. **Find bugs** - They're there, find them
7. **Improve it** - Make it better

**Remember:** Your job is QA first, implementation second. Find the bugs before they find you.

---

**Signed:** TEAM-027  
**Date:** 2025-10-09T23:51:00+02:00  
**Status:** ⚠️ REQUIRES QA  
**Next Team:** TEAM-028 - Be skeptical, test thoroughly, find bugs! 🔍

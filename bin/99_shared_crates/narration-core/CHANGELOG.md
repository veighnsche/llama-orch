# Changelog

All notable changes to narration-core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2025-10-26 — TEAM-297/298/299/300/301/302/303 Privacy & API Evolution ✅

### Breaking Changes ⚠️

**TEAM-299: Privacy Fix (CRITICAL)**
- **Removed**: Global stderr output from narration-core
- **Why**: Multi-tenant privacy violation - User A could see User B's data
- **Impact**: Narration now only goes to SSE (job-scoped) or capture adapter (tests)
- **Migration**: Daemons no longer print to stderr. Keeper displays via separate subscription.

### Added

**TEAM-297: Ultra-Concise `n!()` Macro API**
- ✅ New `n!()` macro - reduces narration from 5 lines to 1 line
- ✅ Runtime-configurable narration modes (Human/Cute/Story)
- ✅ Full Rust `format!()` support (width, precision, debug, hex)
- ✅ Mode selection: `set_narration_mode(NarrationMode::Cute)`
- ✅ All 3 narration modes now functional (cute/story were previously unusable)

**TEAM-298: SSE Optional**
- ✅ SSE delivery is now opportunistic (not required)
- ✅ Added `try_send()` methods that return bool
- ✅ Narration never fails (stdout always works)
- ✅ Order independence (can narrate before channel creation)

**TEAM-300: Process Stdout Capture**
- ✅ Created `ProcessNarrationCapture` system (350 LOC)
- ✅ Captures child process stdout/stderr
- ✅ Parses narration events using regex
- ✅ Re-emits with job_id for SSE routing
- ✅ Worker startup narration now visible to clients!

**TEAM-301: Keeper Lifecycle Management**
- ✅ Created process output streaming utilities (92 LOC)
- ✅ Keeper can display daemon startup output in real-time
- ✅ Background tasks stream stdout/stderr to terminal

**TEAM-302: Test Harness Infrastructure**
- ✅ Created `NarrationTestHarness` (177 LOC)
- ✅ Created SSE testing utilities (217 LOC)
- ✅ 11 integration tests for job-server
- ✅ Reusable test infrastructure

**TEAM-303: E2E Integration Tests**
- ✅ Lightweight HTTP server for E2E testing (268 LOC)
- ✅ 5 E2E tests using real components
- ✅ No fake binaries needed (66% less code)
- ✅ Fast and reliable (0.67s for all tests)

### Fixed

**TEAM-299: Privacy Violation (CRITICAL)**
- ❌ **Before**: `eprintln!()` in narration-core leaked data across users
- ✅ **After**: Complete removal of stderr output
- ✅ Multi-tenant isolation verified
- ✅ GDPR/SOC 2 compliance
- ✅ 10 privacy tests added

**TEAM-298: SSE Resilience**
- ❌ **Before**: Channel creation order mattered, silent failures
- ✅ **After**: Order doesn't matter, explicit success/failure
- ✅ Graceful degradation when SSE unavailable

### Architecture

**Narration Flow:**
1. **Worker → Hive** (TEAM-300: ProcessNarrationCapture)
2. **Hive → SSE → Client** (existing job-server)
3. **Daemon → Keeper → Terminal** (TEAM-301: process_utils)

**Security Model:**
- SSE is job-scoped (isolated per user)
- No global stderr (privacy by design)
- Fail-fast security (no job_id = dropped)

### Migration Guide

**Using the new `n!()` macro:**
```rust
// Before (5 lines):
NARRATE.action("worker_spawn")
    .context(&worker_id)
    .context(&device)
    .human("Spawning worker {} on device {}")
    .emit();

// After (1 line):
n!("worker_spawn", "Spawning worker {} on device {}", worker_id, device);
```

**All 3 narration modes:**
```rust
n!("deploy",
    human: "Deploying service {}",
    cute: "🚀 Launching {} into the cloud!",
    story: "The orchestrator whispered to {}: 'Time to fly'",
    service_name
);
```

**Runtime mode selection:**
```rust
use observability_narration_core::{set_narration_mode, NarrationMode};

set_narration_mode(NarrationMode::Cute);
// All narration now shows cute version (or falls back to human)
```

### Test Coverage

- TEAM-297: 22 macro tests ✅
- TEAM-298: 14 SSE optional tests ✅
- TEAM-299: 10 privacy tests ✅
- TEAM-300: 15 process capture tests ✅
- TEAM-301: 8 keeper process tests ✅
- TEAM-302: 11 job-server integration tests ✅
- TEAM-303: 5 E2E integration tests ✅
- **Total: 85+ new tests**

### Team Credits

- **TEAM-297**: Ultra-concise `n!()` macro API
- **TEAM-298**: SSE optional delivery
- **TEAM-299**: Privacy fix (critical security)
- **TEAM-300**: Process stdout capture
- **TEAM-301**: Keeper lifecycle management
- **TEAM-302**: Test harness infrastructure
- **TEAM-303**: E2E integration tests

---

## [0.6.0] - 2025-10-26 — TEAM-304/305/306/308 Architecture Fixes ✅

### Breaking Changes ⚠️

**TEAM-304: [DONE] Signal Architecture Fix**
- **Removed**: `n!("done", "[DONE]")` from narration-core
- **Why**: [DONE] signal is a lifecycle event, not an observability event
- **Impact**: narration-core no longer emits [DONE] - job-server does this now
- **Migration**: If you were emitting [DONE] via narration, stop. Use job-server's `execute_and_stream()` instead.

### Added

**TEAM-305: job-registry-interface Crate**
- ✅ Created `job-registry-interface` crate to break circular dependency
- ✅ JobRegistryInterface trait for test binaries
- ✅ Clean dependency graph (no more circular dependencies)
- ✅ Test binaries can now use real JobRegistry

**TEAM-308: Test Fixes**
- ✅ Fixed hanging e2e_job_client_integration tests
- ✅ Added explicit SSE channel cleanup after narration completes
- ✅ Fixed incorrect serialization test expectations
- ✅ Removed deprecated integration.rs (373 lines)

### Fixed

**TEAM-304: Separation of Concerns**
- ❌ **Before**: narration-core emitted [DONE] signal (WRONG - mixing observability with lifecycle)
- ✅ **After**: job-server emits [DONE] when channel closes (CORRECT - proper separation)
- ✅ Test binaries updated to emit [DONE] at transport layer
- ✅ All production code uses job-server's execute_and_stream()

**TEAM-305: Circular Dependency**
- ❌ **Before**: job-server → narration-core → job-server (circular!)
- ✅ **After**: job-registry-interface breaks the cycle
- ✅ Both job-server and narration-core depend on interface
- ✅ Clean architecture, future-proof design

**TEAM-306: Context Propagation**
- ✅ Verified 17 existing tests in thread_local_context_tests.rs
- ✅ Context propagates through nested tasks, await points, channels
- ✅ Job isolation verified
- ✅ Correlation IDs flow end-to-end
- ✅ Deep nesting works (5+ levels)
- ✅ Concurrent contexts don't interfere

**TEAM-308: Test Suite**
- ✅ Fixed hanging tests (e2e_job_client_integration.rs)
- ✅ Fixed incorrect test (test_payload_serialization_errors)
- ✅ Deleted deprecated test file (integration.rs)
- ✅ 100% test pass rate achieved (180/180 tests passing)
- ✅ All shared crates compile independently

### Architecture

**Separation of Concerns Restored:**
- **job-server**: Manages job lifecycle, emits [DONE]/[ERROR] signals
- **narration-core**: Handles observability events, SSE channel management
- **job-registry-interface**: Shared trait, breaks circular dependency

**Test Coverage:**
- narration-core: 106/106 tests passing ✅
- job-server: 74/74 tests passing ✅
- Total: 180/180 tests passing ✅

### Migration Guide

**If you were emitting [DONE] via narration:**
```rust
// ❌ OLD (WRONG)
n!("done", "[DONE]");

// ✅ NEW (CORRECT)
// Don't emit [DONE] yourself - job-server does this automatically
// when you use execute_and_stream()
```

**If you need JobRegistry in tests:**
```rust
// ✅ NEW - Use the interface
use job_registry_interface::JobRegistryInterface;
use job_server::JobRegistry;

let registry = Arc::new(JobRegistry::<String>::new());
// Registry implements JobRegistryInterface trait
```

### Team Credits

- **TEAM-304**: [DONE] signal architecture fix (4 hours)
- **TEAM-305**: Circular dependency fix (30 minutes)
- **TEAM-306**: Context propagation verification (1 hour)
- **TEAM-308**: Test fixes and 100% pass rate (2 hours)

---

## [0.5.0] - 2025-10-21 — TEAM-192 Fixed-Width Format 📏

### Breaking Changes ⚠️

1. **Output Format Changed** - Fixed 30-character prefix for perfect column alignment
   - **Old**: `[actor                ] message`
   - **New**: `[actor     ] action         : message`
   - **Impact**: Messages always start at column 31, much easier to scan logs
   - **Migration**: No code changes needed, only output format changes

2. **Actor Length Limit** - Max 10 characters (compile-time enforced)
   - **Old**: 20 characters
   - **New**: 10 characters
   - **Impact**: Use short actor names like `"keeper"`, `"queen"`, `"qn-router"`
   - **Migration**: Shorten actor strings, remove emojis if needed

3. **Action Length Limit** - Max 15 characters (runtime enforced)
   - **New**: Runtime validation with clear error messages
   - **Impact**: Use concise action names like `"queen_start"`, `"job_submit"`

4. **Method Renamed** - `.narrate()` → `.action()`
   - **Why**: More semantic and clearer
   - **Impact**: Update all calls from `NARRATE.narrate("action")` to `NARRATE.action("action")`
   - **Migration**: Simple find-and-replace

### Added

- 📏 **Fixed-Width Format** - 30-char prefix ensures perfect column alignment
  - Format: `[{actor:<10}] {action:<15}: {message}`
  - Actor: 10 chars (left-aligned, space-padded)
  - Action: 15 chars (left-aligned, space-padded)
  - Total prefix: 30 chars (including brackets, spaces, colon)

- 🔒 **Compile-Time Actor Validation** - Actor length checked at compile time
  - Prevents runtime errors
  - Clear error messages if actor exceeds 10 chars
  - Uses const fn character counting (Unicode-aware)

- ✅ **Runtime Action Validation** - Action length checked at runtime
  - Panics with clear message if action exceeds 15 chars
  - Helps catch mistakes early in development

- 🎯 **Renamed Method** - `.action()` replaces `.narrate()`
  - More semantic: "perform an action" vs "narrate something"
  - Clearer intent in code

### Improved

- **Log Readability** - Fixed-width format makes logs dramatically easier to scan
- **Error Messages** - Clear, actionable messages for length violations
- **API Clarity** - `.action()` method name better reflects purpose

### Example

```rust
// Define factory with short actor (≤10 chars)
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

// Use .action() instead of .narrate()
NARRATE.action("queen_start")
    .context("http://localhost:8500")
    .human("Starting queen on {}")
    .emit();

// Output:
// [keeper    ] queen_start    : Starting queen on http://localhost:8500
```

### Migration Guide

1. **Shorten actors to ≤10 chars:**
   - `"🧑‍🌾 rbee-keeper"` → `"keeper"`
   - `"keeper/queen-life"` → `"kpr-life"`
   - `"👑 queen-rbee"` → `"queen"`
   - `"queen-router"` → `"qn-router"`

2. **Rename method calls:**
   - `.narrate("action")` → `.action("action")`

3. **Verify action lengths ≤15 chars:**
   - Most existing actions already comply
   - Shorten if needed: `"queen_status_check"` → `"queen_status"`

---

## [0.4.0] - 2025-10-21 — TEAM-191 Format & Factory 🏭

### Breaking Changes ⚠️
- **Output Format Changed**: Actor now appears inline on same line (not on separate line)
  - **Old**: `[actor]\n  message`
  - **New**: `[actor                ] message` (20-char column alignment)
  - **Impact**: Better readability, consistent column alignment for messages
  - **Migration**: No code changes needed, only output format changes

### Added
- 🏭 **NarrationFactory** - Define default actor once per crate, reuse everywhere
  - `const fn new(actor: &'static str)` - Create factory
  - `.narrate(action, target)` - Create narration with default actor
  - `.actor()` - Get factory's default actor
  - Reduces boilerplate significantly
  - Compile-time constant support
- 📏 **Column Alignment** - Actor field padded to 20 chars for consistent message alignment
- ✨ **Improved Readability** - Messages start at same column, easier to scan logs

### Improved
- **Log Scanning** - Consistent column alignment makes it easier to scan through logs
- **Crate-level Consistency** - Factory pattern ensures same actor used throughout crate

### Example
```rust
// Define at module/crate level
const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);

// Use throughout the crate
NARRATE.narrate(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();
```

---

## [0.3.0] - 2025-10-21 — TEAM-191 Upgrade 🎀

### Added
- 📊 **Table Formatting Documentation** - Comprehensive guide for `.table()` method with examples
- 👑 **Queen-Rbee Actor Constants** - `ACTOR_QUEEN_RBEE`, `ACTOR_QUEEN_ROUTER` (with cute emojis!)
- 🎯 **Job Routing Action Constants** - `ACTION_ROUTE_JOB`, `ACTION_PARSE_OPERATION`, `ACTION_JOB_CREATE`
- 🏠 **Hive Management Action Constants** - `ACTION_HIVE_INSTALL`, `ACTION_HIVE_UNINSTALL`, `ACTION_HIVE_START`, `ACTION_HIVE_STOP`, `ACTION_HIVE_STATUS`, `ACTION_HIVE_LIST`
- 🚀 **System Action Constants** - `ACTION_STATUS`, `ACTION_START`, `ACTION_LISTEN`, `ACTION_READY`, `ACTION_ERROR`
- 😊 **Emoji Support Validation** - Confirmed emojis work perfectly in human fields
- 📝 **Multi-line Message Support** - Long formatted messages with newlines fully supported

### Improved
- 📚 **README** - Table formatting prominently documented with examples and output
- 🎯 **Taxonomy** - Extended with 15+ new action constants for queen-rbee operations
- 🎀 **Editorial Quality** - Queen-rbee usage patterns reviewed and approved (⭐⭐⭐⭐⭐)

### Fixed
- None! Everything was already working! 🎉

### What We Learned
- Table formatting is AMAZING for status displays and lists!
- Emojis make debugging delightful! Keep using them! 🎀
- Multi-line messages work great for complex operations!
- Queen-rbee team writes excellent narrations! 💝

## [0.2.0] - 2025-10-04

### Added
- **Builder Pattern API** - Ergonomic fluent API reduces boilerplate by 43%
- **Axum Middleware** - Built-in correlation ID middleware with auto-extraction/generation
- **Comprehensive Documentation** - Policy guide, field reference, troubleshooting sections
- **Code Quality** - Reduced duplication from ~400 lines to ~90 lines (78% reduction)

### Testing & Quality
- **100% Functional Test Pass Rate** - 75/75 tests passing (50 unit + 16 integration + 9 property)
- **Zero Flaky Tests** - Fixed global state issues with improved `CaptureAdapter`
- **Property-Based Tests** - Comprehensive invariant testing for security & correctness
- **Comprehensive Specification** - 42 normative requirements (NARR-1001..NARR-8005)

### Features
- **7 Logging Levels** - MUTE, TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- **6 Secret Patterns** - Bearer tokens, API keys, JWT, private keys, URL passwords, UUIDs
- **Auto-Injection** - `narrate_auto()` automatically adds provenance metadata
- **Correlation ID Helpers** - Generate, validate (<100ns), extract from headers
- **HTTP Context Propagation** - Extract/inject correlation IDs from HTTP headers
- **Unicode Safety** - ASCII fast path, CRLF sanitization, zero-width character filtering
- **ReDoS-Safe Redaction** - Bounded quantifiers with `OnceLock` caching

## [0.1.0] - 2025-09-15

### Added
- Initial release
- Basic narration functionality with actor/action/target taxonomy
- Secret redaction (Bearer tokens, API keys)
- Correlation ID support
- Test capture adapter for BDD tests
- Human-readable descriptions
- Cute mode (children's book narration) 🎀
- Story mode (dialogue-based narration) 🎭

### Core Features
- Structured logging with `tracing` backend
- Automatic secret redaction
- Correlation ID propagation
- Test capture adapter
- OpenTelemetry integration (optional)

---

**Maintained by**: The Narration Core Team 🎀  
**License**: GPL-3.0-or-later

*May your logs be readable, your correlation IDs present, and your debugging experience absolutely DELIGHTFUL! 🎀✨*

# rbee Architecture Documentation

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Complete  
**Purpose:** Comprehensive architectural overview for engineering teams

---

## Document Structure

This directory contains a complete architectural overview split into 6 parts:

### [Part 1: System Design](00_OVERVIEW_PART_1.md)

**Topics:**
- Mission & Philosophy
- The Four-Binary System
- Intelligence Hierarchy
- Communication Patterns
- **Queen Build Configurations** (NEW: distributed vs integrated)
- Deployment Topologies
- Key Architectural Decisions

**Key Insights:**
- Smart/Dumb architecture (queen decides, workers execute)
- Process isolation (each worker in separate process)
- Job-based architecture (all operations are jobs with SSE streams)
- Daemon vs CLI separation (daemons for data, CLIs for control)
- **Two queen modes:** Distributed (HTTP) vs Integrated (local-hive feature)

**Read this first** to understand the overall system design.

---

### [Part 2: Component Deep Dive](01_COMPONENTS_PART_2.md)

**Topics:**
- rbee-keeper (CLI - User Interface)
  - **NEW:** Queen lifecycle management
  - **NEW:** Smart prompts for localhost optimization
- queen-rbee (HTTP Daemon - The Brain)
  - **NEW:** Distributed vs Integrated build modes
  - **NEW:** Dual-mode hive forwarder (HTTP or direct)
- rbee-hive (HTTP Daemon - Pool Manager)
- llm-worker-rbee (HTTP Daemon - Executor)
- Inter-Component Communication

**Key Insights:**
- rbee-keeper is the PRIMARY user interface (not a testing tool)
- **NEW:** rbee-keeper manages queen builds and prompts for optimization
- queen-rbee makes ALL intelligent decisions
- **NEW:** Queen can embed hive logic (local-hive feature) for 50-100x faster localhost ops
- rbee-hive manages worker lifecycle on a single machine
- llm-worker-rbee is a dumb executor (stateless)
- TEAM-261: Hive heartbeat removed, workers send directly to queen

**Read this second** to understand each component in detail.

---

### [Part 3: Shared Infrastructure](02_SHARED_INFRASTRUCTURE_PART_3.md)

**Topics:**
- Job Client/Server Pattern
- Observability (Narration System)
- Security Crates (5 total)
- Configuration Management (Cross-Platform)
- Shared Utilities

**Key Insights:**
- job-server + job-client provide unified job submission pattern
- Narration requires `.job_id()` for SSE routing (CRITICAL!)
- 5 security crates for defense-in-depth
- File-based configuration (not environment variables)
- **Cross-platform support:** Linux, macOS, Windows with platform-appropriate directories

**Read this third** to understand shared infrastructure.

---

### [Part 4: Data Flow & Protocols](03_DATA_FLOW_PART_4.md)

**Topics:**
- Request Flow Patterns (3 types)
- SSE Streaming Protocol
- **SSE Routing in Integrated Mode** (NEW - verified correct!)
- Heartbeat Architecture
- Operation Routing Logic

**Key Insights:**
- Pattern 1: Hive lifecycle (queen-handled, no HTTP forwarding)
- Pattern 2: Worker/model ops (hive-forwarded, 2 HTTP hops)
- Pattern 3: Inference (direct to worker, queen circumvents hive)
- Dual-call pattern: POST /v1/jobs, GET /v1/jobs/{job_id}/stream
- TEAM-261: Workers → queen direct (no hive aggregation)
- **NEW:** SSE routing works correctly in integrated mode (same process = same SSE_BROADCASTER)

**Read this fourth** to understand data flow and communication.

**Also see:** 
- [SSE_ROUTING_ANALYSIS.md](SSE_ROUTING_ANALYSIS.md) - Deep dive on integrated mode SSE routing
- [CLEANUP_PLAN_TEAM261.md](CLEANUP_PLAN_TEAM261.md) - Post-TEAM-261 cleanup tasks

---

### [Part 5: Development Patterns](04_DEVELOPMENT_PART_5.md)

**Topics:**
- Crate Structure
- BDD Testing Strategy
- Character-Driven Development
- Code Organization Principles
- Development Workflow

**Key Insights:**
- Binary-thin, crate-fat (logic in crates, not binaries)
- BDD testing with Gherkin scenarios
- 6 AI teams with distinct personalities
- TEAM-XXX attribution pattern
- SSE-first design (all operations provide real-time feedback)

**Read this fifth** to understand development practices.

---

### [Part 6: Security & Compliance](05_SECURITY_PART_6.md)

**Topics:**
- Defense in Depth Strategy
- 5 Security Crates (detailed)
- Threat Model
- GDPR Compliance
- Security Best Practices

**Key Insights:**
- 5-layer defense-in-depth: auth-min, audit-logging, input-validation, secrets-management, deadline-propagation
- Timing-safe authentication (prevent timing attacks)
- Immutable audit trail with blockchain-style hash chain
- GDPR compliance (7-year retention, data subject rights)
- Security rating: A- (production-ready)

**Read this sixth** to understand security architecture.

---

## Quick Reference

### Key Files by Component

**rbee-keeper (CLI):**
- `bin/00_rbee_keeper/src/main.rs` - CLI entry point
- `bin/00_rbee_keeper/src/job_client.rs` - HTTP client
- `bin/99_shared_crates/job-client/` - Shared client library

**queen-rbee (Brain):**
- `bin/10_queen_rbee/src/main.rs` - HTTP server
- `bin/10_queen_rbee/src/job_router.rs` - Operation routing
- `bin/10_queen_rbee/src/hive_forwarder.rs` - Forward to hive
- `bin/15_queen_rbee_crates/hive-lifecycle/` - Hive operations

**rbee-hive (Pool Manager):**
- `bin/20_rbee_hive/src/main.rs` - HTTP server
- `bin/20_rbee_hive/src/job_router.rs` - Operation routing
- `bin/25_rbee_hive_crates/device-detection/` - GPU detection

**llm-worker-rbee (Executor):**
- `bin/30_llm_worker_rbee/src/main.rs` - HTTP server
- `bin/30_llm_worker_rbee/src/heartbeat.rs` - Heartbeat to queen

**Shared Infrastructure:**
- `bin/99_shared_crates/job-server/` - Job registry
- `bin/99_shared_crates/job-client/` - Job submission
- `bin/99_shared_crates/rbee-operations/` - Operation enum
- `bin/99_shared_crates/narration-core/` - Observability

**Security:**
- `bin/99_shared_crates/auth-min/` - Authentication
- `bin/99_shared_crates/audit-logging/` - GDPR compliance
- `bin/99_shared_crates/input-validation/` - Injection prevention
- `bin/99_shared_crates/secrets-management/` - Credential protection
- `bin/99_shared_crates/deadline-propagation/` - Resource protection

---

## Key Architectural Decisions

### Queen Build Configurations (Oct 23, 2025)

**Decision:** Support two queen build modes for different deployment scenarios

**Distributed Queen (Default):**
```bash
cargo build --bin queen-rbee
```
- All operations via HTTP
- Manages remote hives
- ~5-10ms overhead per operation

**Integrated Queen (local-hive feature):**
```bash
cargo build --bin queen-rbee --features local-hive
```
- Direct Rust calls for localhost (~0.1ms)
- HTTP for remote hives (still available!)
- 50-100x faster localhost operations

**Benefits:**
- Performance: 50-100x faster localhost operations
- Flexibility: Same binary can manage both local and remote
- Simplicity: One binary instead of two for single-machine setups
- Compatibility: Distributed mode still available for all use cases

**Smart Recommendations:**
- rbee-keeper detects queen configuration
- Prompts user to rebuild with local-hive when installing localhost hive
- Provides performance comparison and recommendations

**Impact:**
- Queen can embed hive crates (optional dependencies)
- Hive forwarder supports dual-mode routing
- rbee-keeper gains queen lifecycle management
- New `/v1/build-info` endpoint for configuration detection

### TEAM-261: Heartbeat Simplification (Oct 23, 2025)

**Decision:** Remove hive heartbeat, workers send directly to queen

**Before:**
```
Worker → Hive (aggregation) → Queen
```

**After:**
```
Worker → Queen (direct)
```

**Benefits:**
- Simpler architecture (no aggregation)
- Single source of truth (queen)
- ~110 LOC removed
- Direct worker visibility

**Impact:**
- `bin/20_rbee_hive/src/heartbeat.rs` deleted (80 LOC)
- `bin/20_rbee_hive/src/main.rs` simplified (~30 LOC removed)
- `bin/10_queen_rbee/src/http/heartbeat.rs` updated (worker endpoint added)
- `bin/30_llm_worker_rbee/src/heartbeat.rs` updated (sends to queen)

**See:** `bin/.plan/TEAM_261_IMPLEMENTATION_COMPLETE.md`

### TEAM-258: Operation Consolidation

**Decision:** Consolidate worker/model operations into generic forwarding

**Before:** 8 separate match arms in job_router.rs

**After:** Single catch-all with `should_forward_to_hive()`

**Benefits:**
- 200+ LOC removed from queen
- New operations don't require queen changes
- Single source of truth

### TEAM-259: Job Client Consolidation

**Decision:** Extract job submission pattern into shared crate

**Before:** Duplicated code in keeper and queen (~240 LOC total)

**After:** Single `job-client` crate (207 LOC)

**Benefits:**
- 80+ LOC removed
- Single source of truth
- Consistent behavior

---

## System Metrics

### Code Statistics

| Component | LOC | Status |
|-----------|-----|--------|
| rbee-keeper | ~500 | ✅ M0 Complete |
| queen-rbee | ~1,500 | 🚧 In Progress |
| rbee-hive | ~800 | ✅ M0 Complete |
| llm-worker-rbee | ~1,000 | ✅ M0 Complete |
| Shared crates | ~3,000 | ✅ Complete |
| **Total** | **~6,800** | **68% Complete** |

### Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Hive operation (HTTP) | 1-5ms | Fast daemon-to-daemon |
| Worker spawn | 2-5s | Model loading |
| Inference (first token) | 100-500ms | Depends on model |
| Inference (subsequent) | 20-50ms | 20-50 tokens/s |
| SSE streaming | <1ms overhead | Real-time |

### Scalability Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| Hives per queen | 100+ | Network bandwidth |
| Workers per hive | 10-20 | GPU count |
| Concurrent inferences | 1000+ | Worker count |
| SSE connections | 10000+ | File descriptors |

---

## For New Engineering Teams

### Getting Started

1. **Read Part 1** - Understand system design and philosophy
2. **Read Part 2** - Understand each component's role
3. **Read Part 3** - Understand shared infrastructure
4. **Skim Parts 4-6** - Reference as needed

### Key Concepts to Understand

1. **Smart/Dumb Architecture**
   - Queen makes ALL decisions
   - Workers are dumb executors

2. **Job-Based Pattern**
   - POST /v1/jobs → job_id
   - GET /v1/jobs/{job_id}/stream → SSE events

3. **SSE Routing (CRITICAL!)**
   - Server-side narration MUST include `.job_id()`
   - Without job_id, events go to stdout (not SSE)

4. **Operation Routing**
   - Hive lifecycle → queen handles
   - Worker/model ops → queen forwards to hive
   - Inference → queen schedules to worker

5. **TEAM-XXX Attribution**
   - All code changes attributed to teams
   - Handoff documents maintain continuity

### Common Pitfalls

1. **Forgetting `.job_id()` in narration**
   ```rust
   // ❌ WRONG (events go to stdout)
   NARRATE.action("x").emit();
   
   // ✅ CORRECT (events go to SSE)
   NARRATE.action("x").job_id(&job_id).emit();
   ```

2. **Building queen without considering deployment**
   ```bash
   # ❌ SUBOPTIMAL (for single-machine setup)
   cargo build --bin queen-rbee  # Distributed mode
   # Then installing localhost hive (5-10ms HTTP overhead)
   
   # ✅ OPTIMAL (for single-machine setup)
   cargo build --bin queen-rbee --features local-hive
   # Direct Rust calls (~0.1ms overhead)
   ```

3. **Not using timing-safe comparison**
   ```rust
   // ❌ WRONG (timing attack)
   if token == stored_token { ... }
   
   // ✅ CORRECT (constant-time)
   if compare_tokens(&token, &stored_token) { ... }
   ```

4. **Logging secrets**
   ```rust
   // ❌ WRONG (secret exposed)
   eprintln!("Token: {}", token);
   
   // ✅ CORRECT (fingerprint only)
   eprintln!("Token fingerprint: {}", fingerprint_token(&token));
   ```

5. **Not validating inputs**
   ```rust
   // ❌ WRONG (injection risk)
   let path = PathBuf::from(user_input);
   
   // ✅ CORRECT (validated)
   let path = validate_file_path(user_input)?;
   ```

---

## Special Topics

### [Cross-Platform Architecture](CROSS_PLATFORM_ARCHITECTURE.md)

**Topics:**
- Platform support matrix (Linux, macOS, Windows)
- Directory structure for each platform
- Implementation strategy using `dirs` crate
- Component-specific implementations
- Testing strategy
- Migration guide

**Key Insights:**
- Config: `~/.config/rbee/` (Linux), `~/Library/Application Support/rbee/` (macOS), `%APPDATA%\rbee\` (Windows)
- Cache: `~/.cache/rbee/` (Linux), `~/Library/Caches/rbee/` (macOS), `%LOCALAPPDATA%\rbee\` (Windows)
- Model catalog already cross-platform ✅
- rbee-config needs update (3-4 hours)

**Implementation Plans:**
- `bin/.plan/CROSS_PLATFORM_CONFIG_PLAN.md` - Detailed implementation
- `bin/.plan/CROSS_PLATFORM_SUMMARY.md` - Quick reference
- `bin/.plan/STORAGE_ARCHITECTURE.md` - Model storage (already cross-platform)

**Read this** to understand cross-platform support and implementation.

---

## Document Maintenance

### Updating These Documents

When making significant architectural changes:

1. Update relevant part (1-6)
2. Update this README if needed
3. Add note to "Key Architectural Decisions"
4. Reference TEAM-XXX that made the change

### Document History

- **Oct 23, 2025 (v1.2):** Cross-platform architecture added (TEAM-266)
  - Added CROSS_PLATFORM_ARCHITECTURE.md
  - Updated Part 3 with cross-platform config details
  - Updated Part 5 with cross-platform development patterns
  - Documented platform-specific directories (Linux, macOS, Windows)
  - Created implementation plans in bin/.plan/

- **Oct 23, 2025 (v1.1):** Queen build configurations added
  - Added distributed vs integrated queen modes
  - Documented local-hive feature flag
  - Added queen lifecycle management to rbee-keeper
  - Updated deployment topologies
  - Added smart prompts for localhost optimization

- **Oct 23, 2025 (v1.0):** Initial version (TEAM-261)
  - All 6 parts created
  - Covers complete architecture as of M0
  - Documents TEAM-261 heartbeat simplification

---

## Questions?

For architectural questions, refer to:

1. **This documentation** (start here)
2. **Plan documents** (`bin/.plan/TEAM_*.md`)
3. **Code comments** (TEAM-XXX attributed)
4. **BDD tests** (`bin/*/bdd/features/*.feature`)

---

**Architecture documentation complete. Happy building! 🐝**

# Architecture Documentation Changelog

## v1.1 - Queen Build Configurations (Oct 23, 2025)

### Major Addition: Dual-Mode Queen Architecture

Added comprehensive documentation for queen-rbee's two build configurations:

#### 1. Distributed Queen (Default)
- All operations via HTTP (job-client)
- Manages remote hives
- ~5-10ms overhead per operation
- Requires separate rbee-hive binary

#### 2. Integrated Queen (local-hive feature)
- Direct Rust calls for localhost (~0.1ms)
- HTTP forwarding for remote hives (still available!)
- 50-100x faster localhost operations
- No separate rbee-hive binary needed

### Documentation Updates

#### Part 1: System Design (00_OVERVIEW_PART_1.md)
- **NEW SECTION:** "Queen Build Configurations"
  - Distributed vs Integrated comparison
  - Use cases for each mode
  - Smart recommendations system
- **UPDATED:** "Deployment Topologies"
  - Single Machine - Integrated (recommended)
  - Single Machine - Distributed (optional)
  - Multi-Machine - Hybrid (production)
  - Multi-Machine - Pure Distributed
- **UPDATED:** "System Characteristics"
  - Performance comparison table (distributed vs integrated)
  - 50-100x speedup for localhost operations

#### Part 2: Component Deep Dive (01_COMPONENTS_PART_2.md)
- **UPDATED:** rbee-keeper responsibilities
  - NEW: Queen lifecycle management
  - NEW: Smart prompts for localhost optimization
- **UPDATED:** rbee-keeper command structure
  - NEW: `QueenCommands` enum
  - Commands: start, stop, status, rebuild, info
- **NEW:** Smart prompts example
  - Performance comparison display
  - Recommendation for local-hive rebuild
  - User choice preservation
- **UPDATED:** queen-rbee section
  - NEW: "Build Configurations" subsection
  - Distributed vs Integrated comparison
- **UPDATED:** Hive forwarder implementation
  - Dual-mode routing (HTTP or direct)
  - `#[cfg(feature = "local-hive")]` conditional compilation
  - `forward_via_local_hive()` for direct calls
- **UPDATED:** queen-rbee dependencies
  - Feature flags: `local-hive`
  - Optional dependencies on hive crates
  - job-client always available (for remote hives)
- **NEW:** Build info endpoint
  - `/v1/build-info` for configuration detection
  - Returns version, features, build timestamp

#### README.md Updates
- **UPDATED:** Part 1 summary
  - Added "Queen Build Configurations" topic
  - Added key insight about two queen modes
- **UPDATED:** Part 2 summary
  - Added NEW items for rbee-keeper and queen-rbee
  - Highlighted smart prompts and dual-mode forwarder
- **UPDATED:** Key Architectural Decisions
  - NEW: Queen Build Configurations decision
  - Benefits, smart recommendations, impact
- **UPDATED:** Common Pitfalls
  - NEW: Building queen without considering deployment
  - Example of suboptimal vs optimal builds
- **UPDATED:** Document History
  - Added v1.1 entry with all changes

### Key Benefits

1. **Performance:** 50-100x faster localhost operations with integrated mode
2. **Flexibility:** Same binary can manage both local and remote hives
3. **Simplicity:** One binary instead of two for single-machine setups
4. **User Experience:** Smart prompts guide users to optimal configuration
5. **Backward Compatibility:** Distributed mode still default, no breaking changes

### Implementation Status

- ✅ Architecture documented
- ✅ Use cases defined
- ✅ API design specified
- ✅ Smart prompts designed
- 🚧 Implementation pending (future work)

### Files Modified

1. `.arch/00_OVERVIEW_PART_1.md` - Added queen configurations and deployment topologies
2. `.arch/01_COMPONENTS_PART_2.md` - Updated keeper and queen sections
3. `.arch/README.md` - Updated summaries and key decisions
4. `.arch/CHANGELOG.md` - Created this file

### Lines Changed

- **00_OVERVIEW_PART_1.md:** ~150 lines added/modified
- **01_COMPONENTS_PART_2.md:** ~200 lines added/modified
- **README.md:** ~80 lines added/modified
- **Total:** ~430 lines of documentation updates

---

## Pending: Post-TEAM-261 Cleanup

### Status: 🚧 PENDING IMPLEMENTATION

**Trigger:** TEAM-261 heartbeat simplification exposed deprecated code

**See:** [CLEANUP_PLAN_TEAM261.md](CLEANUP_PLAN_TEAM261.md) for full details

### Summary of Cleanup Needed

1. **Delete Dead Code:**
   - `bin/25_rbee_hive_crates/worker-registry` (obsolete)
   - `bin/99_shared_crates/daemon-ensure` (empty file)
   - `bin/99_shared_crates/hive-core` (unused)
   - Hive heartbeat logic in `bin/99_shared_crates/heartbeat/`

2. **Rename for Clarity:**
   - `hive-registry` → `worker-registry` (reflects reality)
   - `SseBroadcaster` → `SseChannelRegistry` (no longer broadcasts)

3. **New Features:**
   - Queen lifecycle management (install/uninstall)
   - Smart prompts for localhost optimization

**Impact:** ~910 LOC dead code removed, +200 LOC new features

---

## v1.0 - Initial Architecture Documentation (Oct 23, 2025)

### Initial Release

Created comprehensive 6-part architecture documentation:

1. **Part 1:** System Design & Philosophy
2. **Part 2:** Component Deep Dive
3. **Part 3:** Shared Infrastructure
4. **Part 4:** Data Flow & Protocols
5. **Part 5:** Development Patterns
6. **Part 6:** Security & Compliance

### Key Features Documented

- Four-binary system (keeper, queen, hive, worker)
- Smart/dumb architecture
- Job-based pattern with SSE streaming
- TEAM-261 heartbeat simplification
- Defense-in-depth security (5 layers)
- GDPR compliance

### Total Documentation

- 7 files created
- ~3,500 lines of documentation
- Comprehensive coverage of entire architecture

---

**For detailed information, see individual part files in `.arch/` directory.**

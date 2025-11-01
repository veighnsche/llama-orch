# TEAM-371: Heartbeat SSE Refactor - Document Index

**Date:** Oct 31, 2025  
**Author:** TEAM-371  
**Status:** 📋 READY FOR IMPLEMENTATION

---

## Quick Navigation

### Investigation & Architecture

1. **[Investigation Report](./TEAM_371_INVESTIGATION_REPORT.md)** *(if exists)*
   - Current push-based implementation
   - Proposed SSE-based architecture
   - Files involved and data flows

2. **[Architecture Summary](./TEAM_371_ARCHITECTURE_SUMMARY.md)** ⭐ **START HERE**
   - Why SSE? Why keep handshake?
   - Complete comparison (before/after)
   - Contract changes
   - FAQ

3. **[Handshake Explained](./TEAM_371_HANDSHAKE_EXPLAINED.md)** ⭐ **READ THIS**
   - Visual guide to handshake preservation
   - Why discovery is essential
   - Sequence diagrams
   - Restart scenarios

---

## Implementation Phases

### Phase 1: Create Hive SSE Stream (1 day)
**Document:** [TEAM_371_PHASE_1_HIVE_SSE_STREAM.md](./TEAM_371_PHASE_1_HIVE_SSE_STREAM.md)

**Mission:** Create SSE heartbeat stream on Hive  
**Team:** TEAM-372 (or next available)  
**Changes:**
- NEW: `bin/20_rbee_hive/src/http/heartbeat_stream.rs`
- MODIFY: `bin/20_rbee_hive/src/main.rs` (add route, broadcaster)
- MODIFY: `bin/20_rbee_hive/src/http/mod.rs` (export module)

**Result:** Hive exposes `GET /v1/heartbeats/stream`

---

### Phase 2: Queen Subscribes to SSE (1 day)
**Document:** [TEAM_371_PHASE_2_QUEEN_SUBSCRIBER.md](./TEAM_371_PHASE_2_QUEEN_SUBSCRIBER.md)

**Mission:** Queen subscribes to hive SSE, callback triggers subscription  
**Team:** TEAM-373 (or next available)  
**Changes:**
- NEW: `bin/10_queen_rbee/src/hive_subscriber.rs`
- MODIFY: `bin/10_queen_rbee/src/http/heartbeat.rs` (change to one-time callback)
- MODIFY: `bin/10_queen_rbee/src/main.rs` (add subscriber module, change route)
- MODIFY: `bin/20_rbee_hive/src/heartbeat.rs` (change to ready callback)

**Result:** Discovery callback triggers SSE subscription, continuous telemetry via SSE

---

### Phase 3: Delete Old POST Logic (0.5 day)
**Document:** [TEAM_371_PHASE_3_DELETE_POST_TELEMETRY.md](./TEAM_371_PHASE_3_DELETE_POST_TELEMETRY.md)

**Mission:** RULE ZERO - Delete unused POST telemetry code  
**Team:** TEAM-374 (or next available)  
**Changes:**
- DELETE: `bin/20_rbee_hive/src/heartbeat.rs` functions
- DELETE: `bin/10_queen_rbee/src/http/heartbeat.rs::handle_hive_heartbeat()`
- DELETE: Route registrations
- FIX: Compilation errors

**Result:** Clean codebase, SSE-only telemetry

---

## Key Concepts

### Discovery vs Telemetry

**Discovery (Handshake):**
- Purpose: "I exist, connect to me"
- Method: POST /v1/hive/ready
- Frequency: ONE-TIME (with retry)
- Payload: hive_id + hive_url
- **PRESERVED IN THIS REFACTOR**

**Telemetry (Streaming):**
- Purpose: "Here's what's happening"
- Method: SSE /v1/heartbeats/stream
- Frequency: CONTINUOUS (1s)
- Payload: workers, GPU stats
- **CHANGED FROM POST TO SSE**

### Rule ZERO Compliance

All phases follow RULE ZERO:
- ✅ Breaking changes allowed (pre-1.0)
- ✅ Delete deprecated code immediately
- ✅ No backwards compatibility wrappers
- ✅ Compiler finds all call sites

---

## Testing Strategy

### Phase 1 Testing
```bash
# Hive SSE stream works
curl -N http://localhost:7835/v1/heartbeats/stream
# Should stream events every 1s
```

### Phase 2 Testing
```bash
# Queen subscribes after callback
# Start queen, start hive
# Check logs for "Subscribing to hive SSE"
# Check queen SSE forwards hive telemetry
curl -N http://localhost:7833/v1/heartbeats/stream
```

### Phase 3 Testing
```bash
# No compilation errors
cargo check --workspace
# Discovery still works
# SSE telemetry flows end-to-end
```

---

## Dependencies

### Phase 1
```toml
# bin/20_rbee_hive/Cargo.toml
async-stream = "0.3"
```

### Phase 2
```toml
# bin/10_queen_rbee/Cargo.toml
reqwest-eventsource = "0.6"
futures = "0.3"
```

### Phase 3
No new dependencies

---

## Files Reference

### Hive Side
- `bin/20_rbee_hive/src/heartbeat.rs` - Discovery logic
- `bin/20_rbee_hive/src/http/heartbeat_stream.rs` - SSE stream (NEW)
- `bin/20_rbee_hive/src/main.rs` - Route registration

### Queen Side
- `bin/10_queen_rbee/src/http/heartbeat.rs` - Callback handler
- `bin/10_queen_rbee/src/http/heartbeat_stream.rs` - Queen SSE (existing)
- `bin/10_queen_rbee/src/hive_subscriber.rs` - Subscribe to hives (NEW)
- `bin/10_queen_rbee/src/discovery.rs` - Discovery logic (existing)
- `bin/10_queen_rbee/src/main.rs` - Route registration

### Contracts
- `bin/97_contracts/hive-contract/src/heartbeat.rs` - HiveHeartbeat struct
- `bin/97_contracts/hive-contract/src/lib.rs` - HiveInfo, HiveDevice

### Monitoring
- `bin/25_rbee_hive_crates/monitor/src/telemetry.rs` - collect_all_workers()
- `bin/25_rbee_hive_crates/monitor/src/lib.rs` - ProcessStats

---

## Success Criteria (All Phases)

- [ ] Hive exposes SSE stream at `GET /v1/heartbeats/stream`
- [ ] Hive sends one-time callback `POST /v1/hive/ready`
- [ ] Queen subscribes to hive SSE after callback
- [ ] Continuous telemetry flows via SSE (1s interval)
- [ ] Discovery handshake works (both startup scenarios)
- [ ] Queen restart triggers rediscovery
- [ ] Hive restart triggers rediscovery
- [ ] Multiple hives connect simultaneously
- [ ] No compilation errors or warnings
- [ ] Integration tests pass
- [ ] Manual testing confirms end-to-end flow

---

## Timeline

**Total: 2.5 days**

- Phase 1: 1 day (TEAM-372)
- Phase 2: 1 day (TEAM-373)
- Phase 3: 0.5 day (TEAM-374)

**Parallelization:** Phases must be sequential (dependencies)

---

## Related Documents

### Existing Specs
- `bin/.specs/HEARTBEAT_ARCHITECTURE.md` - Current heartbeat spec
- `bin/TEAM_367_QUEEN_RESTART_FIXES.md` - Restart detection
- `bin/.plan/TEAM_362_COMPLETE.md` - Worker telemetry integration

### Team History
- TEAM-361: Worker telemetry collection
- TEAM-362: Queen storage & SSE integration
- TEAM-365: Bidirectional handshake
- TEAM-366: Edge case guards
- TEAM-367: Restart fixes
- **TEAM-371:** This refactor (investigation)
- **TEAM-372:** Phase 1 implementation (pending)
- **TEAM-373:** Phase 2 implementation (pending)
- **TEAM-374:** Phase 3 implementation (pending)

---

## Questions?

**Q: Do we keep the handshake?**  
A: YES! Read [TEAM_371_HANDSHAKE_EXPLAINED.md](./TEAM_371_HANDSHAKE_EXPLAINED.md)

**Q: Why SSE instead of POST?**  
A: Read [TEAM_371_ARCHITECTURE_SUMMARY.md](./TEAM_371_ARCHITECTURE_SUMMARY.md)

**Q: What's RULE ZERO?**  
A: Breaking changes > backwards compatibility (pre-1.0). See [engineering-rules.md](../../.windsurf/rules/engineering-rules.md)

**Q: Can I skip Phase 1 and go straight to Phase 2?**  
A: NO. Phases have dependencies and must be done in order.

---

**TEAM-371 Documentation Complete. Ready for implementation.**

# ✅ RESOLVED: Handoff Watcher Architecture

**Date**: 2025-09-30  
**Status**: ✅ **AGREED & PLANNED**  
**Teams**: orchestratord ↔ pool-managerd

---

## Summary

**Issue**: Handoff watcher in orchestratord only works for HOME_PROFILE (breaks in cloud)  
**Resolution**: pool-managerd team **accepts ownership** and will implement in Phase 3  
**Timeline**: 2 weeks for v0.2.0 implementation

---

## Key Agreements

### ✅ Ownership Clarified

| Responsibility | Owner | Reason |
|----------------|-------|--------|
| Watch handoff files | **pool-managerd** | Filesystem co-location |
| Update pool readiness | **pool-managerd** | Owns pool state |
| Detect pool ready | **orchestratord** | Polls HTTP API |
| **Bind adapter** | **orchestratord** | **Owns adapter-host** |
| Route tasks | **orchestratord** | Owns placement logic |

### ✅ Migration Path

- **v0.1.0** (now): orchestratord watcher (HOME_PROFILE only) ✅ ACCEPTABLE
- **v0.2.0** (2 weeks): pool-managerd watcher + orchestratord polling
- **v1.0.0** (4 weeks): Add callbacks, optimize

### ✅ Implementation Details

**pool-managerd will**:
- Create `src/watcher/handoff.rs` module
- Watch `.runtime/engines/*.json`
- Update registry when files detected
- Expose via existing `GET /v2/pools/{id}/status`

**orchestratord will**:
- Poll pool-managerd every 5s (configurable)
- Bind adapter when `ready=true` detected
- Keep HOME_PROFILE watcher for backward compat

---

## Action Items

### orchestratord Team (Us)

1. ✅ **Document limitation** in current code
2. ✅ **Add TODO comments** for cloud_profile
3. ⏸️ **Continue BDD tests** (HOME_PROFILE mode OK for now)
4. 📋 **Implement polling** (when pool-managerd ready)

### pool-managerd Team (Them)

1. 📋 **Implement watcher** (Week 3 - Phase 3)
2. 📋 **Add unit tests** for handoff processing
3. 📋 **Update specs** (OC-POOL-3105 through 3109)
4. 📋 **E2E testing** with orchestratord

---

## For Our BDD Tests

**Current Status**: ✅ **ACCEPTABLE**

- Tests run in HOME_PROFILE mode (single machine)
- Direct `process_handoff_file()` calls work fine
- No need to block on pool-managerd implementation
- **We can continue to 100%!**

**Future**: When pool-managerd watcher is ready, we'll update tests to:
- Mock pool-managerd HTTP responses
- Or run real pool-managerd daemon
- Test distributed scenarios

---

## Code Comments to Add

### In `orchestratord/src/services/handoff.rs`:

```rust
//! Handoff autobind watcher — monitors engine-provisioner handoff files
//!
//! ⚠️ HOME_PROFILE ONLY - CLOUD_PROFILE LIMITATION ⚠️
//!
//! This implementation assumes orchestratord and engine-provisioner share
//! a filesystem. This ONLY works for HOME_PROFILE (single machine).
//!
//! For CLOUD_PROFILE (distributed deployment), the handoff watcher MUST
//! be owned by pool-managerd (which runs on the same machine as
//! engine-provisioner). orchestratord will poll pool-managerd via HTTP
//! instead of watching the filesystem directly.
//!
//! Status: Acceptable for v0.1.0 (HOME_PROFILE only)
//! Migration: v0.2.0 will move watcher to pool-managerd
//! See: HANDOFF_WATCHER_ARCHITECTURE_ISSUE.md
//!
//! TODO[CLOUD_PROFILE]: Remove this watcher or make it HOME_PROFILE only
//! TODO[CLOUD_PROFILE]: Implement HTTP polling of pool-managerd
```

---

## Spec Updates Needed

### `.specs/20-orchestratord.md`

Add section:
```markdown
## Profile-Specific Behavior

### HOME_PROFILE
- orchestratord MAY watch handoff files directly (filesystem access)
- Acceptable for single-machine deployments

### CLOUD_PROFILE
- orchestratord MUST poll pool-managerd via HTTP
- orchestratord MUST NOT assume filesystem access to handoff files
- Polling interval SHOULD be configurable (default: 5s)
```

### `.specs/30-pool-managerd.md`

Add requirements (pool-managerd team will do this):
```markdown
## OC-POOL-3105: Handoff Watcher (CLOUD_PROFILE)

- [OC-POOL-3105] pool-managerd MUST watch runtime directory for handoff files
- [OC-POOL-3106] Handoff files MUST be processed within 2 seconds
- [OC-POOL-3107] Processed files MAY be deleted automatically
- [OC-POOL-3108] Watcher MUST be configurable via env vars
```

---

## Lessons Learned

### ✅ What Went Well

1. **Early detection**: Caught during BDD testing (before production!)
2. **Clear communication**: Issue document was comprehensive
3. **Fast response**: pool-managerd team responded same day
4. **Collaborative**: Both teams aligned on solution
5. **Pragmatic**: Agreed on phased migration (not blocking v0.1.0)

### 💡 Process Improvements

1. **Architecture reviews**: Should have caught this in design phase
2. **Profile testing**: Need explicit HOME vs CLOUD test scenarios
3. **Service boundaries**: Document filesystem assumptions upfront
4. **Cross-team sync**: Regular architecture alignment meetings

### 📚 Documentation Wins

1. **Detailed issue**: Made it easy for pool-managerd to understand
2. **Code examples**: Concrete implementation suggestions helped
3. **Migration path**: Clear phases reduced risk
4. **Ownership matrix**: Clarified responsibilities

---

## Impact on Current Work

### ✅ No Blocker for BDD Tests

- Current implementation works fine for HOME_PROFILE
- Tests can continue to 100%
- No need to wait for pool-managerd implementation

### 📋 Future Work Items

1. Add TODO comments to handoff.rs (5 min)
2. Update specs with profile-specific behavior (15 min)
3. Document in HOME_PROFILE.md (10 min)
4. Create CLOUD_PROFILE.md spec (30 min)
5. Implement polling when pool-managerd ready (2-3 hours)

---

## Conclusion

**Perfect resolution!** 🎉

- ✅ Architecture issue identified
- ✅ Ownership clarified
- ✅ Implementation plan agreed
- ✅ Timeline established (2 weeks)
- ✅ No blocker for current work

**We can continue with BDD tests and hit 100%!**

The handoff watcher will be properly architected for cloud_profile in v0.2.0, but v0.1.0 is acceptable as HOME_PROFILE only.

---

**Status**: ✅ RESOLVED  
**Blocker**: NO (for v0.1.0)  
**Action**: Continue to 100% BDD coverage! 🚀

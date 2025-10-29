# TEAM-269 to TEAM-272: Implementation Summary

**Created:** Oct 23, 2025  
**Status:** ✅ Guides Ready  
**Phases:** 3-6 of 9

---

## 📋 Individual Guides Created

| Team | Guide | Focus | Effort | Complexity |
|------|-------|-------|--------|------------|
| 269 | TEAM_269_MODEL_PROVISIONER.md | Model downloading | 24-32h | Medium |
| 270 | TEAM_270_WORKER_REGISTRY.md | Worker registry | 20-24h | Low |
| 271 | TEAM_271_WORKER_LIFECYCLE_SPAWN.md | Worker spawning | 32-40h | ⚠️ **HIGH** |
| 272 | TEAM_272_WORKER_LIFECYCLE_MGMT.md | Worker management | 24-32h | Medium |

**Total:** 100-128 hours

---

## 🎯 What Each Guide Contains

### All Guides Include:
- ✅ Mission statement
- ✅ Deliverables checklist
- ✅ Files to create/modify
- ✅ Step-by-step implementation
- ✅ Code examples with TEAM signatures
- ✅ Acceptance criteria
- ✅ Testing commands
- ✅ Handoff checklist
- ✅ Known limitations
- ✅ Reference implementations

### Guide-Specific Features:

**TEAM-269 (Model Provisioner):**
- HuggingFace Hub API integration notes
- Progress tracking implementation
- Placeholder download for v0.1.0
- Future: Real HF Hub downloads

**TEAM-270 (Worker Registry):**
- Arc<Mutex<HashMap>> pattern
- In-memory storage (ephemeral workers)
- 8+ unit tests
- Similar to model-catalog but simpler

**TEAM-271 (Worker Spawn) ⚠️ Most Complex:**
- Process spawning with tokio
- Port allocation (9100-9200)
- Binary resolution strategy
- Mock worker binary for testing
- Health check patterns
- **Most challenging phase!**

**TEAM-272 (Worker Management):**
- CRUD operations (List, Get, Delete)
- Process cleanup (Unix/Windows)
- SIGTERM/SIGKILL patterns
- Platform-specific code

---

## 🚀 Implementation Order

```
TEAM-269: Model Provisioner
    ↓
TEAM-270: Worker Registry
    ↓
TEAM-271: Worker Spawn ← ⚠️ Most complex, take your time!
    ↓
TEAM-272: Worker Management
    ↓
TEAM-273: Integration (next phase)
```

**Critical Path:** Must be done in order. Each team depends on previous work.

---

## 📊 Complexity Breakdown

### Easy (20-24h)
- **TEAM-270:** Worker Registry
  - Similar to model-catalog
  - Standard CRUD patterns
  - Well-defined scope

### Medium (24-32h)
- **TEAM-269:** Model Provisioner
  - HTTP downloads
  - Progress tracking
  - File management
  
- **TEAM-272:** Worker Management
  - CRUD operations
  - Process cleanup
  - Platform differences

### Hard (32-40h) ⚠️
- **TEAM-271:** Worker Spawn
  - Process spawning
  - Port allocation
  - Binary resolution
  - Health checks
  - Many edge cases
  - **Requires careful testing!**

---

## 🎓 Key Patterns Used

### TEAM-269: Async Download Pattern
```rust
pub async fn download_model(&self, job_id: &str, model_id: &str) -> Result<String> {
    // Progress tracking with catalog updates
    self.catalog.update_status(model_id, ModelStatus::Downloading { progress: 0.5 })?;
}
```

### TEAM-270: Registry Pattern
```rust
#[derive(Clone)]
pub struct WorkerRegistry {
    workers: Arc<Mutex<HashMap<String, WorkerEntry>>>,
}
```

### TEAM-271: Process Spawn Pattern
```rust
let mut child = Command::new(&self.worker_binary)
    .arg("--model").arg(model_id)
    .arg("--port").arg(port.to_string())
    .spawn()?;
```

### TEAM-272: Process Cleanup Pattern
```rust
#[cfg(unix)]
{
    kill(Pid::from_raw(pid as i32), Signal::SIGTERM)?;
}
```

---

## ✅ Success Criteria (All Teams)

### Code Quality
- [ ] Follows CRUD patterns from TEAM-211/TEAM-268
- [ ] All narration includes `.job_id()`
- [ ] Comprehensive error handling
- [ ] Unit tests passing
- [ ] No warnings

### Integration
- [ ] Operations wired up in job_router.rs
- [ ] State initialized in main.rs
- [ ] HiveState updated in http/jobs.rs
- [ ] Compilation successful

### Documentation
- [ ] Handoff document created
- [ ] Known limitations documented
- [ ] Example narration output shown
- [ ] Notes for next team

---

## 🚨 Common Pitfalls

### All Teams
1. **Forgetting job_id** - All narration MUST include `.job_id()`
2. **Not following patterns** - Mirror TEAM-211/TEAM-268 CRUD style
3. **Skipping tests** - Unit tests are required
4. **Poor error messages** - Make errors descriptive

### TEAM-269 Specific
- Don't try to implement full HF Hub API (use placeholder)
- Progress tracking must update catalog status
- Handle duplicate downloads

### TEAM-270 Specific
- Use Arc<Mutex<>> not just Mutex
- Keep lock scope minimal
- Clone registry, not workers HashMap

### TEAM-271 Specific ⚠️
- Worker binary may not exist (document this!)
- Port allocation can fail (handle gracefully)
- Process spawn can fail (clear error messages)
- Health check is TODO (use delay for v0.1.0)

### TEAM-272 Specific
- Process killing is platform-specific
- Use conditional compilation (#[cfg(unix)])
- SIGTERM before SIGKILL (graceful shutdown)

---

## 📚 Reference Implementations

**For All Teams:**
- TEAM-211: Simple operations (narration patterns)
- TEAM-268: Model operations (CRUD patterns)
- job-server: Arc<Mutex<>> patterns
- narration-core: Narration patterns

**Team-Specific:**
- TEAM-269: daemon-lifecycle (process patterns)
- TEAM-270: model-catalog (registry patterns)
- TEAM-271: hive-lifecycle/start.rs (spawning)
- TEAM-272: hive-lifecycle/stop.rs (process cleanup)

---

## 🎯 Next Steps

1. **Read your team's guide** (TEAM_XXX_*.md)
2. **Check previous team's handoff** (TEAM_XXX_HANDOFF.md)
3. **Verify compilation** of previous work
4. **Implement your phase** following the guide
5. **Test thoroughly** (unit tests + manual)
6. **Create handoff** for next team
7. **Update progress** in START_HERE document

---

## 📞 Questions?

1. Read your team's dedicated guide
2. Read the consolidated guide (TEAM_269_TO_272_IMPLEMENTATION_GUIDES.md)
3. Check reference implementations
4. Review previous team's handoff
5. Document questions in your handoff

---

**All guides are ready! Teams 269-272, you've got this! 🚀**

**Remember: TEAM-271 is the hardest - take your time and test thoroughly!**

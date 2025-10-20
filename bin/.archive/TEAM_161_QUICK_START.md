# TEAM-161 Quick Start Guide

**Mission:** Make all E2E tests pass with zero warnings

---

## 🚀 Quick Commands

```bash
# Fix compilation
cargo check --bin queen-rbee

# Run tests
cargo xtask e2e:queen
cargo xtask e2e:hive
cargo xtask e2e:cascade

# Check warnings
cargo check 2>&1 | grep warning

# Kill all processes
pkill -9 rbee
```

---

## ✅ TODO Checklist (Copy This)

```
[ ] 1. Add async-trait = "0.1" to bin/10_queen_rbee/Cargo.toml
[ ] 2. Fix lifetime in device_detector.rs
[ ] 3. Fix type mismatch in heartbeat.rs  
[ ] 4. Add device_detector to HeartbeatState
[ ] 5. Verify: cargo check --bin queen-rbee (0 errors)
[ ] 6. Add process spawning to handle_add_hive
[ ] 7. Create heartbeat_sender.rs in rbee-hive
[ ] 8. Add device detection trigger in handle_heartbeat
[ ] 9. Implement cascading shutdown in handle_shutdown
[ ] 10. Fix all warnings (dead code, unused vars, missing docs)
[ ] 11. Run: cargo xtask e2e:queen → PASS
[ ] 12. Run: cargo xtask e2e:hive → PASS
[ ] 13. Run: cargo xtask e2e:cascade → PASS
[ ] 14. Verify: cargo check (0 errors, 0 warnings)
[ ] 15. Verify: ps aux | grep rbee (0 processes)
```

---

## 📁 Files to Modify

**Critical:**
1. `bin/10_queen_rbee/Cargo.toml` - Add dependency
2. `bin/10_queen_rbee/src/http/device_detector.rs` - Fix lifetime
3. `bin/10_queen_rbee/src/http/heartbeat.rs` - Fix type + add logic
4. `bin/10_queen_rbee/src/main.rs` - Add field
5. `bin/10_queen_rbee/src/http/add_hive.rs` - Spawn process
6. `bin/10_queen_rbee/src/http/shutdown.rs` - Cascade shutdown

**Create:**
7. `bin/20_rbee_hive/src/heartbeat_sender.rs` - New file

**Cleanup:**
8. `xtask/src/e2e/helpers.rs` - Fix warnings
9. `bin/99_shared_crates/rbee-types/src/*.rs` - Add docs

---

## 🎯 Success Criteria

```bash
# All these should pass:
cargo check                    # 0 errors, 0 warnings
cargo xtask e2e:queen         # ✅ PASSED
cargo xtask e2e:hive          # ✅ PASSED
cargo xtask e2e:cascade       # ✅ PASSED
ps aux | grep rbee            # No processes
```

---

## 📚 Read These First

1. `bin/TEAM_160_HANDOFF.md` - Full handoff (this is your bible)
2. `xtask/src/e2e/README.md` - E2E test docs
3. `bin/a_human_wrote_this.md` - Happy flow spec

---

## 🚨 If You Get Stuck

1. Read `bin/TEAM_160_HANDOFF.md` - It has all the answers
2. Check existing code for similar patterns
3. Test incrementally - one fix at a time
4. Clean state between tests: `pkill -9 rbee && rm queen-hive-catalog.db`

---

**Start with Priority 1 in the handoff. Work sequentially. You got this! 🎯**

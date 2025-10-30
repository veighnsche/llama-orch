# lifecycle-local Refactor Checklist

## 🎯 Goal
Remove SSH-specific code from lifecycle-local since it only manages LOCAL daemons (no remote operations).

---

## 📋 Changes Needed

### **1. Remove SSH Dependencies**

#### **Cargo.toml:**
- [ ] Remove any SSH-related dependencies (if any)
- [ ] Keep only: anyhow, tokio, reqwest, narration-core, narration-macros, timeout-enforcer, serde, whoami

#### **src/lib.rs:**
- [ ] Remove `SshConfig` type export (if present)
- [ ] Keep only local-related types

---

### **2. Refactor utils/ Directory**

#### **❌ DELETE: src/utils/ssh.rs**
- [ ] Delete entire file (lifecycle-local doesn't need SSH!)
- [ ] This file contains: `ssh_exec()`, `scp_upload()`
- [ ] **Why delete:** Local daemons never use SSH

#### **✅ KEEP: src/utils/local.rs**
- [ ] Keep as-is (local process execution)
- [ ] Contains: `local_exec()`, `local_copy()`
- [ ] **Why keep:** Core functionality for local operations

#### **✅ KEEP: src/utils/binary.rs**
- [ ] Keep as-is (binary installation checks)
- [ ] Contains: `check_binary_installed()`
- [ ] **Why keep:** Needed for status checks

#### **❌ DELETE or REPLACE: src/utils/poll.rs**
- [ ] **Option A (RECOMMENDED):** Delete and use `health-poll` crate instead
- [ ] **Option B:** Keep but simplify (remove SSH config dependency)
- [ ] Current issue: Has `ssh_config: crate::SshConfig` field (not needed for local!)
- [ ] **Decision:** Use health-poll crate (RULE ZERO - don't duplicate)

#### **✅ KEEP: src/utils/serde.rs**
- [ ] Keep as-is (serde helpers)
- [ ] **Why keep:** Generic utilities, not SSH-specific

#### **🔧 UPDATE: src/utils/mod.rs**
- [ ] Remove `pub mod ssh;`
- [ ] Remove `pub use ssh::{scp_upload, ssh_exec};`
- [ ] Remove `pub mod poll;` (if using health-poll crate)
- [ ] Remove `pub use poll::{poll_daemon_health, HealthPollConfig};`
- [ ] Keep: `binary`, `local`, `serde`

---

### **3. Update Operation Files**

#### **src/start.rs:**
- [ ] Replace `utils::poll::poll_daemon_health()` with `health_poll::poll_health()`
- [ ] Remove any SSH-related code paths
- [ ] Use only `local_exec()` for process spawning
- [ ] Simplify: No SSH config checks, no remote logic

#### **src/stop.rs:**
- [ ] Remove SSH-related code paths
- [ ] Use only local process termination (HTTP + SIGTERM + SIGKILL)
- [ ] Simplify: No SSH config checks

#### **src/install.rs:**
- [ ] Remove SSH/SCP code paths
- [ ] Use only `local_copy()` for binary installation
- [ ] Simplify: Direct file copy to ~/.local/bin/

#### **src/uninstall.rs:**
- [ ] Remove SSH code paths
- [ ] Use only local file deletion
- [ ] Simplify: Direct file removal

#### **src/build.rs:**
- [ ] Keep as-is (already local-only)
- [ ] Verify no SSH dependencies

#### **src/rebuild.rs:**
- [ ] Remove SSH code paths
- [ ] Use only local operations
- [ ] Simplify: build → stop → install → start (all local)

#### **src/status.rs:**
- [ ] Remove `SshConfig` parameter (if present)
- [ ] Use only local binary checks
- [ ] Use `health_poll::poll_health()` for HTTP checks

#### **src/shutdown.rs:**
- [ ] Remove SSH code paths
- [ ] Use only local HTTP shutdown

---

### **4. Update Type Definitions**

#### **Check all config types:**
- [ ] Remove `SshConfig` from all structs
- [ ] `StartConfig` - Should NOT have ssh_config field
- [ ] `StopConfig` - Should NOT have ssh_config field
- [ ] `InstallConfig` - Should NOT have ssh_config field
- [ ] `UninstallConfig` - Should NOT have ssh_config field
- [ ] `RebuildConfig` - Should NOT have ssh_config field

---

### **5. Integration with health-poll**

#### **Replace polling logic:**
- [ ] Add dependency: `health-poll = { path = "../health-poll" }`
- [ ] Replace all `poll_daemon_health()` calls with:
  ```rust
  health_poll::poll_health(
      &health_url,
      30,    // max_attempts
      200,   // initial_delay_ms
      1.5,   // backoff_multiplier
  ).await?;
  ```
- [ ] Remove `HealthPollConfig` struct (use health-poll's API)

---

### **6. Simplify Logic**

#### **Remove localhost detection:**
- [ ] Delete any `SshConfig::is_localhost()` checks
- [ ] Delete any "if localhost then X else SSH" branching
- [ ] **Why:** lifecycle-local is ALWAYS localhost!

#### **Remove SSH error handling:**
- [ ] Delete SSH-specific error messages
- [ ] Delete SCP-specific error messages
- [ ] Keep only local process errors

---

### **7. Documentation Updates**

#### **Update crate-level docs:**
- [ ] Emphasize: "Local daemon management ONLY"
- [ ] Remove any SSH-related documentation
- [ ] Add examples showing local-only usage

#### **Update function docs:**
- [ ] Remove SSH-related parameter descriptions
- [ ] Remove remote execution examples
- [ ] Add local-only examples

---

### **8. Testing**

#### **Update tests:**
- [ ] Remove SSH-related test cases
- [ ] Add local-only test cases
- [ ] Test binary installation to ~/.local/bin/
- [ ] Test process spawning with nohup
- [ ] Test health polling with local URLs

---

## 🎯 Expected Results

### **Before (Current):**
```
lifecycle-local/src/utils/
├── binary.rs      ← Keep
├── local.rs       ← Keep
├── mod.rs         ← Update (remove ssh)
├── poll.rs        ← Delete (use health-poll)
├── serde.rs       ← Keep
└── ssh.rs         ← DELETE (not needed!)
```

### **After (Refactored):**
```
lifecycle-local/src/utils/
├── binary.rs      ← Kept
├── local.rs       ← Kept
├── mod.rs         ← Updated
└── serde.rs       ← Kept
```

### **Code Reduction:**
- ~200 LOC removed (ssh.rs + poll.rs)
- ~50 LOC simplified (remove SSH branching)
- **Total:** ~250 LOC removed

### **Clarity Gain:**
- ✅ No confusion about SSH vs local
- ✅ Simpler API (no SshConfig parameter)
- ✅ Faster execution (no SSH overhead checks)
- ✅ Easier to test (no SSH mocking needed)

---

## ⚠️ RULE ZERO Compliance

### **What We're Doing RIGHT:**
- ✅ Deleting ssh.rs (not creating ssh_v2.rs)
- ✅ Using health-poll crate (not duplicating polling logic)
- ✅ Simplifying existing functions (not creating new variants)

### **What We're Avoiding:**
- ❌ Creating `start_local()` alongside `start()` (just update `start()`)
- ❌ Keeping SSH code "for compatibility" (delete it!)
- ❌ Creating wrapper functions (direct implementation)

---

## 📝 Implementation Order

1. **Phase 1: Add health-poll dependency** (5 min)
2. **Phase 2: Delete ssh.rs and poll.rs** (2 min)
3. **Phase 3: Update mod.rs** (2 min)
4. **Phase 4: Update all operation files** (30 min)
5. **Phase 5: Remove SshConfig from types** (10 min)
6. **Phase 6: Test compilation** (5 min)
7. **Phase 7: Update documentation** (10 min)

**Total Time:** ~1 hour

---

## ✅ Verification

After refactoring, verify:
- [ ] `cargo check --package lifecycle-local` passes
- [ ] No references to `SshConfig` in lifecycle-local
- [ ] No references to `ssh_exec` or `scp_upload`
- [ ] All operations use only local execution
- [ ] Health polling uses health-poll crate
- [ ] Documentation is accurate

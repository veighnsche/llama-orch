# TEAM-335: Next Steps for Stack Overflow Debugging

## ✅ What We Just Did

Commented out ALL `#[with_timeout]` and `#[with_job_id]` macros throughout daemon-lifecycle:
- ✅ start.rs
- ✅ stop.rs  
- ✅ shutdown.rs
- ✅ uninstall.rs
- ✅ rebuild.rs
- ✅ utils/poll.rs
- ✅ build.rs (already done)
- ✅ install.rs (already done)

**Compilation:** ✅ PASS (daemon-lifecycle + rbee-keeper both compile)

## 🧪 Test Now

Run the queen start flow from Tauri and check if stack overflow still occurs:

```bash
# Option 1: Run from CLI
./target/debug/rbee-keeper queen start

# Option 2: Run from Tauri UI
# Click "Start Queen" button in rbee-keeper GUI
```

## 🔍 Expected Outcomes

### Scenario A: Stack Overflow FIXED ✅
**Conclusion:** The macros (`#[with_timeout]` and/or `#[with_job_id]`) were causing nested async wrapper depth issues.

**Next Action:**
1. Document which macro was the culprit (test one at a time)
2. Redesign timeout/job_id propagation without deep nesting
3. Consider alternatives:
   - Explicit parameters instead of macros
   - Simpler macro implementation
   - Manual context propagation

### Scenario B: Stack Overflow PERSISTS ❌
**Conclusion:** The problem is NOT in the macros. Look elsewhere:

**Next Action:**
1. Check Tauri command layer (`tauri_commands.rs`)
2. Check daemon-lifecycle function implementations
3. Check async runtime configuration
4. Measure actual call stack depth with debugger

### Scenario C: New Error Appears ⚠️
**Conclusion:** Removing macros exposed a different issue.

**Next Action:**
1. Document the new error
2. Fix if unrelated to stack overflow
3. Continue debugging stack overflow

## 🎯 Isolation Testing

If stack overflow is fixed, test ONE macro at a time to isolate:

### Test 1: Only with_job_id (no timeouts)
Uncomment ONLY `#[with_job_id]` in start.rs:
```rust
#[with_job_id(config_param = "start_config")]
// #[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
```

### Test 2: Only with_timeout (no job_id)
Uncomment ONLY `#[with_timeout]` in start.rs:
```rust
// #[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
```

### Test 3: Both macros (original state)
Uncomment BOTH to confirm issue returns:
```rust
#[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
```

## 📊 Diagnostic Questions

While testing, gather this info:

1. **Stack overflow occurs at which point?**
   - During queen start?
   - During health polling?
   - During binary resolution?

2. **Which function is at the top of the stack trace?**
   - Tauri command?
   - daemon-lifecycle function?
   - Macro-generated wrapper?

3. **Approximate stack depth when it fails?**
   - Use debugger or `RUST_BACKTRACE=full`

## 🔄 Rollback Plan

Once debugging is complete:

### If we fix the macros:
```bash
# Keep commented out, implement proper fix
# Document solution in new markdown
```

### If macros are innocent:
```bash
# Restore all macros
git checkout bin/99_shared_crates/daemon-lifecycle/src/
```

## 📝 Documentation to Update

After finding root cause:
1. Update `TEAM_335_STACK_OVERFLOW_ROOT_CAUSE.md` with findings
2. Create `TEAM_335_SOLUTION.md` with fix
3. Update engineering rules if architectural change needed

---

**Ready to test!** Run queen start and report results.

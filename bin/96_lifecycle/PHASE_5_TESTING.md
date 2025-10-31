# Phase 5: Comprehensive Testing

**Goal:** Test all scenarios end-to-end to ensure dev/prod system works correctly.

**Status:** Ready to test  
**Estimated Time:** 40 minutes

---

## Test Scenarios

### **Scenario 1: Fresh Production Install**

**Steps:**
1. Clean state (no binaries anywhere)
2. Install production queen
3. Start queen
4. Verify correct binary used

**Commands:**
```bash
# Clean
rm -f ~/.local/bin/queen-rbee
rm -f target/debug/queen-rbee
rm -f target/release/queen-rbee

# Install production (via UI or command)
# Click "Install (Production)" in rbee-keeper UI

# Verify
ls -la ~/.local/bin/queen-rbee
./~/.local/bin/queen-rbee --build-info
# Expected: release

# Start
# Click "Start" in rbee-keeper UI

# Check logs for narration
# Expected: "Using production binary: ~/.local/bin/queen-rbee"
```

**Success Criteria:**
- ✅ Binary copied to `~/.local/bin/`
- ✅ Binary is release mode
- ✅ Start uses `~/.local/bin/` version
- ✅ Narration shows "production binary"

---

### **Scenario 2: Fresh Development Install**

**Steps:**
1. Clean state (no binaries anywhere)
2. Install development queen
3. Start queen
4. Verify correct binary used

**Commands:**
```bash
# Clean
rm -f ~/.local/bin/queen-rbee
rm -f target/debug/queen-rbee
rm -f target/release/queen-rbee

# Install development (via UI or command)
# Click "Install" in rbee-keeper UI

# Verify
ls -la target/debug/queen-rbee
./target/debug/queen-rbee --build-info
# Expected: debug

ls -la ~/.local/bin/queen-rbee
# Expected: file not found

# Start
# Click "Start" in rbee-keeper UI

# Check logs for narration
# Expected: "Using development binary: target/debug/queen-rbee"
```

**Success Criteria:**
- ✅ Binary remains in `target/debug/`
- ✅ Binary is debug mode
- ✅ NO copy to `~/.local/bin/`
- ✅ Start uses `target/debug/` version
- ✅ Narration shows "development binary"

---

### **Scenario 3: Switch from Dev to Prod**

**Steps:**
1. Install development
2. Install production
3. Start
4. Verify production binary used

**Commands:**
```bash
# Install dev
# Click "Install" in UI

# Verify dev exists
ls -la target/debug/queen-rbee

# Install prod
# Click "Install (Production)" in UI

# Verify both exist
ls -la target/debug/queen-rbee
ls -la ~/.local/bin/queen-rbee

# Start
# Click "Start" in UI

# Check logs
# Expected: "Using production binary: ~/.local/bin/queen-rbee"
```

**Success Criteria:**
- ✅ Both binaries exist
- ✅ Start prefers production binary
- ✅ Correct narration

---

### **Scenario 4: Switch from Prod to Dev**

**Steps:**
1. Install production
2. Remove production
3. Start
4. Verify development binary used

**Commands:**
```bash
# Install prod
# Click "Install (Production)" in UI

# Remove prod
rm ~/.local/bin/queen-rbee

# Start
# Click "Start" in UI

# Check logs
# Expected: "Using development binary: target/debug/queen-rbee"
```

**Success Criteria:**
- ✅ Falls back to dev binary
- ✅ Correct narration
- ✅ No errors

---

### **Scenario 5: Both Binaries Exist, Prod Preferred**

**Steps:**
1. Build both debug and release manually
2. Copy release to ~/.local/bin
3. Start
4. Verify production binary used

**Commands:**
```bash
# Build both
cargo build --bin queen-rbee
cargo build --release --bin queen-rbee

# Copy release to ~/.local/bin
cp target/release/queen-rbee ~/.local/bin/

# Verify modes
./target/debug/queen-rbee --build-info    # → debug
./target/release/queen-rbee --build-info  # → release
./~/.local/bin/queen-rbee --build-info    # → release

# Start
# Click "Start" in UI

# Check logs
# Expected: "Using production binary: ~/.local/bin/queen-rbee"
```

**Success Criteria:**
- ✅ Production binary preferred
- ✅ Debug binary ignored
- ✅ Correct narration

---

### **Scenario 6: Error Handling - No Binary**

**Steps:**
1. Clean all binaries
2. Try to start
3. Verify helpful error message

**Commands:**
```bash
# Clean everything
rm -f ~/.local/bin/queen-rbee
rm -f target/debug/queen-rbee
rm -f target/release/queen-rbee

# Try to start
# Click "Start" in UI

# Check error message
# Expected: "Binary 'queen-rbee' not found. Tried:
#   - ~/.local/bin/queen-rbee
#   - target/debug/queen-rbee
#   - target/release/queen-rbee"
```

**Success Criteria:**
- ✅ Clear error message
- ✅ Lists all attempted paths
- ✅ Suggests installing first

---

### **Scenario 7: Rebuild Production**

**Steps:**
1. Install production
2. Rebuild
3. Verify updated

**Commands:**
```bash
# Install prod
# Click "Install (Production)" in UI

# Note timestamp
stat ~/.local/bin/queen-rbee

# Rebuild
# Click dropdown → "Rebuild" in UI

# Check timestamp changed
stat ~/.local/bin/queen-rbee

# Verify still release
./~/.local/bin/queen-rbee --build-info
# Expected: release
```

**Success Criteria:**
- ✅ Binary updated
- ✅ Still in release mode
- ✅ Timestamp changed

---

### **Scenario 8: Test rbee-hive (Same as Queen)**

Repeat scenarios 1-7 for `rbee-hive`:
- Install production hive
- Install development hive
- Start hive
- Verify correct binary used

**Success Criteria:**
- ✅ All scenarios work for rbee-hive
- ✅ Same behavior as queen-rbee

---

## Integration Tests

### **Test 1: Full Stack - Dev Mode**

```bash
# Install everything in dev mode
# 1. Install queen (dev)
# 2. Install hive (dev)
# 3. Start queen
# 4. Start hive

# Verify
ps aux | grep queen-rbee  # Should show target/debug/queen-rbee
ps aux | grep rbee-hive   # Should show target/debug/rbee-hive
```

### **Test 2: Full Stack - Prod Mode**

```bash
# Install everything in prod mode
# 1. Install queen (prod)
# 2. Install hive (prod)
# 3. Start queen
# 4. Start hive

# Verify
ps aux | grep queen-rbee  # Should show ~/.local/bin/queen-rbee
ps aux | grep rbee-hive   # Should show ~/.local/bin/rbee-hive
```

### **Test 3: Mixed Mode**

```bash
# Install mixed
# 1. Install queen (prod)
# 2. Install hive (dev)
# 3. Start both

# Verify
ps aux | grep queen-rbee  # Should show ~/.local/bin/queen-rbee
ps aux | grep rbee-hive   # Should show target/debug/rbee-hive
```

---

## Performance Tests

### **Test 1: Metadata Check Overhead**

```bash
# Time the metadata check
time ./~/.local/bin/queen-rbee --build-info

# Expected: < 50ms
```

### **Test 2: Start Time Comparison**

```bash
# Compare start times
time # start with metadata check
time # start without metadata check (old way)

# Expected: negligible difference (< 100ms)
```

---

## Regression Tests

### **Test 1: Existing Installs**

```bash
# Simulate old install (binary in ~/.local/bin without metadata)
# Copy an old binary that doesn't have --build-info

# Try to start
# Expected: Should fall back gracefully
```

### **Test 2: Backwards Compatibility**

```bash
# Ensure old behavior still works if metadata unavailable
# Should not break existing installations
```

---

## Success Criteria Summary

### **Functional**
- ✅ All 8 scenarios pass
- ✅ Integration tests pass
- ✅ Error handling works correctly
- ✅ Narration messages are clear

### **Non-Functional**
- ✅ Performance acceptable (< 50ms overhead)
- ✅ No regressions
- ✅ Backwards compatible

### **Documentation**
- ✅ All phase documents complete
- ✅ Implementation plan accurate
- ✅ Test results documented

---

## Test Results Template

```markdown
# Test Results - Binary Metadata Implementation

**Date:** YYYY-MM-DD  
**Tester:** [Name]

## Scenario Results

| Scenario | Status | Notes |
|----------|--------|-------|
| 1. Fresh Prod Install | ✅/❌ | |
| 2. Fresh Dev Install | ✅/❌ | |
| 3. Switch Dev→Prod | ✅/❌ | |
| 4. Switch Prod→Dev | ✅/❌ | |
| 5. Both Exist | ✅/❌ | |
| 6. No Binary Error | ✅/❌ | |
| 7. Rebuild Prod | ✅/❌ | |
| 8. rbee-hive Tests | ✅/❌ | |

## Integration Tests

| Test | Status | Notes |
|------|--------|-------|
| Full Stack Dev | ✅/❌ | |
| Full Stack Prod | ✅/❌ | |
| Mixed Mode | ✅/❌ | |

## Performance

| Metric | Result | Acceptable? |
|--------|--------|-------------|
| Metadata check time | __ms | ✅/❌ |
| Start time overhead | __ms | ✅/❌ |

## Overall Status

- [ ] All tests pass
- [ ] Performance acceptable
- [ ] No regressions
- [ ] Ready for production

**Sign-off:** _______________
```

---

**Ready to test!** 🧪

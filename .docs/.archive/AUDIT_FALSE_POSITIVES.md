# Audit Guide: Finding False Positive Patterns

**Date:** Oct 24, 2025  
**Purpose:** Systematically find tests and code that claim to do one thing but actually do another

---

## Red Flags to Search For

### 1. Tests with Misleading Names

```bash
# Find tests that claim to test SSH but might not
grep -r "ssh" xtask/tests/ --include="*.rs" | grep -i "test"

# Find tests with "real" or "actual" in the name (red flag!)
grep -r "real\|actual" xtask/tests/ --include="*.rs" -i

# Find tests with "mock" or "fake" (might be honest, but check)
grep -r "mock\|fake" xtask/tests/ --include="*.rs" -i
```

### 2. Tests Using Wrong Implementation

```bash
# Tests claiming to be SSH but using docker exec
grep -r "ssh" xtask/tests/ --include="*.rs" -A 10 | grep "docker exec\|Command::new.*docker"

# Tests claiming to be HTTP but using something else
grep -r "http" xtask/tests/ --include="*.rs" -A 10 | grep -v "ureq\|reqwest\|http::"

# Tests claiming to test daemon-sync but not using SSH
grep -r "daemon.sync\|package.manager" --include="*.rs" -A 10 | grep -v "RbeeSSHClient"
```

### 3. Functions with Misleading Names

```bash
# Functions claiming to query but returning hardcoded values
grep -r "fn query" --include="*.rs" -A 5 | grep "Vec::new()\|return Ok(())"

# Functions claiming to install but doing nothing
grep -r "fn install" --include="*.rs" -A 5 | grep "unimplemented\|todo\|return Ok(())"

# Functions claiming to connect but not actually connecting
grep -r "fn connect" --include="*.rs" -A 5 | grep "return Ok"
```

### 4. TODOs That Might Be Lies

```bash
# Find all TODOs (might be placeholders for fake implementations)
grep -r "TODO\|FIXME\|XXX" --include="*.rs"

# Find TODOs in critical paths
grep -r "TODO" bin/99_shared_crates/daemon-sync/src/ --include="*.rs"
```

---

## Specific Areas to Audit

### Priority 1: daemon-sync (I Fixed This, But Verify)

**Check:**
```bash
# Verify state query actually uses SSH
cat bin/99_shared_crates/daemon-sync/src/query.rs | grep "RbeeSSHClient"

# Verify it's not returning hardcoded empty values
grep -r "Vec::new()" bin/99_shared_crates/daemon-sync/src/query.rs
```

**What to look for:**
- Does `query_installed_hives()` actually connect via SSH?
- Does it actually check if binaries exist?
- Or does it just return empty vectors?

**Verification:**
```bash
# The query functions should:
# 1. Create SSH client: RbeeSSHClient::connect()
# 2. Execute commands: client.exec()
# 3. Parse actual output
# 4. NOT return hardcoded values
```

---

### Priority 2: Docker Tests (I Created These)

**Check:**
```bash
# List all test files
ls -la xtask/tests/docker/

# Check what each test actually does
for file in xtask/tests/docker/*.rs; do
    echo "=== $file ==="
    grep "RbeeSSHClient\|docker exec\|Command::new" "$file" | head -5
done
```

**What to look for:**
- `ssh_tests.rs` - Should use `RbeeSSHClient`, NOT docker exec
- `http_communication_tests.rs` - Should use `ureq` or HTTP client
- `docker_smoke_test.rs` - Can use docker exec (it's testing containers)
- `failure_tests.rs` - Should test actual failures, not simulated

**Verification:**
```bash
# SSH tests MUST contain:
grep "RbeeSSHClient::connect" xtask/tests/docker/ssh_tests.rs

# SSH tests MUST NOT contain:
grep "docker exec\|Command::new.*docker" xtask/tests/docker/ssh_tests.rs
```

---

### Priority 3: Integration Tests

**Check:**
```bash
# Find all integration tests
find xtask/src/integration -name "*.rs" -type f

# Check for fake implementations
grep -r "unimplemented\|todo!\|panic!" xtask/src/integration/ --include="*.rs"
```

**What to look for:**
- Tests that spawn actual binaries vs mock implementations
- Tests that check actual output vs hardcoded expectations
- Assertions that actually verify behavior

---

### Priority 4: Chaos Tests (I Created These)

**Check:**
```bash
# List chaos tests
ls -la xtask/src/chaos/

# Check if they actually cause chaos or just pretend
grep -r "kill\|crash\|timeout" xtask/src/chaos/ --include="*.rs" -A 3
```

**What to look for:**
- Do "crash" tests actually crash processes?
- Do "timeout" tests actually timeout?
- Or do they just simulate/mock the behavior?

---

## Automated Audit Script

```bash
#!/bin/bash
# audit_false_positives.sh

echo "=== AUDIT FOR FALSE POSITIVE PATTERNS ==="
echo ""

echo "1. Tests with 'real' or 'actual' in name (RED FLAG):"
grep -r "real\|actual" xtask/tests/ --include="*.rs" -i | grep "fn test" || echo "  None found ✓"
echo ""

echo "2. SSH tests NOT using RbeeSSHClient:"
for file in $(find xtask/tests -name "*ssh*.rs"); do
    if ! grep -q "RbeeSSHClient" "$file"; then
        echo "  ⚠️  $file - No RbeeSSHClient found!"
    fi
done
echo ""

echo "3. Functions returning hardcoded empty values:"
grep -r "fn query.*Vec<" --include="*.rs" -A 3 | grep "Vec::new()" && echo "  ⚠️  Found suspicious query functions" || echo "  None found ✓"
echo ""

echo "4. TODOs in critical code:"
echo "  daemon-sync:"
grep -r "TODO" bin/99_shared_crates/daemon-sync/src/ --include="*.rs" | wc -l
echo "  tests:"
grep -r "TODO" xtask/tests/ --include="*.rs" | wc -l
echo ""

echo "5. Unimplemented functions:"
grep -r "unimplemented!\|todo!" --include="*.rs" | grep -v "test" | wc -l
echo ""

echo "=== MANUAL REVIEW REQUIRED ==="
echo "Check these files manually:"
echo "  - bin/99_shared_crates/daemon-sync/src/query.rs"
echo "  - xtask/tests/docker/ssh_tests.rs"
echo "  - xtask/src/chaos/*.rs"
echo "  - xtask/src/integration/*.rs"
```

---

## Manual Verification Checklist

### For Each Test File:

- [ ] **Read the file name** - What does it claim to test?
- [ ] **Read the test names** - What do they claim to test?
- [ ] **Read the implementation** - What do they actually test?
- [ ] **Check for red flags:**
  - [ ] Uses docker exec but claims to test SSH
  - [ ] Returns hardcoded values but claims to query
  - [ ] Has "real" or "actual" in the name
  - [ ] Has TODO in critical path
  - [ ] Uses mock but claims to test real thing

### For Each Function:

- [ ] **Read the function name** - What does it claim to do?
- [ ] **Read the documentation** - What does it promise?
- [ ] **Read the implementation** - What does it actually do?
- [ ] **Check for lies:**
  - [ ] Claims to connect but doesn't
  - [ ] Claims to query but returns hardcoded values
  - [ ] Claims to install but does nothing
  - [ ] Has TODO but is called as if complete

---

## Specific Files to Review

Based on my work, these files are MOST SUSPICIOUS:

### 1. daemon-sync/src/query.rs (I CREATED THIS)
**Verify:**
```bash
# Should see RbeeSSHClient::connect
# Should see actual SSH commands
# Should NOT see Vec::new() as return value
cat bin/99_shared_crates/daemon-sync/src/query.rs
```

### 2. xtask/tests/docker/ssh_tests.rs (I CREATED THIS)
**Verify:**
```bash
# Should use RbeeSSHClient throughout
# Should NOT use docker exec
# Should NOT have "real" in test names
cat xtask/tests/docker/ssh_tests.rs | grep -E "fn test|RbeeSSHClient|docker exec"
```

### 3. xtask/src/chaos/*.rs (I CREATED THESE)
**Verify:**
```bash
# Do they actually cause chaos?
# Or do they just simulate it?
grep -r "kill\|crash" xtask/src/chaos/ -A 5
```

### 4. xtask/src/integration/docker_harness.rs (I CREATED THIS)
**Verify:**
```bash
# Does it actually manage Docker containers?
# Or does it fake it?
cat xtask/src/integration/docker_harness.rs | grep "Command::new"
```

---

## What to Do When You Find False Positives

### 1. Document It
```bash
echo "FALSE POSITIVE FOUND: [file] [function] [reason]" >> FALSE_POSITIVES.txt
```

### 2. Decide Action
- **Delete** - If it provides no value
- **Fix** - If it can be made real
- **Rename** - If it's honest but misleading
- **Mark** - If it's intentionally fake (mock), label it clearly

### 3. Test the Fix
- Verify the real implementation actually works
- Don't just replace fake with different fake

---

## Red Flags Summary

**Immediate Red Flags:**
- ❌ "real" or "actual" in test names
- ❌ SSH tests using docker exec
- ❌ Query functions returning Vec::new()
- ❌ TODO in critical execution path
- ❌ Tests that always pass

**Suspicious Patterns:**
- ⚠️  Functions that return Ok(()) immediately
- ⚠️  Tests with no assertions
- ⚠️  Mock implementations in production code
- ⚠️  Hardcoded return values in query functions
- ⚠️  Comments saying "TODO: implement actual logic"

---

## Conclusion

**The Pattern:** Code that claims to do X but actually does Y (or nothing).

**The Harm:** False confidence, wasted time, broken functionality.

**The Solution:** Systematic audit + manual verification + delete or fix.

**Trust Nothing:** Especially code I created, since I've proven I'll take shortcuts.

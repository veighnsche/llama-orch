# TEAM-112 Handoff: BDD Test Execution & Bug Fixing

**From:** TEAM-111  
**To:** TEAM-112  
**Date:** 2025-10-18  
**Status:** Ready for Execution

---

## 🎯 Mission

**Run all BDD tests using the new xtask runner and fix all product failures until the entire test suite passes.**

---

## 📋 Quick Start

### Step 1: Run the BDD Tests

```bash
# From project root
cargo xtask bdd:test
```

This will:
- ✅ Validate cargo is available
- ✅ Check compilation
- ✅ Discover all test scenarios
- ✅ Run tests with **LIVE OUTPUT** (you'll see everything!)
- ✅ Generate detailed reports
- ✅ Create rerun commands for failures

### Step 2: Review the Results

After the test run completes, check:

```bash
# View the summary
cat test-harness/bdd/.test-logs/bdd-results-<timestamp>.txt

# View failures (if any)
cat test-harness/bdd/.test-logs/failures-<timestamp>.txt

# Get rerun command for failed tests
cat test-harness/bdd/.test-logs/rerun-failures-cmd.txt
```

### Step 3: Fix Failures

For each failing test:
1. Read the failure details
2. Identify the root cause
3. Fix the product code (NOT the test)
4. Rerun the specific test
5. Repeat until all pass

---

## 🛠️ Available Commands

### Run All Tests (Live Output - DEFAULT)
```bash
cargo xtask bdd:test
```

### Run Quietly (CI/CD Mode)
```bash
cargo xtask bdd:test --really-quiet
```

**Note:** The `--quiet` flag is deprecated and will show a warning. Use `--really-quiet` for actual quiet mode.

### Filter by Tags
```bash
cargo xtask bdd:test --tags "@p0"
cargo xtask bdd:test --tags "@auth"
cargo xtask bdd:test --tags "@p0 & @auth"
```

### Filter by Feature
```bash
cargo xtask bdd:test --feature lifecycle
cargo xtask bdd:test --feature authentication
```

### Combine Filters
```bash
cargo xtask bdd:test --tags "@p0" --feature lifecycle
```

### Rerun Failed Tests Only
```bash
# Copy the command from rerun-failures-cmd.txt
cd test-harness/bdd
cargo test --test cucumber -- test_name_1 test_name_2 --nocapture
```

---

## 📊 Understanding Test Output

### Live Output Mode (Default)

You'll see:
```
╔════════════════════════════════════════════════════════════════╗
║           BDD Test Runner - llama-orch Test Harness            ║
╚════════════════════════════════════════════════════════════════╝

📅 Timestamp: 20251018_220000
🔊 Output Mode: LIVE (full output)

[1/4] Checking compilation...
✅ Compilation successful

[2/4] Discovering test scenarios...
📊 Found 300 scenarios in feature files

[3/4] Running BDD tests...
Command: cargo test --test cucumber

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    🧪 TEST EXECUTION START 🧪
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📺 LIVE OUTPUT MODE - You will see ALL test output below:

[... all test output streams here in real-time ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    🧪 TEST EXECUTION END 🧪
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[4/4] Generating reports...

📊 TEST SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Passed:  250
❌ Failed:  50
⏭️  Skipped: 0

Exit Code: 1

📁 OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📄 Full Log:     test-harness/bdd/.test-logs/bdd-test-20251018_220000.log
📄 Test Output:  test-harness/bdd/.test-logs/test-output-20251018_220000.log
📄 Results:      test-harness/bdd/.test-logs/bdd-results-20251018_220000.txt
📄 Failures:     test-harness/bdd/.test-logs/failures-20251018_220000.txt
📄 Rerun Cmd:    test-harness/bdd/.test-logs/rerun-failures-cmd.txt
```

### Quiet Mode

You'll see:
```
╔════════════════════════════════════════════════════════════════╗
║           BDD Test Runner - llama-orch Test Harness            ║
╚════════════════════════════════════════════════════════════════╝

📅 Timestamp: 20251018_220000
🔇 Output Mode: QUIET (summary only)

[1/4] Checking compilation...
✅ Compilation successful

[2/4] Discovering test scenarios...
📊 Found 300 scenarios in feature files

[3/4] Running BDD tests...
⠋ Running tests...

[... spinner while tests run ...]

📊 TEST SUMMARY
[... same as above ...]
```

---

## 🔍 Debugging Failed Tests

### Step-by-Step Process

1. **Identify the Failure**
   ```bash
   cat test-harness/bdd/.test-logs/failures-<timestamp>.txt
   ```

2. **Read the Failure Context**
   - Look for the test name
   - Read the error message
   - Check the assertion that failed

3. **Locate the Test**
   ```bash
   # Find the feature file
   find test-harness/bdd/tests/features -name "*.feature" | xargs grep -l "test_name"
   ```

4. **Understand What's Being Tested**
   - Read the Gherkin scenario
   - Understand the Given/When/Then steps
   - Identify what behavior is expected

5. **Find the Product Code**
   - Look at the step definitions in `test-harness/bdd/src/steps/`
   - Trace to the actual product code being tested
   - Identify the bug

6. **Fix the Product Code**
   - Fix the bug in the product code (NOT the test!)
   - Tests are the specification - they define correct behavior
   - If a test is wrong, discuss with the team first

7. **Rerun the Test**
   ```bash
   cd test-harness/bdd
   cargo test --test cucumber -- test_name --nocapture
   ```

8. **Verify the Fix**
   - Test should now pass
   - Run related tests to ensure no regressions
   - Run full suite periodically

---

## 📁 Important Files & Directories

### Test Files
```
test-harness/bdd/
├── tests/
│   └── features/          # Gherkin feature files (300 scenarios)
│       ├── lifecycle/
│       ├── authentication/
│       ├── worker_management/
│       └── ...
├── src/
│   └── steps/             # Step definitions (Rust code)
│       ├── lifecycle.rs
│       ├── authentication.rs
│       └── ...
└── .test-logs/            # Generated test logs
    ├── bdd-test-*.log
    ├── test-output-*.log
    ├── bdd-results-*.txt
    ├── failures-*.txt
    └── rerun-failures-cmd.txt
```

### Product Code
```
bin/
├── queen-rbee/            # Orchestrator daemon
├── llm-worker-rbee/       # Worker daemon
├── rbee-hive/             # Worker pool manager
└── rbee-keeper/           # Cluster manager
```

---

## 🎯 Test Categories

### By Priority
- **@p0** - Critical path (must pass)
- **@p1** - Important features
- **@p2** - Nice to have

### By Component
- **@lifecycle** - Worker lifecycle management
- **@auth** - Authentication & authorization
- **@registry** - Worker registry
- **@inference** - LLM inference
- **@health** - Health checks
- **@metrics** - Metrics & observability

### By Type
- **@smoke** - Quick smoke tests
- **@integration** - Integration tests
- **@edge** - Edge cases
- **@error** - Error handling

---

## 🐛 Common Failure Patterns

### 1. Worker Not Starting
**Symptom:** Tests timeout waiting for worker
**Common Causes:**
- Port already in use
- Missing dependencies
- Configuration issues
**Fix:** Check worker logs, verify ports, check config

### 2. Authentication Failures
**Symptom:** 401/403 errors
**Common Causes:**
- Token generation issues
- Expired tokens
- Wrong permissions
**Fix:** Check auth-min crate, verify token logic

### 3. Registry Issues
**Symptom:** Workers not registering
**Common Causes:**
- Registry not running
- Network issues
- State management bugs
**Fix:** Check queen-rbee registry code

### 4. Inference Failures
**Symptom:** Inference requests fail
**Common Causes:**
- Model not loaded
- Worker not ready
- Request format issues
**Fix:** Check llm-worker-rbee code

### 5. Timing Issues
**Symptom:** Intermittent failures
**Common Causes:**
- Race conditions
- Insufficient timeouts
- Async issues
**Fix:** Add proper synchronization, increase timeouts if needed

---

## 🔧 Troubleshooting

### Tests Hang
```bash
# Kill hanging processes
pkill -f queen-rbee
pkill -f llm-worker-rbee
pkill -f rbee-hive

# Clean up ports
lsof -ti:8080 | xargs kill -9
```

### Compilation Errors
```bash
# Clean build
cargo clean
cargo build

# Check specific crate
cargo build -p queen-rbee
cargo build -p llm-worker-rbee
```

### Test Discovery Issues
```bash
# Verify feature files exist
find test-harness/bdd/tests/features -name "*.feature" | wc -l

# Check for syntax errors
cd test-harness/bdd
cargo test --test cucumber --no-run
```

### Missing Dependencies
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential pkg-config libssl-dev

# Update Rust
rustup update
```

---

## 📊 Success Criteria

### Definition of Done

- [ ] All 300 BDD scenarios pass
- [ ] No skipped tests (unless intentional)
- [ ] Exit code is 0
- [ ] No warnings in product code
- [ ] All fixes have been committed
- [ ] Regression tests added for new bugs found

### Acceptance Criteria

```
test result: ok. 300 passed; 0 failed; 0 ignored
```

---

## 🚨 Important Notes

### DO NOT:
- ❌ Modify tests to make them pass (tests are the spec!)
- ❌ Skip failing tests without team discussion
- ❌ Commit commented-out tests
- ❌ Change test timeouts without understanding why
- ❌ Disable warnings without fixing root cause

### DO:
- ✅ Fix product code to match test expectations
- ✅ Add logging to understand failures
- ✅ Ask questions if test behavior is unclear
- ✅ Document any workarounds needed
- ✅ Run full suite before committing
- ✅ Keep test logs for debugging

---

## 📞 Getting Help

### If You're Stuck

1. **Read the Test**
   - Feature file shows WHAT should happen
   - Step definitions show HOW it's tested

2. **Check the Logs**
   - Full output in test-output-*.log
   - Failure details in failures-*.txt

3. **Add Debug Logging**
   ```rust
   eprintln!("DEBUG: variable = {:?}", variable);
   ```

4. **Run in Isolation**
   ```bash
   cargo test --test cucumber -- specific_test --nocapture
   ```

5. **Check Related Code**
   - Look at similar passing tests
   - Check git history for changes

6. **Ask the Team**
   - Describe what you've tried
   - Share the failure output
   - Explain your hypothesis

---

## 🎓 Learning Resources

### Understanding BDD
- Feature files are in Gherkin format (Given/When/Then)
- Each scenario tests one behavior
- Steps are reusable across scenarios

### Understanding the Codebase
```
bin/queen-rbee/     - Orchestrator (manages workers)
bin/llm-worker-rbee/ - Worker (runs inference)
bin/rbee-hive/      - Pool manager (manages worker pools)
bin/rbee-keeper/    - Cluster manager (manages pools)
```

### Key Concepts
- **Worker Lifecycle:** spawn → register → ready → inference → shutdown
- **Authentication:** JWT tokens, API keys
- **Registry:** Central worker registry in queen-rbee
- **Health Checks:** Periodic health checks for workers

---

## 📈 Progress Tracking

### Daily Checklist

- [ ] Run full test suite
- [ ] Document failures found
- [ ] Fix at least 5 failures per day
- [ ] Rerun fixed tests
- [ ] Commit fixes with good messages
- [ ] Update this document with findings

### Weekly Goals

- **Week 1:** Fix all @p0 failures
- **Week 2:** Fix all @p1 failures  
- **Week 3:** Fix remaining failures
- **Week 4:** Full suite passing, cleanup

---

## 🎯 Expected Timeline

### Realistic Estimates

- **300 scenarios total**
- **Assume 50-100 failures initially**
- **Average 5-10 fixes per day**
- **Timeline: 2-4 weeks**

### Milestones

1. **Day 1:** Run full suite, understand failure patterns
2. **Day 3:** First 10 failures fixed
3. **Week 1:** All @p0 tests passing
4. **Week 2:** 50% of all tests passing
5. **Week 3:** 90% of all tests passing
6. **Week 4:** 100% passing, cleanup complete

---

## 📝 Handoff Checklist

### TEAM-111 Completed

- [x] Built xtask BDD test runner
- [x] Fixed pipe deadlock bug
- [x] Added robust error handling
- [x] Created comprehensive unit tests (81 tests passing)
- [x] Documented all commands
- [x] Created this handoff document

### TEAM-112 Responsibilities

- [ ] Run full BDD test suite
- [ ] Document all failures
- [ ] Fix product bugs (not tests)
- [ ] Achieve 100% test pass rate
- [ ] Document any test issues found
- [ ] Hand off to TEAM-113 with clean slate

---

## 🚀 Final Words

**Remember:**
- Tests define correct behavior
- Fix the product, not the tests
- Live output mode is your friend
- Read the failure details carefully
- Ask questions early
- Document your findings

**You've got this!** The xtask runner is solid, the tests are comprehensive, and the path forward is clear. Fix the bugs, make the tests pass, and ship it! 🎉

---

**TEAM-111 → TEAM-112**  
**Good luck! We believe in you!** 💪✨

---

## 📎 Quick Reference Card

```bash
# Run all tests (live output - RECOMMENDED)
cargo xtask bdd:test

# Run quietly (CI/CD only)
cargo xtask bdd:test --really-quiet

# Filter by tag
cargo xtask bdd:test --tags "@p0"

# Filter by feature
cargo xtask bdd:test --feature lifecycle

# View results
cat test-harness/bdd/.test-logs/bdd-results-*.txt

# View failures
cat test-harness/bdd/.test-logs/failures-*.txt

# Rerun failed tests
cat test-harness/bdd/.test-logs/rerun-failures-cmd.txt
```

**Save this card!** 📌

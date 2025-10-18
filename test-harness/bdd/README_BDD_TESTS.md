# BDD Test Runner Guide

## Quick Start

```bash
cd test-harness/bdd

# Run all tests
./run-bdd-tests.sh

# Run authentication tests only
./run-bdd-tests.sh --tags @auth

# Run lifecycle tests
./run-bdd-tests.sh --feature lifecycle

# Run with verbose output
./run-bdd-tests.sh --tags @auth --verbose
```

## Features

### ✅ Clean Output
- Color-coded progress indicators
- Clear pass/fail summary
- Organized test results
- No clutter from cargo warnings

### 📊 Progress Logging
- Real-time progress spinner (non-verbose mode)
- Step-by-step execution tracking
- Shows what's happening at each stage

### 📝 Automatic Log Files
- All output saved to `.test-logs/bdd-test-TIMESTAMP.log`
- Results summary saved to `.test-logs/bdd-results-TIMESTAMP.txt`
- Easy to review after test run

### 🎯 Filtered Results
- Shows only what you need
- Failed scenarios highlighted
- Quick access commands provided

## Usage Examples

### Run All Tests
```bash
./run-bdd-tests.sh
```

### Run Specific Tags
```bash
# Authentication tests
./run-bdd-tests.sh --tags @auth

# Priority 0 tests
./run-bdd-tests.sh --tags @p0

# Lifecycle tests
./run-bdd-tests.sh --tags @lifecycle

# Critical tests
./run-bdd-tests.sh --tags @critical
```

### Run Specific Features
```bash
# Worker lifecycle
./run-bdd-tests.sh --feature lifecycle

# Authentication
./run-bdd-tests.sh --feature authentication

# Secrets management
./run-bdd-tests.sh --feature secrets-management
```

### Verbose Mode
```bash
# See all output in real-time
./run-bdd-tests.sh --tags @auth --verbose

# Useful for debugging
./run-bdd-tests.sh --feature lifecycle -v
```

## Output Format

### Normal Mode
```
╔════════════════════════════════════════════════════════════════╗
║           BDD Test Runner - llama-orch Test Harness            ║
╚════════════════════════════════════════════════════════════════╝

📅 Timestamp: 20251018_171900
📂 Project Root: /home/vince/Projects/llama-orch
📝 Log File: .test-logs/bdd-test-20251018_171900.log

🏷️  Tags: @auth

[1/4] Checking compilation...
✅ Compilation successful

[2/4] Discovering test scenarios...
📊 Found 20 scenarios in feature files

[3/4] Running BDD tests...
Command: cargo test --test cucumber -- --tags @auth

⏳ Running tests... Done!

[4/4] Parsing test results...

╔════════════════════════════════════════════════════════════════╗
║                        TEST RESULTS                            ║
╚════════════════════════════════════════════════════════════════╝

✅ ALL TESTS PASSED

📊 Summary:
   ✅ Passed:  20
   ❌ Failed:  0
   ⏭️  Skipped: 0

📁 Output Files:
   Results: .test-logs/bdd-results-20251018_171900.txt
   Full Log: .test-logs/bdd-test-20251018_171900.log

💡 Quick Commands:
   View results: cat .test-logs/bdd-results-20251018_171900.txt
   View full log: less .test-logs/bdd-test-20251018_171900.log
   View errors: grep -E '^error' .test-logs/bdd-test-20251018_171900.log
   View failures: grep -A 10 'FAILED' .test-logs/bdd-test-20251018_171900.log

╔════════════════════════════════════════════════════════════════╗
║                    ✅ SUCCESS ✅                               ║
╚════════════════════════════════════════════════════════════════╝
```

## Log Files

### Location
All logs are stored in `test-harness/bdd/.test-logs/`

### Files Created
- `bdd-test-TIMESTAMP.log` - Full test output including compilation
- `bdd-results-TIMESTAMP.txt` - Summary of test results

### Viewing Logs
```bash
# View latest results
cat .test-logs/bdd-results-*.txt | tail -20

# View latest full log
less .test-logs/bdd-test-*.log | tail -1

# Search for errors
grep -r "error" .test-logs/

# Search for failed tests
grep -r "FAILED" .test-logs/
```

## Troubleshooting

### Compilation Errors
If compilation fails, the script will:
1. Stop immediately
2. Show the first 20 errors
3. Point you to the full log file

```bash
# View all compilation errors
grep "^error" .test-logs/bdd-test-*.log
```

### Test Failures
If tests fail, the script will:
1. Show failed scenario names
2. Display pass/fail counts
3. Provide commands to view details

```bash
# View failed scenarios
grep -A 10 "FAILED" .test-logs/bdd-test-*.log
```

### Hanging Tests
If tests appear to hang:
1. Use `--verbose` mode to see real-time output
2. Check if services are running (queen-rbee, rbee-hive, etc.)
3. Look at the log file in another terminal

```bash
# Run with verbose output
./run-bdd-tests.sh --tags @auth --verbose

# Watch log file in another terminal
tail -f .test-logs/bdd-test-*.log
```

## Available Tags

### Priority Tags
- `@p0` - Priority 0 (critical)
- `@p1` - Priority 1 (important)
- `@p2` - Priority 2 (nice to have)

### Feature Tags
- `@auth` - Authentication tests
- `@lifecycle` - Worker lifecycle tests
- `@security` - Security tests
- `@validation` - Input validation tests
- `@secrets` - Secrets management tests
- `@audit` - Audit logging tests

### Type Tags
- `@critical` - Critical functionality
- `@error-handling` - Error handling tests
- `@edge-case` - Edge case tests
- `@concurrency` - Concurrency tests
- `@performance` - Performance tests

## Integration with CI/CD

### GitHub Actions
```yaml
- name: Run BDD Tests
  run: |
    cd test-harness/bdd
    ./run-bdd-tests.sh --tags @p0
```

### GitLab CI
```yaml
test:bdd:
  script:
    - cd test-harness/bdd
    - ./run-bdd-tests.sh --tags @p0
  artifacts:
    paths:
      - test-harness/bdd/.test-logs/
    when: always
```

## Tips

### Speed Up Test Runs
```bash
# Run only critical tests
./run-bdd-tests.sh --tags @p0

# Run specific feature
./run-bdd-tests.sh --feature authentication
```

### Debug Specific Scenario
```bash
# Use verbose mode
./run-bdd-tests.sh --tags @auth --verbose

# Check logs
less .test-logs/bdd-test-*.log
```

### Clean Old Logs
```bash
# Remove logs older than 7 days
find .test-logs -name "*.log" -mtime +7 -delete
find .test-logs -name "*.txt" -mtime +7 -delete
```

## Help

```bash
./run-bdd-tests.sh --help
```

---

**Created by:** TEAM-102  
**Date:** 2025-10-18  
**Purpose:** Make BDD testing easier with clean output and progress logging

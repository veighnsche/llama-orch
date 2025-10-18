# BDD Test Runner - Quick Start Guide

**Modified by:** TEAM-111  
**Last Updated:** 2025-10-18

## 🚀 Quick Start

### Run All Tests (Live Output)
```bash
./run-bdd-tests.sh
```
**You will see ALL stdout/stderr in real-time!** ✨

### Run Specific Tests
```bash
# By tag
./run-bdd-tests.sh --tags @auth
./run-bdd-tests.sh --tags @p0

# By feature
./run-bdd-tests.sh --feature lifecycle
./run-bdd-tests.sh --feature authentication

# Combined
./run-bdd-tests.sh --tags @p0 --feature lifecycle
```

### Quiet Mode (Summary Only)
```bash
./run-bdd-tests.sh --quiet
./run-bdd-tests.sh --tags @auth --quiet
```

### Help
```bash
./run-bdd-tests.sh --help
```

---

## 📺 What You'll See

### Live Mode (Default)
```
╔════════════════════════════════════════════════════════════════╗
║           BDD Test Runner - llama-orch Test Harness            ║
╚════════════════════════════════════════════════════════════════╝

📅 Timestamp: 20251018_142530
📂 Project Root: /home/vince/Projects/llama-orch
📝 Log Directory: /home/vince/Projects/llama-orch/test-harness/bdd/.test-logs

📺 Output Mode: LIVE (all stdout/stderr shown in real-time)

[1/4] Checking compilation...

   Compiling api-types v0.1.0 (/home/vince/Projects/llama-orch/contracts/api-types)
   Compiling config-schema v0.1.0 (/home/vince/Projects/llama-orch/contracts/config-schema)
   ...
   
✅ Compilation successful

[2/4] Discovering test scenarios...
📊 Found 15 scenarios in feature files

[3/4] Running BDD tests...
Command: cargo test --test cucumber

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    🧪 TEST EXECUTION START 🧪
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📺 LIVE OUTPUT MODE - You will see ALL test output below:

running 15 tests
test lifecycle::worker_registration ... ok
test lifecycle::worker_heartbeat ... ok
test lifecycle::worker_shutdown ... ok
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     🧪 TEST EXECUTION END 🧪
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[4/4] Parsing test results...

╔════════════════════════════════════════════════════════════════╗
║                        TEST RESULTS                            ║
╚════════════════════════════════════════════════════════════════╝

✅ ALL TESTS PASSED

📊 Summary:
   ✅ Passed:  15
   ❌ Failed:  0
   ⏭️  Skipped: 0

📁 Output Files:
   Summary:      .test-logs/bdd-results-20251018_142530.txt
   Test Output:  .test-logs/test-output-20251018_142530.log
   Compile Log:  .test-logs/compile-20251018_142530.log
   Full Log:     .test-logs/bdd-test-20251018_142530.log

╔════════════════════════════════════════════════════════════════╗
║                    ✅ SUCCESS ✅                               ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 📁 Output Files

All logs are saved in `.test-logs/` with timestamps:

| File | What's In It | When to Check |
|------|--------------|---------------|
| `failures-*.txt` | **⭐ ONLY failure details** | **Start here when tests fail!** |
| `bdd-results-*.txt` | Summary (passed/failed/skipped) | Quick status |
| `test-output-*.log` | Raw test output | Full test output review |
| `compile-*.log` | Compilation output | Fixing compile errors |
| `bdd-test-*.log` | Everything (consolidated) | Full trace |

**The `failures-*.txt` file is your best friend when debugging!** It contains:
- All FAILED test markers with context
- All Error messages
- All assertion failures
- All panic messages
- Stack traces
- Organized into clear sections

**NEW! Auto-generated rerun scripts:**
- `rerun-failures.sh` - Executable script to re-run ONLY failed tests
- `rerun-failures-cmd.txt` - Copy-paste command alternative

---

## 🔍 Analyzing Results

### View Failures (⭐ START HERE when tests fail!)
```bash
# The script already extracted all failures for you!
less .test-logs/failures-*.txt
```

### Re-run ONLY Failed Tests (🔄 INSTANT RETRY!)
```bash
# Option 1: Run the auto-generated script
.test-logs/rerun-failures.sh

# Option 2: Copy-paste the command
cat .test-logs/rerun-failures-cmd.txt
# Then copy and run the command shown
```

**This is HUGE for iterative debugging!** No need to:
- ❌ Manually type test names
- ❌ Remember which tests failed
- ❌ Re-run all tests (slow!)

Just run `.test-logs/rerun-failures.sh` and iterate! 🚀

### View Summary
```bash
cat .test-logs/bdd-results-*.txt
```

### View Full Test Output
```bash
less .test-logs/test-output-*.log
```

### View Compilation Errors
```bash
less .test-logs/compile-*.log
```

### Advanced: Custom Extraction (Respecting Engineering Rules)
```bash
# Step 1: Extract to file
grep "pattern" .test-logs/test-output-*.log > custom.out 2>&1

# Step 2: View the file
less custom.out
```

**Note:** Never pipe directly into `less`, `grep`, or `head`! Always write to a file first. This follows `engineering-rules.md` and prevents hangs.

**Pro Tip:** The script already does the hard work of extracting failures, so you usually don't need custom extraction!

---

## 🎯 Common Scenarios

### "I want to see everything as it happens"
```bash
./run-bdd-tests.sh
```
✅ This is the **default behavior**!

### "I just want to know if tests pass/fail"
```bash
./run-bdd-tests.sh --quiet
```

### "Run only critical tests"
```bash
./run-bdd-tests.sh --tags @p0
```

### "Debug a specific feature"
```bash
./run-bdd-tests.sh --feature lifecycle
```

### "CI/CD pipeline"
```bash
./run-bdd-tests.sh --quiet --tags @smoke
```

---

## 🐛 Troubleshooting

### Script says "Cannot find Cargo.toml"
**Solution:** Run from the `test-harness/bdd` directory:
```bash
cd /home/vince/Projects/llama-orch/test-harness/bdd
./run-bdd-tests.sh
```

### No output showing up
**Check:** Are you in quiet mode?
```bash
# Remove --quiet flag
./run-bdd-tests.sh  # Not: ./run-bdd-tests.sh --quiet
```

### Tests hang
**Likely cause:** Pipeline anti-pattern somewhere in test code
**Solution:** Check logs in `.test-logs/` directory

### Want to see previous run
```bash
# List all test runs
ls -lt .test-logs/

# View specific run
less .test-logs/test-output-20251018_142530.log
```

---

## 💡 Pro Tips

1. **Live mode is your friend** - Don't use `--quiet` unless you need to
2. **Logs are timestamped** - You can compare runs
3. **Ctrl+C works** - Stop anytime in live mode
4. **Check exit codes** - `echo $?` after running (0=pass, 1=fail, 2=error)
5. **Follow engineering rules** - Always extract to file before processing

---

## 📚 More Information

- **Full details:** See `.docs/BDD_RUNNER_IMPROVEMENTS.md`
- **Engineering rules:** See `../../.windsurf/rules/engineering-rules.md`
- **BDD features:** See `tests/features/*.feature`

---

## 🎉 That's It!

Just run `./run-bdd-tests.sh` and watch your tests execute in real-time! 🚀

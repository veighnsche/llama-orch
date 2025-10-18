# Auto-Generated Rerun Script Feature

**TEAM-111** - Automatic test rerun script generation  
**Date:** 2025-10-18

---

## 🎯 Problem Solved

When tests fail, you want to:
1. ✅ See what failed
2. ✅ Fix the code
3. ✅ **Re-run ONLY the failed tests** (not all tests!)

Previously, you had to:
- ❌ Manually type test names
- ❌ Remember which tests failed
- ❌ Or re-run all tests (slow!)

**Now it's automatic!** 🎉

---

## 🔄 How It Works

When tests fail, the script automatically:

1. **Extracts failed test names** from the output
   ```
   test lifecycle::worker_shutdown ... FAILED
   test auth::token_validation ... FAILED
   test scheduling::priority_ordering ... FAILED
   ```

2. **Generates an executable script** (`rerun-failures.sh`)
   ```bash
   #!/usr/bin/env bash
   # Auto-generated script to re-run ONLY failed tests
   
   cargo test --test cucumber 'lifecycle::worker_shutdown' -- --nocapture
   cargo test --test cucumber 'auth::token_validation' -- --nocapture
   cargo test --test cucumber 'scheduling::priority_ordering' -- --nocapture
   ```

3. **Creates a copy-paste command file** (`rerun-failures-cmd.txt`)
   ```bash
   cd /home/vince/Projects/llama-orch/test-harness/bdd
   cargo test --test cucumber lifecycle::worker_shutdown auth::token_validation scheduling::priority_ordering -- --nocapture
   ```

---

## 📋 Files Generated

### `rerun-failures.sh`
- **Type:** Executable bash script
- **Purpose:** Run to retry only failed tests
- **Usage:** `.test-logs/rerun-failures.sh`
- **Includes:** 
  - Proper shebang and error handling
  - CD to correct directory
  - One `cargo test` command per failed test
  - `--nocapture` flag for full output

### `rerun-failures-cmd.txt`
- **Type:** Text file with command
- **Purpose:** Copy-paste alternative
- **Usage:** `cat .test-logs/rerun-failures-cmd.txt` then copy
- **Includes:**
  - Comment with timestamp
  - CD command
  - Single `cargo test` with all failed tests

---

## 🚀 Usage Examples

### Scenario 1: Quick Iteration
```bash
# Run all tests
./run-bdd-tests.sh

# Some tests fail...
# Script shows: "🔄 Rerun script generated: .test-logs/rerun-failures.sh"

# Fix the code
vim src/my_module.rs

# Re-run ONLY failed tests
.test-logs/rerun-failures.sh

# Repeat until all pass!
```

### Scenario 2: Copy-Paste Workflow
```bash
# After tests fail, view the command
cat .test-logs/rerun-failures-cmd.txt

# Output:
# Re-run failed tests from 20251018_213500
# Copy and paste the command below:
#
# cd /home/vince/Projects/llama-orch/test-harness/bdd
# cargo test --test cucumber lifecycle::worker_shutdown auth::token_validation -- --nocapture

# Copy the cargo test line and paste in terminal
cargo test --test cucumber lifecycle::worker_shutdown auth::token_validation -- --nocapture
```

### Scenario 3: Debugging Workflow
```bash
# 1. Run tests
./run-bdd-tests.sh

# 2. View failures
less .test-logs/failures-*.txt

# 3. Fix code
vim src/lifecycle.rs

# 4. Rerun failed tests
.test-logs/rerun-failures.sh

# 5. Still failing? Check output and repeat
less .test-logs/failures-*.txt
vim src/lifecycle.rs
.test-logs/rerun-failures.sh
```

---

## 🔧 Technical Details

### Test Name Extraction

**Pattern matched:**
```
test module::name ... FAILED
```

**Extraction process (no pipelines!):**
```bash
# Step 1: Extract lines with FAILED
grep "test .* \.\.\. FAILED" "$TEST_OUTPUT" > failed-tests.tmp

# Step 2: Parse test names
sed 's/test \(.*\) \.\.\. FAILED/\1/' failed-tests.tmp > failed-test-names.tmp

# Step 3: Generate script
while IFS= read -r test_name; do
    echo "cargo test --test cucumber '$test_name' -- --nocapture"
done < failed-test-names.tmp > rerun-failures.sh

# Step 4: Make executable
chmod +x rerun-failures.sh

# Step 5: Cleanup
rm -f failed-tests.tmp failed-test-names.tmp
```

### Script Structure

**Generated script includes:**
- `#!/usr/bin/env bash` - Proper shebang
- `set -euo pipefail` - Strict error handling
- `cd "$SCRIPT_DIR"` - Correct working directory
- Individual test commands with `--nocapture`
- Comments with metadata (timestamp, count)

### Command File Structure

**Generated command file includes:**
- Comment header with timestamp
- CD command to correct directory
- Single `cargo test` with all failed tests as arguments
- `--nocapture` for full output

---

## ✅ Benefits

### Speed
- ⚡ Only run failed tests (not all tests)
- ⚡ Faster iteration cycle
- ⚡ No time wasted on passing tests

### Convenience
- 🎯 No manual test name typing
- 🎯 No remembering which tests failed
- 🎯 Just run the script!

### Accuracy
- ✅ Exact test names extracted
- ✅ No typos
- ✅ No missed tests

### Flexibility
- 🔀 Executable script option
- 🔀 Copy-paste command option
- 🔀 Choose what works for you

---

## 🎓 Best Practices

### When to Use

**Use the rerun script when:**
- ✅ You've fixed the code and want to verify
- ✅ You're iterating on a specific feature
- ✅ You want fast feedback

**Don't use when:**
- ❌ You've changed shared code (run all tests)
- ❌ You want to verify the entire suite
- ❌ You're doing final validation before commit

### Workflow Integration

**Recommended TDD workflow:**
```
1. Write test → Run all tests → Test fails
2. Write code → Run rerun script → Test passes
3. Refactor → Run rerun script → Still passes
4. Final check → Run all tests → Everything passes
5. Commit!
```

**Debugging workflow:**
```
1. Tests fail → View failures file
2. Identify issue → Fix code
3. Run rerun script → Check if fixed
4. Repeat steps 2-3 until all pass
5. Run all tests to verify
```

---

## 🔍 Troubleshooting

### Script not generated?

**Possible causes:**
1. No tests failed (script only generated on failure)
2. Output format not recognized
3. Test names couldn't be extracted

**Check:**
```bash
# Look for the warning message
grep "Could not extract test names" .test-logs/test-output-*.log

# Manually check test output format
less .test-logs/test-output-*.log
```

### Script doesn't work?

**Possible causes:**
1. Working directory changed
2. Test files moved
3. Cargo.toml changed

**Fix:**
```bash
# Check the script contents
cat .test-logs/rerun-failures.sh

# Verify paths are correct
# Edit if needed
vim .test-logs/rerun-failures.sh
```

### Want to modify the command?

**You can edit the generated files!**
```bash
# Edit the script
vim .test-logs/rerun-failures.sh

# Or edit the command file
vim .test-logs/rerun-failures-cmd.txt

# They're just text files - customize as needed!
```

---

## 📊 Example Output

### When Script is Generated

```
💾 Detailed failures saved to: .test-logs/failures-20251018_213500.txt

🔄 Rerun script generated:
   Executable:  .test-logs/rerun-failures.sh
   Command:     .test-logs/rerun-failures-cmd.txt

💡 To re-run ONLY the failed tests:
   .test-logs/rerun-failures.sh
   or
   bash .test-logs/rerun-failures.sh
```

### In File Listings

```
📁 Output Files:
   Summary:      .test-logs/bdd-results-20251018_213500.txt
   Failures:     .test-logs/failures-20251018_213500.txt  ⭐ START HERE
   Rerun Script: .test-logs/rerun-failures.sh  🔄 EXECUTABLE
   Rerun Cmd:    .test-logs/rerun-failures-cmd.txt  📋 COPY-PASTE
   Test Output:  .test-logs/test-output-20251018_213500.log
   ...

💡 Quick Commands:
   View failures:   less .test-logs/failures-20251018_213500.txt  ⭐ DEBUG
   Rerun failed:    .test-logs/rerun-failures.sh  🔄 FIX & RETRY
   ...
```

---

## 🎉 Summary

**This feature gives you:**
1. ✅ Automatic extraction of failed test names
2. ✅ Executable script to re-run only failures
3. ✅ Copy-paste command alternative
4. ✅ Faster iteration cycle
5. ✅ No manual test name typing

**The workflow is now:**
```
Test → Fail → View → Fix → Rerun → Pass → Done!
```

**Instead of:**
```
Test → Fail → Scroll → Find → Type → Rerun → Pass → Done!
```

**Saved steps:** Scroll, Find, Type = **Faster debugging!** 🚀

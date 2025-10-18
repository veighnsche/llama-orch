# Example Output - BDD Test Runner

**TEAM-111** - Visual examples of what you'll see

---

## ✅ Successful Test Run

```
╔════════════════════════════════════════════════════════════════╗
║           BDD Test Runner - llama-orch Test Harness            ║
╚════════════════════════════════════════════════════════════════╝

📅 Timestamp: 20251018_213000
📂 Project Root: /home/vince/Projects/llama-orch
📝 Log Directory: /home/vince/Projects/llama-orch/test-harness/bdd/.test-logs

📺 Output Mode: LIVE (all stdout/stderr shown in real-time)

[1/4] Checking compilation...

   Compiling api-types v0.1.0
   Compiling config-schema v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 3.2s

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
test lifecycle::pool_creation ... ok
test auth::token_validation ... ok
test auth::invalid_token ... ok
test scheduling::basic_request ... ok
test scheduling::queue_overflow ... ok
test scheduling::priority_ordering ... ok
test metrics::counter_increment ... ok
test metrics::gauge_update ... ok
test health::liveness_check ... ok
test health::readiness_check ... ok
test config::valid_yaml ... ok
test config::invalid_yaml ... ok

test result: ok. 15 passed; 0 failed; 0 skipped; finished in 2.34s

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
   Summary:      .test-logs/bdd-results-20251018_213000.txt
   Test Output:  .test-logs/test-output-20251018_213000.log
   Compile Log:  .test-logs/compile-20251018_213000.log
   Full Log:     .test-logs/bdd-test-20251018_213000.log

💡 Quick Commands (respecting engineering-rules.md):
   View summary:    cat .test-logs/bdd-results-20251018_213000.txt
   View test log:   less .test-logs/test-output-20251018_213000.log
   View full log:   less .test-logs/bdd-test-20251018_213000.log

╔════════════════════════════════════════════════════════════════╗
║                    ✅ SUCCESS ✅                               ║
╚════════════════════════════════════════════════════════════════╝
```

---

## ❌ Failed Test Run (THE IMPORTANT ONE!)

```
╔════════════════════════════════════════════════════════════════╗
║           BDD Test Runner - llama-orch Test Harness            ║
╚════════════════════════════════════════════════════════════════╝

📅 Timestamp: 20251018_213500
📂 Project Root: /home/vince/Projects/llama-orch
📝 Log Directory: /home/vince/Projects/llama-orch/test-harness/bdd/.test-logs

📺 Output Mode: LIVE (all stdout/stderr shown in real-time)

[1/4] Checking compilation...

   Compiling api-types v0.1.0
   Compiling config-schema v0.1.0
    Finished dev [unoptimized + debuginfo] target(s) in 3.2s

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
test lifecycle::worker_shutdown ... FAILED
test lifecycle::pool_creation ... ok
test auth::token_validation ... FAILED
test auth::invalid_token ... ok
test scheduling::basic_request ... ok
test scheduling::queue_overflow ... ok
test scheduling::priority_ordering ... FAILED
test metrics::counter_increment ... ok
test metrics::gauge_update ... ok
test health::liveness_check ... ok
test health::readiness_check ... ok
test config::valid_yaml ... ok
test config::invalid_yaml ... ok

failures:

---- lifecycle::worker_shutdown stdout ----
thread 'lifecycle::worker_shutdown' panicked at 'assertion failed: `(left == right)`
  left: `ShutdownState::Pending`,
 right: `ShutdownState::Complete`', tests/lifecycle.rs:45:5
stack backtrace:
   0: rust_begin_unwind
             at /rustc/stable/library/std/src/panicking.rs:584:5
   1: core::panicking::panic_fmt
             at /rustc/stable/library/core/src/panicking.rs:142:14
   2: lifecycle::worker_shutdown
             at ./tests/lifecycle.rs:45:5

---- auth::token_validation stdout ----
Error: Invalid token format
thread 'auth::token_validation' panicked at 'called `Result::unwrap()` on an `Err` value: TokenError("Invalid format")', tests/auth.rs:23:10

---- scheduling::priority_ordering stdout ----
assertion failed: queue.peek().unwrap().priority == Priority::High
  left: Priority::Medium
 right: Priority::High

test result: FAILED. 12 passed; 3 failed; 0 skipped; finished in 2.67s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     🧪 TEST EXECUTION END 🧪
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[4/4] Parsing test results...

╔════════════════════════════════════════════════════════════════╗
║                        TEST RESULTS                            ║
╚════════════════════════════════════════════════════════════════╝

❌ TESTS FAILED

📊 Summary:
   ✅ Passed:  12
   ❌ Failed:  3
   ⏭️  Skipped: 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ❌ FAILURE DETAILS ❌
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

test lifecycle::worker_shutdown ... FAILED
test auth::token_validation ... FAILED
test scheduling::priority_ordering ... FAILED

failures:

---- lifecycle::worker_shutdown stdout ----
thread 'lifecycle::worker_shutdown' panicked at 'assertion failed: `(left == right)`
  left: `ShutdownState::Pending`,
 right: `ShutdownState::Complete`', tests/lifecycle.rs:45:5
stack backtrace:
   0: rust_begin_unwind
             at /rustc/stable/library/std/src/panicking.rs:584:5
   1: core::panicking::panic_fmt
             at /rustc/stable/library/core/src/panicking.rs:142:14
   2: lifecycle::worker_shutdown
             at ./tests/lifecycle.rs:45:5

---- auth::token_validation stdout ----
Error: Invalid token format
thread 'auth::token_validation' panicked at 'called `Result::unwrap()` on an `Err` value: TokenError("Invalid format")', tests/auth.rs:23:10

---- scheduling::priority_ordering stdout ----
assertion failed: queue.peek().unwrap().priority == Priority::High
  left: Priority::Medium
 right: Priority::High

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 Detailed failures saved to: .test-logs/failures-20251018_213500.txt

🔄 Rerun script generated:
   Executable:  .test-logs/rerun-failures.sh
   Command:     .test-logs/rerun-failures-cmd.txt

💡 To re-run ONLY the failed tests:
   .test-logs/rerun-failures.sh
   or
   bash .test-logs/rerun-failures.sh

📁 Output Files:
   Summary:      .test-logs/bdd-results-20251018_213500.txt
   Failures:     .test-logs/failures-20251018_213500.txt  ⭐ START HERE
   Rerun Script: .test-logs/rerun-failures.sh  🔄 EXECUTABLE
   Rerun Cmd:    .test-logs/rerun-failures-cmd.txt  📋 COPY-PASTE
   Test Output:  .test-logs/test-output-20251018_213500.log
   Compile Log:  .test-logs/compile-20251018_213500.log
   Full Log:     .test-logs/bdd-test-20251018_213500.log

💡 Quick Commands (respecting engineering-rules.md):
   View failures:   less .test-logs/failures-20251018_213500.txt  ⭐ DEBUG
   Rerun failed:    .test-logs/rerun-failures.sh  🔄 FIX & RETRY
   View summary:    cat .test-logs/bdd-results-20251018_213500.txt
   View test log:   less .test-logs/test-output-20251018_213500.log
   View full log:   less .test-logs/bdd-test-20251018_213500.log

╔════════════════════════════════════════════════════════════════╗
║                    ❌ FAILED ❌                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 📄 What's in the `failures-TIMESTAMP.txt` File?

```
FAILURE DETAILS - 20251018_213500
========================================

Failed Tests: 3
Command: cargo test --test cucumber

========================================

test lifecycle::worker_shutdown ... FAILED
test auth::token_validation ... FAILED
test scheduling::priority_ordering ... FAILED

failures:

---- lifecycle::worker_shutdown stdout ----
thread 'lifecycle::worker_shutdown' panicked at 'assertion failed: `(left == right)`
  left: `ShutdownState::Pending`,
 right: `ShutdownState::Complete`', tests/lifecycle.rs:45:5
stack backtrace:
   0: rust_begin_unwind
             at /rustc/stable/library/std/src/panicking.rs:584:5
   1: core::panicking::panic_fmt
             at /rustc/stable/library/core/src/panicking.rs:142:14
   2: lifecycle::worker_shutdown
             at ./tests/lifecycle.rs:45:5

---- auth::token_validation stdout ----
Error: Invalid token format
thread 'auth::token_validation' panicked at 'called `Result::unwrap()` on an `Err` value: TokenError("Invalid format")', tests/auth.rs:23:10

---- scheduling::priority_ordering stdout ----
assertion failed: queue.peek().unwrap().priority == Priority::High
  left: Priority::Medium
 right: Priority::High

========================================
Errors:
========================================
Error: Invalid token format

========================================
Panics:
========================================
thread 'lifecycle::worker_shutdown' panicked at 'assertion failed: `(left == right)`
  left: `ShutdownState::Pending`,
 right: `ShutdownState::Complete`', tests/lifecycle.rs:45:5

thread 'auth::token_validation' panicked at 'called `Result::unwrap()` on an `Err` value: TokenError("Invalid format")', tests/auth.rs:23:10
```

---

## 🔄 Rerun Script Contents

### `rerun-failures.sh` (Executable)
```bash
#!/usr/bin/env bash
# Auto-generated script to re-run ONLY failed tests
# Generated: 20251018_213500
# Failed tests: 3

set -euo pipefail

cd "/home/vince/Projects/llama-orch/test-harness/bdd"

# Re-run only the failed tests:
cargo test --test cucumber 'lifecycle::worker_shutdown' -- --nocapture
cargo test --test cucumber 'auth::token_validation' -- --nocapture
cargo test --test cucumber 'scheduling::priority_ordering' -- --nocapture
```

**Usage:**
```bash
# Just run it!
.test-logs/rerun-failures.sh

# Or with bash explicitly
bash .test-logs/rerun-failures.sh
```

### `rerun-failures-cmd.txt` (Copy-Paste)
```bash
# Re-run failed tests from 20251018_213500
# Copy and paste the command below:

cd /home/vince/Projects/llama-orch/test-harness/bdd
cargo test --test cucumber lifecycle::worker_shutdown auth::token_validation scheduling::priority_ordering -- --nocapture
```

**Usage:**
```bash
# View the command
cat .test-logs/rerun-failures-cmd.txt

# Copy the cargo test line and paste it in your terminal
```

---

## 🎯 Key Takeaways

### When Tests Pass ✅
- You see all the live output
- Clean summary at the end
- All logs saved for reference

### When Tests Fail ❌
- You see all the live output
- **Final view shows ONLY failure details** (no scrolling!)
- Dedicated `failures-TIMESTAMP.txt` file created
- **Auto-generated rerun script** for instant retry
- Clear visual separation with red borders
- Files highlighted with ⭐ and 🔄 in output list
- Recommended commands shown prominently

### The Magic ✨
**You don't have to scroll through hundreds of lines of passed tests to find the failures!**

The script automatically:
1. Extracts all FAILED markers
2. Extracts all Error messages
3. Extracts all assertion failures
4. Extracts all panic messages
5. Extracts stack traces
6. Shows them in a clear, focused view
7. Saves them to a dedicated file
8. **Generates executable script to re-run ONLY failed tests** 🔄

### The Workflow 🚀
```
1. Run tests → Some fail
2. See ONLY failure details at the end
3. Read failures-TIMESTAMP.txt for details
4. Fix the code
5. Run .test-logs/rerun-failures.sh
6. Iterate until all pass!
```

**No manual test name typing. No re-running all tests. Just fix and retry!** 🎉

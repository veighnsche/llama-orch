# Anti-Deadlock Rules for Test Harnesses

**Critical:** Drop handlers must NEVER use async/await or block_on()

## The Problem

```rust
// ❌ DEADLOCK - DO NOT DO THIS
impl Drop for TestHarness {
    fn drop(&mut self) {
        let _ = futures::executor::block_on(self.cleanup());  // DEADLOCK!
    }
}
```

**Why it deadlocks:**
- Tests run inside `#[tokio::test]` (runtime already active)
- `block_on()` tries to create new runtime inside existing runtime
- Nested runtimes deadlock
- Test hangs forever

## The Solution

```rust
// ✅ CORRECT - Synchronous cleanup
impl Drop for TestHarness {
    fn drop(&mut self) {
        self.cleanup_sync();  // No async!
    }
}

fn cleanup_sync(&mut self) {
    // Use std::thread::sleep, not tokio::time::sleep
    std::thread::sleep(Duration::from_millis(200));
    
    // Direct syscalls, no async
    Command::new("pkill").args(&["-9", "-f", "process"]).output();
}
```

## Rules for Test Harnesses

### ✅ DO
- Use `std::thread::sleep` in Drop handlers
- Make cleanup synchronous
- Keep async `cleanup()` separate for explicit calls
- Use direct syscalls (Command::new, etc.)

### ❌ DON'T
- Use `futures::executor::block_on()` in Drop
- Use `tokio::runtime::Runtime::new()` in Drop
- Call `async fn` from Drop
- Use `.await` in Drop

## Pattern

```rust
pub struct TestHarness {
    // fields
}

impl TestHarness {
    /// Async cleanup for explicit calls
    pub async fn cleanup(&mut self) -> Result<()> {
        // Can use tokio::time::sleep
        // Can be async
        Ok(())
    }
    
    /// Sync cleanup for Drop
    fn cleanup_sync(&mut self) {
        // MUST be synchronous
        // Use std::thread::sleep
        // Direct syscalls only
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        self.cleanup_sync();  // Never block_on!
    }
}
```

## Historical Context

**TEAM-252** discovered this issue when integration tests hung indefinitely:
- 62 tests would hang forever instead of exiting
- Required manual Ctrl+C to kill
- CI/CD pipelines blocked
- Fix: Synchronous Drop handler, 100x+ speedup

**Reference:** `/home/vince/Projects/llama-orch/TEAM-252-FINAL-HANG-FIX-SUMMARY.md`

## Detection

If tests hang with these symptoms:
```
test xyz has been running for over 60 seconds
test xyz has been running for over 60 seconds
^C  # Manual kill required
```

**Likely cause:** Drop handler using `block_on()` or async

**Fix:** Make Drop handler synchronous

## Verification

```bash
# Tests should exit in < 5 seconds even on failure
timeout 10 cargo test --package your-crate --lib

# Should NOT timeout (exit code 124 = timeout)
echo $?  # Should be 0 or 101, not 124
```

---

**Rule:** Drop = Synchronous. Always. No exceptions.

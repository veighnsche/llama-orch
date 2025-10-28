# TEAM-335: Incremental Test Plan

**Purpose:** Systematically find the EXACT breaking point

---

## Test 1: Minimal Function

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_install_test1() -> Result<String, String> {
    Ok("Test 1: Minimal function".to_string())
}
```

**Expected:** ✅ Should work  
**If fails:** Tauri itself is broken (unlikely)

---

## Test 2: Add Sleep

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_install_test2() -> Result<String, String> {
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    Ok("Test 2: With async sleep".to_string())
}
```

**Expected:** ✅ Should work  
**If fails:** Basic async in Tauri is broken

---

## Test 3: Load Config

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_install_test3() -> Result<String, String> {
    use crate::Config;
    let config = Config::load()
        .map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();
    Ok(format!("Test 3: Config loaded, URL: {}", queen_url))
}
```

**Expected:** ✅ Should work  
**If fails:** Config loading has deep stack

---

## Test 4: Call handle_queen (Minimal Action)

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_install_test4() -> Result<String, String> {
    use crate::handlers::handle_queen;
    use crate::cli::QueenAction;
    use crate::Config;
    
    let config = Config::load()
        .map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();
    
    // Use Status (simplest operation, just HTTP check)
    handle_queen(QueenAction::Status, &queen_url)
        .await
        .map_err(|e| format!("{}", e))?;
    
    Ok("Test 4: handle_queen Status worked".to_string())
}
```

**Expected:** ✅ Should work  
**If fails:** handle_queen itself has issues

---

## Test 5: Call Install with Pre-built Binary

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_install_test5(binary_path: String) -> Result<String, String> {
    use crate::handlers::handle_queen;
    use crate::cli::QueenAction;
    use crate::Config;
    
    let config = Config::load()
        .map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();
    
    // Install with explicit binary (SKIPS cargo build!)
    handle_queen(
        QueenAction::Install { binary: Some(binary_path) }, 
        &queen_url
    )
    .await
    .map_err(|e| format!("{}", e))?;
    
    Ok("Test 5: Install with pre-built binary worked".to_string())
}
```

**Expected:** 
- ✅ If works: Problem is in cargo build
- ❌ If fails: Problem is in install SSH operations

---

## Test 6: Full Install (No Pre-built)

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_install_test6() -> Result<String, String> {
    use crate::handlers::handle_queen;
    use crate::cli::QueenAction;
    use crate::Config;
    
    let config = Config::load()
        .map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();
    
    // Full install, builds from source
    handle_queen(QueenAction::Install { binary: None }, &queen_url)
        .await
        .map_err(|e| format!("{}", e))?;
    
    Ok("Test 6: Full install worked".to_string())
}
```

**Expected:** ❌ Should fail with stack overflow

---

## Test 7: Explicit Large Stack Thread

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_install_test7() -> Result<String, String> {
    // Spawn thread with EXPLICIT large stack
    let handle = std::thread::Builder::new()
        .name("queen-install".to_string())
        .stack_size(32 * 1024 * 1024)  // 32MB stack
        .spawn(move || {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("Runtime error: {}", e))?;
            
            rt.block_on(async move {
                use crate::handlers::handle_queen;
                use crate::cli::QueenAction;
                use crate::Config;
                
                let config = Config::load()
                    .map_err(|e| format!("Config error: {}", e))?;
                let queen_url = config.queen_url();
                
                handle_queen(QueenAction::Install { binary: None }, &queen_url)
                    .await
                    .map_err(|e| format!("{}", e))
            })
        })
        .map_err(|e| format!("Thread spawn error: {}", e))?;
    
    handle.join()
        .map_err(|e| format!("Thread join error: {:?}", e))?
}
```

**Expected:**
- ✅ If works: Stack size WAS the issue, just needed more than 8MB
- ❌ If fails: Not a stack size issue

---

## Test 8: Use Handle::current() Instead of New Runtime

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_install_test8() -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        // DON'T create new runtime - use current
        tokio::runtime::Handle::try_current()
            .map_err(|e| format!("No current runtime: {}", e))?
            .block_on(async move {
                use crate::handlers::handle_queen;
                use crate::cli::QueenAction;
                use crate::Config;
                
                let config = Config::load()
                    .map_err(|e| format!("Config error: {}", e))?;
                let queen_url = config.queen_url();
                
                handle_queen(QueenAction::Install { binary: None }, &queen_url)
                    .await
                    .map_err(|e| format!("{}", e))
            })
    })
    .await
    .map_err(|e| format!("Join error: {}", e))?
}
```

**Expected:**
- ✅ If works: Creating new runtime was the issue
- ❌ If fails: Something else

---

## How to Run Tests

1. **Add test command to tauri_commands.rs:**
   ```rust
   #[tauri::command]
   #[specta::specta]
   pub async fn queen_install_test_X() -> Result<String, String> {
       // Test code here
   }
   ```

2. **Register in main.rs:**
   ```rust
   tauri::Builder::default()
       .invoke_handler(tauri::generate_handler![
           ssh_list,
           queen_install_test_1,
           queen_install_test_2,
           // ... etc
       ])
   ```

3. **Update TypeScript bindings:**
   ```bash
   cargo test --lib export_typescript_bindings
   ```

4. **Call from UI console:**
   ```javascript
   await window.__TAURI__.invoke('queen_install_test_1')
   ```

5. **Document result** in this file

---

## Test Results Template

```
Test 1: Minimal Function
- Status: [ ] Pass [ ] Fail
- Notes: 

Test 2: Add Sleep
- Status: [ ] Pass [ ] Fail
- Notes:

Test 3: Load Config
- Status: [ ] Pass [ ] Fail
- Notes:

Test 4: Call handle_queen (Status)
- Status: [ ] Pass [ ] Fail
- Notes:

Test 5: Install with Pre-built Binary
- Status: [ ] Pass [ ] Fail
- Notes:
- Binary path used:

Test 6: Full Install (No Pre-built)
- Status: [ ] Pass [ ] Fail
- Notes:

Test 7: Explicit Large Stack Thread
- Status: [ ] Pass [ ] Fail
- Notes:

Test 8: Use Handle::current()
- Status: [ ] Pass [ ] Fail
- Notes:
```

---

## Expected Outcome

By the end of these tests, you will know:

1. Which operations work
2. Which operation FIRST causes the crash
3. Whether it's stack size or something else
4. Whether new runtime creation is the issue

**This eliminates guessing and gives us DATA.**

---

**END OF TEST PLAN**

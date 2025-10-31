# Phase 1: Add Build Metadata (REVISED - Using shadow-rs)

**Goal:** Use `shadow-rs` crate to embed build metadata in all daemon binaries.

**Status:** Ready to implement  
**Estimated Time:** 20 minutes (faster with shadow-rs!)

---

## Why shadow-rs?

Instead of writing custom `build.rs` files for each binary, we use the **`shadow-rs`** crate:

✅ **Centralized** - One dependency, consistent across all binaries  
✅ **Maintained** - 112 releases, actively developed  
✅ **Rich metadata** - Git info, build time, Rust version, etc.  
✅ **Simple API** - Just `shadow!(build)` macro  
✅ **Provides `BUILD_RUST_CHANNEL`** - "debug" or "release"  

**Crate:** https://github.com/baoyachi/shadow-rs  
**Docs:** https://docs.rs/shadow-rs

---

## Binaries That Need Metadata

All daemon binaries:
1. **queen-rbee** - Orchestrator daemon
2. **rbee-hive** - Worker manager daemon
3. **llm-worker-rbee** - LLM inference worker

---

## Implementation Checklist

### **Step 1: Add shadow-rs to queen-rbee**

**File:** `bin/10_queen_rbee/Cargo.toml` (MODIFY)

```toml
[package]
name = "queen-rbee"
version = "0.1.0"
edition = "2021"
build = "build.rs"  # ← Add this line

[dependencies]
# ... existing dependencies ...
shadow-rs = { version = "0.35", default-features = false }

[build-dependencies]
shadow-rs = "0.35"
```

**File:** `bin/10_queen_rbee/build.rs` (NEW)

```rust
fn main() {
    shadow_rs::ShadowBuilder::builder()
        .build()
        .unwrap();
}
```

**File:** `bin/10_queen_rbee/src/main.rs` (MODIFY)

**Add at top of file (after other use statements):**
```rust
use shadow_rs::shadow;

shadow!(build);
```

**Update Args struct:**
```rust
#[derive(Parser, Debug)]
#[command(name = "queen-rbee")]
#[command(about = "rbee Orchestrator Daemon - Job scheduling and hive management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "7833")]
    port: u16,

    /// Print build information and exit
    #[arg(long, hide = true)]
    build_info: bool,
}
```

**Update main() function:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        // shadow-rs provides BUILD_RUST_CHANNEL: "debug" or "release"
        println!("{}", build::BUILD_RUST_CHANNEL);
        std::process::exit(0);
    }

    // ... rest of main logic ...
}
```

---

### **Step 2: Add shadow-rs to rbee-hive**

**File:** `bin/20_rbee_hive/Cargo.toml` (MODIFY)

```toml
[package]
name = "rbee-hive"
version = "0.1.0"
edition = "2021"
build = "build.rs"  # ← Add this line

[dependencies]
# ... existing dependencies ...
shadow-rs = { version = "0.35", default-features = false }

[build-dependencies]
shadow-rs = "0.35"
```

**File:** `bin/20_rbee_hive/build.rs` (NEW)

```rust
fn main() {
    shadow_rs::ShadowBuilder::builder()
        .build()
        .unwrap();
}
```

**File:** `bin/20_rbee_hive/src/main.rs` (MODIFY)

**Add at top:**
```rust
use shadow_rs::shadow;

shadow!(build);
```

**Update Args struct:**
```rust
/// Print build information and exit
#[arg(long, hide = true)]
build_info: bool,
```

**Update main():**
```rust
if args.build_info {
    println!("{}", build::BUILD_RUST_CHANNEL);
    std::process::exit(0);
}
```

---

### **Step 3: Add shadow-rs to llm-worker-rbee**

**File:** `bin/30_llm_worker_rbee/Cargo.toml` (MODIFY)

```toml
[package]
name = "llm-worker-rbee"
version = "0.1.0"
edition = "2021"
build = "build.rs"  # ← Add this line

[dependencies]
# ... existing dependencies ...
shadow-rs = { version = "0.35", default-features = false }

[build-dependencies]
shadow-rs = "0.35"
```

**File:** `bin/30_llm_worker_rbee/build.rs` (NEW)

```rust
fn main() {
    shadow_rs::ShadowBuilder::builder()
        .build()
        .unwrap();
}
```

**File:** `bin/30_llm_worker_rbee/src/main.rs` (MODIFY)

**Add at top:**
```rust
use shadow_rs::shadow;

shadow!(build);
```

**Update Args struct:**
```rust
/// Print build information and exit
#[arg(long, hide = true)]
build_info: bool,
```

**Update main():**
```rust
if args.build_info {
    println!("{}", build::BUILD_RUST_CHANNEL);
    std::process::exit(0);
}
```

---

## Testing Phase 1

### **Test 1: Build and check metadata**

```bash
# Build debug version
cargo build --bin queen-rbee

# Check metadata
./target/debug/queen-rbee --build-info
# Expected output: debug

# Build release version
cargo build --release --bin queen-rbee

# Check metadata
./target/release/queen-rbee --build-info
# Expected output: release
```

### **Test 2: Verify all three binaries**

```bash
# Test queen-rbee
./target/debug/queen-rbee --build-info    # → debug
./target/release/queen-rbee --build-info  # → release

# Test rbee-hive
./target/debug/rbee-hive --build-info     # → debug
./target/release/rbee-hive --build-info   # → release

# Test llm-worker-rbee
./target/debug/llm-worker-rbee --build-info    # → debug
./target/release/llm-worker-rbee --build-info  # → release
```

### **Test 3: Verify normal operation**

```bash
# Start queen-rbee normally
./target/debug/queen-rbee --port 7833
# Should start normally, not exit
```

### **Test 4: Check shadow-rs generated file**

```bash
# shadow-rs generates a file in target/
cat target/debug/build/queen-rbee-*/out/shadow.rs
# Should see all the build constants
```

---

## Success Criteria

- ✅ `shadow-rs` added to all three binaries
- ✅ `build.rs` files created (identical for all)
- ✅ `--build-info` flag works
- ✅ Debug builds output "debug"
- ✅ Release builds output "release"
- ✅ Normal operation unaffected
- ✅ Compilation successful

---

## Files Modified

### **NEW Files**
- `bin/10_queen_rbee/build.rs`
- `bin/20_rbee_hive/build.rs`
- `bin/30_llm_worker_rbee/build.rs`

### **MODIFIED Files**
- `bin/10_queen_rbee/Cargo.toml`
- `bin/10_queen_rbee/src/main.rs`
- `bin/20_rbee_hive/Cargo.toml`
- `bin/20_rbee_hive/src/main.rs`
- `bin/30_llm_worker_rbee/Cargo.toml`
- `bin/30_llm_worker_rbee/src/main.rs`

---

## Benefits of shadow-rs

### **vs Custom build.rs**
- ✅ **No chrono dependency** - shadow-rs handles timestamps
- ✅ **No manual env!() calls** - Auto-generated constants
- ✅ **Consistent across binaries** - Same API everywhere
- ✅ **Rich metadata** - Git info, build time, etc. (for future use)

### **Future Capabilities**
Once shadow-rs is integrated, we can easily add:
- Git commit hash (for debugging)
- Build timestamp (for version tracking)
- Rust version (for compatibility checks)
- Branch name (for CI/CD)

All without changing any code!

---

## Next Phase

After Phase 1 is complete and tested, proceed to:
**`PHASE_2_MODE_DETECTION.md`** - Read metadata from binaries

---

**Ready to implement with shadow-rs!** 🚀

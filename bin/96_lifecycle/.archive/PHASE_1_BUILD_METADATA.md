# Phase 1: Add Build Metadata

**Goal:** Embed `BUILD_PROFILE` and `BUILD_TIMESTAMP` in queen-rbee and rbee-hive binaries.

**Status:** Ready to implement  
**Estimated Time:** 30 minutes

---

## What We're Building

Each daemon binary will contain compile-time metadata:
- `BUILD_PROFILE` - "debug" or "release"
- `BUILD_TIMESTAMP` - ISO 8601 timestamp
- Accessible via `--build-info` CLI flag

---

## Binaries That Need Metadata

All daemon binaries that can be installed/started:
1. **queen-rbee** - Orchestrator daemon
2. **rbee-hive** - Worker manager daemon
3. **llm-worker-rbee** - LLM inference worker

---

## Implementation Checklist

### **Step 1: Create build.rs for queen-rbee**

**File:** `bin/10_queen_rbee/build.rs` (NEW)

```rust
// build.rs - Embed build metadata at compile time

fn main() {
    // Embed build profile (debug or release)
    println!(
        "cargo:rustc-env=BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap()
    );

    // Embed build timestamp
    println!(
        "cargo:rustc-env=BUILD_TIMESTAMP={}",
        chrono::Utc::now().to_rfc3339()
    );

    // Re-run if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
```

**Dependencies needed in Cargo.toml:**
```toml
[build-dependencies]
chrono = "0.4"
```

---

### **Step 2: Add --build-info flag to queen-rbee**

**File:** `bin/10_queen_rbee/src/main.rs` (MODIFY)

**Add at top of file (after imports):**
```rust
// Embedded build metadata (set by build.rs)
pub const BUILD_PROFILE: &str = env!("BUILD_PROFILE");
pub const BUILD_TIMESTAMP: &str = env!("BUILD_TIMESTAMP");
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

**Update main() function (before server starts):**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        println!("{}", BUILD_PROFILE);
        std::process::exit(0);
    }

    // ... rest of main logic ...
}
```

---

### **Step 3: Create build.rs for rbee-hive**

**File:** `bin/20_rbee_hive/build.rs` (NEW)

```rust
// build.rs - Embed build metadata at compile time

fn main() {
    // Embed build profile (debug or release)
    println!(
        "cargo:rustc-env=BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap()
    );

    // Embed build timestamp
    println!(
        "cargo:rustc-env=BUILD_TIMESTAMP={}",
        chrono::Utc::now().to_rfc3339()
    );

    // Re-run if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
```

**Dependencies needed in Cargo.toml:**
```toml
[build-dependencies]
chrono = "0.4"
```

---

### **Step 4: Add --build-info flag to rbee-hive**

**File:** `bin/20_rbee_hive/src/main.rs` (MODIFY)

**Add at top of file (after imports):**
```rust
// Embedded build metadata (set by build.rs)
pub const BUILD_PROFILE: &str = env!("BUILD_PROFILE");
pub const BUILD_TIMESTAMP: &str = env!("BUILD_TIMESTAMP");
```

**Update Args struct:**
```rust
#[derive(Parser, Debug)]
#[command(name = "rbee-hive")]
#[command(about = "rbee Hive Daemon - Worker and model management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "7835")]
    port: u16,

    /// Queen URL for heartbeat reporting
    #[arg(long, default_value = "http://localhost:7833")]
    queen_url: String,

    /// Print build information and exit
    #[arg(long, hide = true)]
    build_info: bool,
}
```

**Update main() function (before server starts):**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        println!("{}", BUILD_PROFILE);
        std::process::exit(0);
    }

    // ... rest of main logic ...
}
```

---

### **Step 5: Create build.rs for llm-worker-rbee**

**File:** `bin/30_llm_worker_rbee/build.rs` (NEW)

```rust
// build.rs - Embed build metadata at compile time

fn main() {
    // Embed build profile (debug or release)
    println!(
        "cargo:rustc-env=BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap()
    );

    // Embed build timestamp
    println!(
        "cargo:rustc-env=BUILD_TIMESTAMP={}",
        chrono::Utc::now().to_rfc3339()
    );

    // Re-run if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
```

**Dependencies needed in Cargo.toml:**
```toml
[build-dependencies]
chrono = "0.4"
```

---

### **Step 6: Add --build-info flag to llm-worker-rbee**

**File:** `bin/30_llm_worker_rbee/src/main.rs` (MODIFY)

**Add at top of file (after imports):**
```rust
// Embedded build metadata (set by build.rs)
pub const BUILD_PROFILE: &str = env!("BUILD_PROFILE");
pub const BUILD_TIMESTAMP: &str = env!("BUILD_TIMESTAMP");
```

**Update Args struct (find and modify):**
Add the `build_info` field:
```rust
/// Print build information and exit
#[arg(long, hide = true)]
build_info: bool,
```

**Update main() function (before server starts):**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        println!("{}", BUILD_PROFILE);
        std::process::exit(0);
    }

    // ... rest of main logic ...
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

### **Test 3: Verify normal operation still works**

```bash
# Start queen-rbee normally (should ignore --build-info)
./target/debug/queen-rbee --port 7833
# Should start normally, not exit
```

---

## Success Criteria

- ✅ `build.rs` files created for both binaries
- ✅ `BUILD_PROFILE` and `BUILD_TIMESTAMP` constants defined
- ✅ `--build-info` flag added to both CLIs
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
- `bin/10_queen_rbee/src/main.rs`
- `bin/10_queen_rbee/Cargo.toml` (add chrono to build-dependencies)
- `bin/20_rbee_hive/src/main.rs`
- `bin/20_rbee_hive/Cargo.toml` (add chrono to build-dependencies)
- `bin/30_llm_worker_rbee/src/main.rs`
- `bin/30_llm_worker_rbee/Cargo.toml` (add chrono to build-dependencies)

---

## Next Phase

After Phase 1 is complete and tested, proceed to:
**`PHASE_2_MODE_DETECTION.md`** - Read metadata from binaries

---

**Ready to implement!** 🚀

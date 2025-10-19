# RUST LIBRARY SUGGESTIONS FOR TEAM-130B

**📖 NOTE: This document is a REFERENCE for external Rust library recommendations.**

**Use these suggestions when writing Part 2 (Libraries) sections of final investigations.**

**👉 For when to write Part 2 sections, see: `TEAM_130B_PHASED_APPROACH.md`**

---

## Testing Libraries

### rstest = "0.18"
**Purpose:** Parameterized and fixture-based testing  
**Why:** Reduce test boilerplate by 30-50%  
**Example:**
```rust
#[rstest]
#[case("valid-id-1", true)]
#[case("valid-id-2", true)]
#[case("invalid!", false)]
fn test_id_validation(#[case] input: &str, #[case] expected: bool) {
    assert_eq!(is_valid_id(input), expected);
}
```

### proptest = "1.4"
**Purpose:** Property-based testing  
**Why:** Find edge cases automatically  
**Example:**
```rust
proptest! {
    #[test]
    fn parse_always_succeeds_or_errors(s in "\\PC*") {
        let result = parse(& s);
        assert!(result.is_ok() || result.is_err());
    }
}
```

### wiremock = "0.5"
**Purpose:** HTTP mocking for testing  
**Why:** Test HTTP clients without real servers  
**Use in:** queen-rbee, llm-worker-rbee

### tempfile = "3.8"
**Purpose:** Temporary files/directories  
**Why:** Safe temp file handling in tests  
**Already used in:** Most projects

---

## CLI Libraries (rbee-keeper)

### clap = "4.4" with derive
**Purpose:** CLI argument parsing  
**Why:** Modern, type-safe, auto-generated help  
**Example:**
```rust
#[derive(Parser)]
struct Cli {
    /// The configuration file
    #[arg(short, long)]
    config: PathBuf,
    
    #[command(subcommand)]
    command: Commands,
}
```

### indicatif = "0.17"
**Purpose:** Progress bars and spinners  
**Why:** Professional CLI UX  
**Example:**
```rust
let pb = ProgressBar::new(100);
for i in 0..100 {
    pb.set_position(i);
    // work...
}
pb.finish_with_message("Done!");
```

### console = "0.15"
**Purpose:** Colored terminal output  
**Why:** Better readability  
**Example:**
```rust
println!("{}", style("Success!").green().bold());
println!("{}", style("Error:").red());
```

### dialoguer = "0.11"
**Purpose:** Interactive prompts  
**Why:** User-friendly setup wizards  
**Example:**
```rust
let selection = Select::new()
    .with_prompt("Choose environment")
    .items(&["development", "production"])
    .interact()?;
```

---

## SSH Libraries (queen-rbee, rbee-keeper)

### russh = "0.40"
**Purpose:** Pure Rust SSH client  
**Why:** Replace vulnerable tokio::process::Command SSH  
**Priority:** HIGH (security fix)  
**Example:**
```rust
let session = russh::client::connect(config, host, handler).await?;
let channel = session.channel_open_session().await?;
channel.exec(true, command).await?;
```

### Alternative: async-ssh2-tokio = "0.8"
**Purpose:** Async wrapper for libssh2  
**Why:** If russh has issues  
**Note:** Requires system libssh2

---

## Observability Libraries

### tracing-opentelemetry = "0.21"
**Purpose:** OpenTelemetry integration  
**Why:** Distributed tracing across binaries  
**Use in:** All binaries

### metrics = "0.21"
**Purpose:** Metrics collection  
**Why:** Prometheus-compatible metrics  
**Example:**
```rust
metrics::counter!("requests_total").increment(1);
metrics::gauge!("active_workers").set(worker_count as f64);
```

### console-subscriber = "0.2"
**Purpose:** Tokio console for debugging  
**Why:** Debug async task performance  
**Use in:** Development only

---

## Configuration Libraries

### figment = "0.10"
**Purpose:** Layered configuration  
**Why:** Env vars + files + CLI in priority order  
**Example:**
```rust
let config: Config = Figment::new()
    .merge(Toml::file("config.toml"))
    .merge(Env::prefixed("APP_"))
    .extract()?;
```

### Alternative: config = "0.13"
**Purpose:** Similar to figment  
**Why:** More established, larger ecosystem

---

## Secrets Management

### secrecy = "0.8"
**Purpose:** Wrap secrets to prevent leaking  
**Why:** Secrets don't log or display  
**Example:**
```rust
use secrecy::{Secret, ExposeSecret};

let api_key: Secret<String> = Secret::new(key);
// api_key.to_string() panics!
// Must explicitly: api_key.expose_secret()
```

### keyring = "2.0"
**Purpose:** OS keychain integration  
**Why:** Store secrets in system keychain  
**Example:**
```rust
let entry = Entry::new("app-name", "username")?;
entry.set_password("secret")?;
let password = entry.get_password()?;
```

---

## Serialization/Performance

### simd-json = "0.13"
**Purpose:** SIMD-accelerated JSON  
**Why:** 2-3x faster JSON parsing  
**Use in:** Hot paths (inference responses)  
**Note:** Requires mutable input buffer

---

## GPU/System Info

### sysinfo = "0.30"
**Purpose:** System information  
**Why:** Cross-platform CPU, memory, disk info  
**Example:**
```rust
let mut sys = System::new_all();
sys.refresh_all();
println!("Total memory: {} KB", sys.total_memory());
```

### nvml-wrapper = "0.9"
**Purpose:** NVIDIA GPU information  
**Why:** Direct NVML binding for GPU stats  
**Use in:** rbee-hive, queen-rbee (if NVIDIA-specific)

---

## HTTP Client (if not using reqwest)

### ureq = "2.9"
**Purpose:** Lightweight sync HTTP  
**Why:** Simpler for simple cases  
**Note:** Most code uses reqwest async, stick with it

---

## Error Handling (Current State)

### anyhow = "1.0"
**Status:** ✅ Keep using in binaries  
**Purpose:** Easy error handling  
**Why:** Ergonomic for applications

### thiserror = "1.0"
**Status:** ✅ Keep using in libraries  
**Purpose:** Custom error types  
**Why:** Better error APIs for library consumers

**Note:** NO CHANGES NEEDED, current approach is correct!

---

## Async Runtime (Current State)

### tokio = "1.35"
**Status:** ✅ Keep using  
**Check:** Ensure correct features enabled  
**Common features:**
- `full` - All features (binary use)
- `rt-multi-thread` - Multi-threaded runtime
- `macros` - #[tokio::main], #[tokio::test]
- `sync` - Channels, mutexes
- `fs` - Async file I/O
- `net` - Async networking

**Recommendation:** Use `full` in binaries, specific features in libraries

---

## Web Frameworks (Current State)

### axum = "0.7" (if used)
**Status:** ✅ Modern choice, keep using  
**Alternative:** actix-web (more mature)  
**Note:** Check which one is actually used

---

## Dependency Recommendations by Binary

### rbee-hive
- ✅ Keep: tokio, serde, axum/actix-web
- ➕ Add: metrics (Prometheus), sysinfo, nvml-wrapper
- ⚠️ Check: tracing setup

### queen-rbee
- ✅ Keep: tokio, serde, reqwest
- ➕ Add: russh (CRITICAL - replace Command SSH)
- ➕ Add: wiremock (testing HTTP calls)
- ⚠️ Fix: Use auth-min properly

### llm-worker-rbee
- ✅ Keep: tokio, serde
- ➕ Add: simd-json (fast inference response parsing)
- ➕ Add: metrics (inference metrics)
- ⚠️ Prepare: auth-min for future

### rbee-keeper
- ✅ Keep: tokio, serde
- ➕ Add: clap v4 (modern CLI)
- ➕ Add: indicatif (progress bars)
- ➕ Add: console (colored output)
- ➕ Add: dialoguer (interactive prompts)
- ➕ Add: russh (if doing SSH)
- ⚠️ Expand: input-validation usage

---

## Testing Recommendations (All Binaries)

**Add to [dev-dependencies]:**
```toml
[dev-dependencies]
rstest = "0.18"
tempfile = "3.8"
proptest = "1.4"
wiremock = "0.5"  # If HTTP client used
```

---

## Priority Matrix

| Library | Priority | Reason |
|---------|----------|--------|
| russh | ❌ CRITICAL | Security vulnerability fix |
| clap v4 | 🟡 HIGH | rbee-keeper UX |
| indicatif | 🟡 HIGH | rbee-keeper UX |
| rstest | 🟡 HIGH | Reduce test boilerplate |
| metrics | 🟡 HIGH | Production observability |
| simd-json | 🟢 MEDIUM | Performance optimization |
| figment | 🟢 MEDIUM | Better config management |
| proptest | 🟢 MEDIUM | Find edge cases |
| secrecy | 🟢 MEDIUM | Secret safety |
| wiremock | 🟢 MEDIUM | Better testing |
| console | 🔵 LOW | Nice to have |
| dialoguer | 🔵 LOW | Nice to have |

---

## How to Research More

```bash
# Search crates.io
open https://crates.io/search?q=[topic]

# Check reverse dependencies (popularity)
open https://crates.io/crates/[crate-name]/reverse_dependencies

# Check maintenance
cargo info [crate-name]

# Check security advisories
cargo audit --json | jq '.vulnerabilities'

# Check dependency tree
cargo tree -p [binary] | grep [crate-name]
```

---

**Use these suggestions in your final investigation files!**

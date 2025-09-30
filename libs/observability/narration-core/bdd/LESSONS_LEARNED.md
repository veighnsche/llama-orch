# Lessons Learned: How to Wire BDD Properly

**Date**: 2025-09-30 23:17  
**Source**: Analysis of BDD_WIRING.md and existing BDD implementations

## Key Discovery: The World Struct Pattern

### ✅ Correct Pattern (from orchestratord and proof-bundle)

```rust
// orchestratord/bdd/src/steps/world.rs
#[derive(cucumber::World)]
pub struct World {
    facts: Vec<serde_json::Value>,
    pub mode_commit: bool,
    pub state: AppState,
    // ... other fields
}

// Manual Debug implementation when fields don't derive Debug
impl fmt::Debug for World {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("World")
            .field("facts_len", &self.facts.len())
            .field("state", &"<AppState redacted>")
            .finish()
    }
}
```

### ✅ Alternative: All Fields Derive Debug (from proof-bundle)

```rust
// proof-bundle/bdd/src/steps/world.rs
#[derive(Debug, Default, cucumber::World)]
pub struct BddWorld {
    pub temp_root: Option<tempfile::TempDir>,
    pub bundle_root: Option<std::path::PathBuf>,
    pub selected_type: Option<proof_bundle::TestType>,
}
```

## Critical Rules for Regex in Step Definitions

### ✅ CORRECT: Plain regex strings (no escaping needed)

From orchestratord/bdd examples:

```rust
#[when(regex = "^I create a model with id (.+) and digest (.+)$")]
pub async fn when_create_model_with_digest(world: &mut World, id: String, digest: String) {
    // ...
}

#[when(regex = r"^I enqueue a completion task with valid payload$")]
pub async fn when_enqueue_valid_completion(world: &mut World) {
    // ...
}

#[when(regex = "^I get model (.+)$")]
pub async fn when_get_model(world: &mut World, id: String) {
    // ...
}
```

**Key Insight**: 
- Use plain `"..."` strings - NO escaping needed!
- Use `r"..."` for raw strings when you want to avoid any escaping
- **NEVER use `r#"..."#` with quotes inside** - that was my mistake!
- **NEVER escape quotes with `\"`** - that was the original problem!

### ❌ WRONG: What I Did (escaped quotes or raw string delimiters)

```rust
// WRONG - escaped quotes
#[when(regex = r"^I narrate with actor \"([^\"]+)\"$")]

// WRONG - r#"..."# with quotes
#[when(regex = r#"^I narrate with actor "([^"]+)"$"#)]
```

## The Main Runner Pattern

### Standard Pattern (from BDD_WIRING.md)

```rust
mod steps;

use cucumber::World as _;
use steps::world::World;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features_env = std::env::var("LLORCH_BDD_FEATURE_PATH").ok();
    let features = if let Some(p) = features_env {
        let pb = std::path::PathBuf::from(p);
        if pb.is_absolute() {
            pb
        } else {
            root.join(pb)
        }
    } else {
        root.join("tests/features")
    };

    World::cucumber().fail_on_skipped().run_and_exit(features).await;
}
```

## The CaptureAdapter Problem & Solution

### Problem
`CaptureAdapter` doesn't implement `Debug`, so we can't use `#[derive(Debug)]` on World.

### Solution Options

**Option 1**: Manual Debug Implementation (orchestratord pattern)
```rust
#[derive(cucumber::World)]
pub struct World {
    pub adapter: Option<CaptureAdapter>,  // Doesn't impl Debug
    pub fields: NarrationFields,
    // ... other fields
}

impl fmt::Debug for World {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("World")
            .field("adapter_present", &self.adapter.is_some())
            .field("fields", &self.fields)
            .finish()
    }
}
```

**Option 2**: Add Debug to CaptureAdapter (upstream fix)
```rust
// In narration-core/src/capture.rs
#[derive(Debug)]  // Add this
pub struct CaptureAdapter {
    // ...
}
```

**Option 3**: Skip Debug derive entirely
```rust
#[derive(cucumber::World)]  // No Debug
pub struct World {
    pub adapter: Option<CaptureAdapter>,
    // ...
}
```

## Cargo.toml Configuration

### From proof-bundle/bdd/Cargo.toml

```toml
[package]
name = "observability-narration-core-bdd"
version = "0.0.0"
edition = "2021"
publish = false

[[bin]]
name = "bdd-runner"
path = "src/main.rs"

[dependencies]
observability-narration-core = { path = "..", features = ["test-support", "otel"] }
cucumber = "0.20"
tokio = { workspace = true, features = ["macros", "rt-multi-thread"] }
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
```

## Directory Structure

```
libs/observability/narration-core/bdd/
├── Cargo.toml
├── src/
│   ├── main.rs              # Runner entrypoint
│   └── steps/
│       ├── mod.rs           # Re-export all step modules
│       ├── world.rs         # World struct
│       ├── core_narration.rs
│       ├── auto_injection.rs
│       └── ...
└── tests/
    └── features/
        ├── core_narration.feature
        ├── auto_injection.feature
        └── ...
```

## The Fix: Simple Regex Strings

### What Works (from real examples)

```rust
// Literal match - use plain string
#[given("a clean capture adapter")]

// Simple regex - use plain string
#[when(regex = "^I create a model with id (.+) and digest (.+)$")]

// Raw string for complex patterns (but still no internal quotes!)
#[when(regex = r"^I enqueue a completion task with valid payload$")]

// Capture groups with plain strings
#[when(regex = "^I get model (.+)$")]
```

### Key Rules

1. **Default to plain strings**: `"..."` works for 99% of cases
2. **Use raw strings for backslashes**: `r"..."` when you have `\d`, `\s`, etc.
3. **Never escape quotes**: Cucumber's regex parser doesn't need it
4. **Never use alternate delimiters**: `r#"..."#` confuses the macro parser

## Summary: The Root Cause

My mistake was **over-engineering the regex escaping**. I tried:
1. First: `r"^I narrate with actor \"([^\"]+)\"$"` - escaped quotes (WRONG)
2. Then: `r#"^I narrate with actor "([^"]+)"$"#` - alternate delimiter (ALSO WRONG)

The correct solution is **embarrassingly simple**:
```rust
#[when(regex = "^I narrate with actor (.+), action (.+), target (.+), and human (.+)$")]
```

Just use **plain strings with capture groups** - no quotes around the captured values in the regex!

## Action Plan

1. ✅ Fix World struct: Add manual Debug impl
2. ✅ Fix all regex patterns: Use plain strings, remove quotes from regex
3. ✅ Update Cargo.toml: Add tokio features
4. ✅ Verify against working examples (orchestratord, proof-bundle)
5. ✅ Run `cargo build` to confirm compilation
6. ✅ Run `cargo run -p observability-narration-core-bdd` to execute tests

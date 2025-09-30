# pool-managerd BDD Scaffolding - Complete ✓

**Status**: Scaffolding complete and verified  
**Date**: 2025-09-30  
**Pattern**: Follows monorepo BDD standards from orchestratord-bdd, catalog-core-bdd, orchestrator-core-bdd

## What Was Created

### 1. Core Structure
```
bin/pool-managerd/bdd/
├── Cargo.toml                    # BDD harness crate
├── README.md                     # Documentation with spec links
├── src/
│   ├── main.rs                   # bdd-runner binary
│   └── steps/
│       ├── mod.rs                # Step module registry
│       └── world.rs              # Cucumber World (BddWorld)
├── tests/
│   └── features/
│       └── .gitkeep              # Ready for .feature files
└── .docs/
    ├── BDD_SCAFFOLDING.md        # Architecture overview
    └── PATTERN_COMPARISON.md     # Pattern compliance verification
```

### 2. Workspace Integration
- Added `"bin/pool-managerd/bdd"` to workspace members in root `Cargo.toml`
- Package name: `pool-managerd-bdd`
- Binary target: `bdd-runner`

### 3. Dependencies
- cucumber 0.20 with macros
- tokio with multi-thread runtime
- futures, axum, http, hyper, tower
- Parent crate: `pool-managerd`
- Workspace-shared: regex, walkdir, serde_json

## Verification Results

### Build ✓
```bash
$ cargo check -p pool-managerd-bdd
Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.87s
```

### Binary ✓
```bash
$ cargo build -p pool-managerd-bdd --bin bdd-runner
Finished `dev` profile [unoptimized + debuginfo] target(s) in 11.89s
```

### Format ✓
```bash
$ cargo fmt --package pool-managerd-bdd -- --check
# No issues
```

## Pattern Compliance

| Aspect | Requirement | Status |
|--------|-------------|--------|
| Directory structure | Match catalog-core-bdd | ✓ |
| Cargo.toml format | Standard BDD pattern | ✓ |
| Binary name | `bdd-runner` | ✓ |
| World struct | `#[derive(cucumber::World)]` | ✓ |
| LLORCH_BDD_FEATURE_PATH | Environment variable support | ✓ |
| Default features path | `tests/features/` | ✓ |
| Tokio runtime | Multi-threaded | ✓ |
| Workspace integration | Added to members | ✓ |

## Spec Traceability

BDD scenarios will cover `.specs/30-pool-managerd.md` requirements:

- **OC-POOL-3001**: Preload at serve start, no Ready until success
- **OC-POOL-3002**: Preload fail-fast on insufficient VRAM/RAM
- **OC-POOL-3003**: Readiness endpoints reflect preload state
- **OC-POOL-3010**: Driver/CUDA errors trigger Unready + backoff-restart
- **OC-POOL-3011**: Restart storms bounded by exponential backoff
- **OC-POOL-3012**: CPU inference spillover disallowed
- **OC-POOL-3020**: Placement respects device masks
- **OC-POOL-3021**: Heterogeneous split ratios explicit and capped
- **OC-POOL-3030**: Emit preload outcomes, VRAM/RAM utilization, driver_reset events

## Usage Examples

### Run all features (when added)
```bash
cargo run -p pool-managerd-bdd --bin bdd-runner
```

### Run specific feature
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/preload.feature \
  cargo run -p pool-managerd-bdd --bin bdd-runner
```

### Run specific directory
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/restart/ \
  cargo run -p pool-managerd-bdd --bin bdd-runner
```

### Integration with dev loop
```bash
# Will include pool-managerd-bdd once features are added
cargo xtask dev:loop
```

## Next Steps (Implementation Phase)

### 1. Expand World State
Add to `src/steps/world.rs`:
```rust
pub struct BddWorld {
    // HTTP client for pool-managerd API
    pub client: Option<reqwest::Client>,
    pub base_url: String,
    
    // Response tracking
    pub last_status: Option<http::StatusCode>,
    pub last_headers: Option<http::HeaderMap>,
    pub last_body: Option<String>,
    
    // Test fixtures
    pub pool_id: Option<String>,
    pub engine_config: Option<EngineConfig>,
    pub device_mask: Option<String>,
    
    // Assertion helpers
    pub facts: Vec<serde_json::Value>,
}
```

### 2. Create First Feature
Example: `tests/features/preload_readiness.feature`
```gherkin
Feature: Preload and Readiness Lifecycle
  # Traceability: OC-POOL-3001, OC-POOL-3002, OC-POOL-3003
  
  Scenario: Worker preloads successfully and becomes ready
    Given a pool-managerd instance
    And a valid engine configuration
    When the worker starts serving
    Then preload completes successfully
    And the readiness endpoint returns ready
    And no error is recorded
```

### 3. Implement Step Definitions
Create `src/steps/preload.rs`:
```rust
use crate::steps::world::BddWorld;
use cucumber::{given, then, when};

#[given(regex = r"^a pool-managerd instance$")]
pub async fn given_pool_managerd(world: &mut BddWorld) {
    // Setup test instance
}

#[when(regex = r"^the worker starts serving$")]
pub async fn when_worker_starts(world: &mut BddWorld) {
    // Trigger serve start
}

#[then(regex = r"^preload completes successfully$")]
pub async fn then_preload_succeeds(world: &mut BddWorld) {
    // Assert preload success
}
```

### 4. Add Step Validation
Create `tests/bdd.rs` (like orchestratord-bdd):
```rust
#[test]
fn features_have_no_undefined_or_ambiguous_steps() {
    // Validate all steps in .feature files are defined
}
```

### 5. Add Registry Function
In `src/steps/mod.rs`:
```rust
pub mod preload;
pub mod restart_backoff;
pub mod device_masks;
pub mod observability;
pub mod world;

use regex::Regex;

pub fn registry() -> Vec<Regex> {
    vec![
        Regex::new(r"^a pool-managerd instance$").unwrap(),
        // ... all step patterns
    ]
}
```

## References

- **Spec**: `.specs/30-pool-managerd.md`
- **Requirements**: `requirements/30-pool-managerd.yaml` (to be created)
- **Parent crate**: `bin/pool-managerd/`
- **Pattern examples**:
  - `bin/orchestratord/bdd/` (full pattern)
  - `libs/catalog-core/bdd/` (minimal pattern)
  - `libs/orchestrator-core/bdd/` (minimal pattern)

## Documentation

- **BDD_SCAFFOLDING.md**: Architecture and usage guide
- **PATTERN_COMPARISON.md**: Detailed pattern compliance verification
- **README.md**: Standard crate documentation with spec traceability

## Compliance with AGENTS.md

✓ **Spec-First Workflow**: Scaffolding references `.specs/30-pool-managerd.md`  
✓ **Coding Style**: Follows rustfmt defaults, Clippy clean  
✓ **Testing Guidelines**: BDD structure ready for behavior tests  
✓ **Commit Discipline**: Ready for `ORCH-####:` prefixed commits  
✓ **Build Commands**: Integrates with `cargo xtask dev:loop`  

## Summary

The BDD scaffolding for `pool-managerd` is **complete and production-ready**. It follows all established monorepo patterns, integrates cleanly with the workspace, and is ready for feature and step implementation.

The scaffolding provides a solid foundation for implementing behavior-driven tests that will verify all requirements from the pool-managerd spec, ensuring compliance with the spec-first, test-driven development workflow defined in `AGENTS.md`.

**Status**: ✅ SCAFFOLDING COMPLETE - Ready for feature implementation

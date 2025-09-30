# BDD Pattern Comparison

This document compares the `pool-managerd-bdd` scaffolding with existing BDD crates in the monorepo to ensure consistency.

## Structure Comparison

### Minimal Pattern (catalog-core-bdd, orchestrator-core-bdd)

```
libs/{crate}/bdd/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs
│   └── steps/
│       ├── mod.rs
│       └── world.rs
└── tests/
    └── features/
```

### Full Pattern (orchestratord-bdd)

```
bin/orchestratord/bdd/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                    # Optional: for step registry
│   ├── main.rs
│   └── steps/
│       ├── mod.rs                # Includes registry() function
│       ├── world.rs              # Rich World with HTTP helpers
│       ├── control_plane.rs      # Domain-specific steps
│       ├── data_plane.rs
│       ├── security.rs
│       └── ...
└── tests/
    ├── bdd.rs                    # Step validation test
    ├── steps.rs                  # Test helper
    └── features/
        ├── control_plane/
        ├── data_plane/
        ├── security/
        └── sse/
```

### pool-managerd-bdd (Current)

```
bin/pool-managerd/bdd/
├── Cargo.toml                    ✓ Matches pattern
├── README.md                     ✓ Matches pattern
├── src/
│   ├── main.rs                   ✓ Matches pattern
│   └── steps/
│       ├── mod.rs                ✓ Matches pattern
│       └── world.rs              ✓ Matches pattern (minimal start)
└── tests/
    └── features/
        └── .gitkeep              ✓ Ready for features
```

## File-by-File Comparison

### Cargo.toml

| Feature | catalog-core-bdd | orchestratord-bdd | pool-managerd-bdd | Status |
|---------|------------------|-------------------|-------------------|--------|
| Package name pattern | `{crate}-bdd` | `{crate}-bdd` | `pool-managerd-bdd` | ✓ |
| Feature flag | `bdd-cucumber` | `bdd-cucumber` | `bdd-cucumber` | ✓ |
| cucumber version | 0.20 | 0.20 | 0.20 | ✓ |
| Binary name | `bdd-runner` | `bdd-runner` | `bdd-runner` | ✓ |
| Parent crate dep | Yes | Yes | Yes | ✓ |
| tokio multi-thread | Yes | Yes | Yes | ✓ |

### main.rs

| Feature | catalog-core-bdd | orchestratord-bdd | pool-managerd-bdd | Status |
|---------|------------------|-------------------|-------------------|--------|
| `#[tokio::main]` | ✓ | ✓ | ✓ | ✓ |
| `LLORCH_BDD_FEATURE_PATH` | ✓ | ✓ | ✓ | ✓ |
| Default path | `tests/features` | `tests/features` | `tests/features` | ✓ |
| Absolute path handling | ✓ | ✓ | ✓ | ✓ |
| `.run_and_exit()` | ✓ | ✓ | ✓ | ✓ |

### steps/world.rs

| Feature | catalog-core-bdd | orchestratord-bdd | pool-managerd-bdd | Status |
|---------|------------------|-------------------|-------------------|--------|
| `#[derive(cucumber::World)]` | ✓ | ✓ | ✓ | ✓ |
| `Debug` trait | ✓ | ✓ | ✓ | ✓ |
| `Default` trait | ✓ | ✓ | ✓ | ✓ |
| World name | `BddWorld` | `World` | `BddWorld` | ✓ |

**Note**: `pool-managerd-bdd` uses `BddWorld` (like catalog-core) rather than `World` (like orchestratord). This is intentional to avoid naming conflicts and follows the simpler pattern.

### steps/mod.rs

| Feature | catalog-core-bdd | orchestratord-bdd | pool-managerd-bdd | Status |
|---------|------------------|-------------------|-------------------|--------|
| `pub mod world;` | ✓ | ✓ | ✓ | ✓ |
| Step modules | None | Multiple | None (yet) | ✓ |
| `registry()` function | No | Yes | No (yet) | ⏳ |

### README.md

| Section | catalog-core-bdd | orchestratord-bdd | pool-managerd-bdd | Status |
|---------|------------------|-------------------|-------------------|--------|
| Name & Purpose | ✓ | ✓ | ✓ | ✓ |
| Spec traceability | ✓ | ✓ | ✓ | ✓ |
| Build & Test | ✓ | ✓ | ✓ | ✓ |
| Runbook | ✓ | ✓ | ✓ | ✓ |
| Status & Owners | ✓ | ✓ | ✓ | ✓ |

## Workspace Integration

### Cargo.toml workspace members

```toml
# All BDD crates follow the same pattern:
members = [
    "libs/orchestrator-core",
    "libs/orchestrator-core/bdd",      # ✓ Pattern
    "bin/orchestratord",
    "bin/orchestratord/bdd",           # ✓ Pattern
    "bin/pool-managerd",
    "bin/pool-managerd/bdd",           # ✓ Pattern (NEW)
    "libs/catalog-core",
    "libs/catalog-core/bdd",           # ✓ Pattern
]
```

## Evolution Path

The `pool-managerd-bdd` scaffolding starts minimal (like catalog-core-bdd) and can evolve toward the full pattern (like orchestratord-bdd) as needed:

### Phase 1: Minimal (CURRENT)
- ✓ Basic World struct
- ✓ No step definitions yet
- ✓ No features yet
- ✓ No validation tests yet

### Phase 2: Basic Features (NEXT)
- Add first feature file (e.g., `preload_readiness.feature`)
- Expand World with HTTP client and response tracking
- Add first step module (e.g., `preload.rs`)
- Import step module in `steps/mod.rs`

### Phase 3: Full Coverage
- Multiple feature files organized by domain
- Multiple step modules (preload, restart_backoff, device_masks, observability)
- Add `registry()` function for step validation
- Add `tests/bdd.rs` for undefined/ambiguous step detection
- Rich World with domain-specific helpers

### Phase 4: Advanced (OPTIONAL)
- Add `lib.rs` if step registry needs to be public
- Add proof bundle integration
- Add metrics verification helpers
- Add chaos/fault injection helpers

## Key Differences from orchestratord-bdd

1. **World naming**: Uses `BddWorld` instead of `World` (simpler, avoids conflicts)
2. **No lib.rs**: Not needed for basic BDD harness
3. **Minimal start**: Will grow organically based on actual test needs
4. **No AppState**: pool-managerd has different state management than orchestratord

## Verification

```bash
# Build check
cargo check -p pool-managerd-bdd
# ✓ PASS

# Format check
cargo fmt --package pool-managerd-bdd -- --check
# ✓ PASS

# Build binary
cargo build -p pool-managerd-bdd --bin bdd-runner
# ✓ PASS

# Run (will fail gracefully with no features)
cargo run -p pool-managerd-bdd --bin bdd-runner
# ✓ Expected: exits with no scenarios found
```

## Conclusion

The `pool-managerd-bdd` scaffolding **perfectly matches** the established monorepo patterns:
- ✓ Directory structure identical to catalog-core-bdd
- ✓ File naming conventions consistent
- ✓ Cargo.toml follows workspace patterns
- ✓ main.rs respects LLORCH_BDD_FEATURE_PATH
- ✓ World struct properly derived
- ✓ Workspace integration complete
- ✓ Ready for feature and step implementation

The scaffolding is production-ready and follows spec-first, test-driven development principles from `AGENTS.md`.

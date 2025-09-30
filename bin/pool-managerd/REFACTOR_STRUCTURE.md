# pool-managerd Structure Refactoring

## Current Structure (Flat, Messy)

```
src/
├── lib.rs
├── main.rs
├── backoff.rs          # Supervision
├── devicemasks.rs      # Placement
├── drain.rs            # Lifecycle
├── health.rs           # Core
├── hetero_split.rs     # Placement
├── preflight.rs        # Validation
├── preload.rs          # Lifecycle
└── registry/           # Core
    ├── entry.rs
    ├── snapshot.rs
    └── types.rs
```

**Problems:**
- Flat structure mixes concerns (lifecycle, placement, validation, core)
- No clear grouping by responsibility
- Hard to navigate (9 files at root level)
- Unclear which modules are related

---

## Proposed Structure (Organized by Domain)

```
src/
├── lib.rs              # Public API exports
├── main.rs             # Daemon entrypoint (stub)
│
├── core/               # Core abstractions (always needed)
│   ├── mod.rs
│   ├── health.rs       # HealthStatus struct
│   └── registry/       # Pool registry
│       ├── mod.rs
│       ├── entry.rs
│       ├── snapshot.rs
│       └── types.rs
│
├── lifecycle/          # Pool lifecycle management
│   ├── mod.rs
│   ├── preload.rs      # Spawn engines, health check, handoff
│   ├── drain.rs        # Drain and reload
│   └── supervision.rs  # Backoff and restart (rename from backoff.rs)
│
├── placement/          # GPU placement and device management
│   ├── mod.rs
│   ├── devicemasks.rs  # Device mask parsing/validation
│   └── hetero_split.rs # Heterogeneous GPU split planning
│
└── validation/         # Preflight checks
    ├── mod.rs
    └── preflight.rs    # GPU-only enforcement, CUDA detection
```

---

## Rationale

### Core Domain
- **health.rs** — fundamental type used everywhere
- **registry/** — central state, used by all other modules

### Lifecycle Domain
- **preload.rs** — spawn and initialize engines
- **drain.rs** — graceful shutdown and reload
- **supervision.rs** (renamed from backoff.rs) — restart policies

### Placement Domain
- **devicemasks.rs** — GPU selection and validation
- **hetero_split.rs** — multi-GPU tensor split planning

### Validation Domain
- **preflight.rs** — environment checks before operations

---

## Benefits

1. **Clear Grouping:** Related modules are together
2. **Easier Navigation:** Find what you need by domain
3. **Better Imports:** `use pool_managerd::lifecycle::preload` is clearer than `use pool_managerd::preload`
4. **Scalability:** Easy to add new modules to appropriate domain
5. **Documentation:** Each domain can have its own README

---

## Migration Plan

1. Create domain directories: `core/`, `lifecycle/`, `placement/`, `validation/`
2. Move files with `git mv` (preserves history)
3. Update `lib.rs` to re-export from new locations
4. Update internal imports
5. Verify all tests still pass

---

## Public API (lib.rs)

Keep public API unchanged for consumers:

```rust
// Re-export core types
pub use core::health;
pub use core::registry;

// Re-export lifecycle
pub use lifecycle::{preload, drain};

// Re-export placement (when implemented)
pub use placement::{devicemasks, hetero_split};

// Re-export validation
pub use validation::preflight;
```

Consumers still import as: `use pool_managerd::preload;`

# WASM Compatibility for Contracts

**Status:** ✅ ENFORCED via CI

## Why WASM Compatibility Matters

Contract crates (`operations-contract`, `api-types`, `config-schema`) are **pure type definitions** that must compile to WASM for use in:

1. **Browser environments** - TypeScript SDK via wasm-bindgen
2. **Edge workers** - Cloudflare Workers, Deno Deploy
3. **WASM plugins** - Future extensibility
4. **Cross-platform** - Single source of truth for all platforms

## Rules

### ✅ Allowed in Contracts

- Pure data structures (`struct`, `enum`)
- Serde serialization/deserialization
- Simple helper methods (no I/O, no threads)
- `const` functions
- Pattern matching and logic

### ❌ Forbidden in Contracts

- `std::io::*` - File I/O operations
- `std::fs::*` - Filesystem access
- `std::path::*` - Path manipulation
- `std::env::*` - Environment variables
- `std::process::*` - Process spawning
- `std::thread::*` - Threading
- `tokio::*` - Async runtime (use in implementation crates, not contracts)

## Enforcement

### 1. Clippy Rules

Each contract crate has a `.clippy.toml` that disallows non-WASM types:

```toml
disallowed-types = [
    "std::io::Error",
    "std::fs::File",
    "std::path::PathBuf",
    # ... etc
]
```

### 2. CI Check

`.github/workflows/contracts-wasm-check.yml` runs on every PR:

```bash
cargo check -p operations-contract --target wasm32-unknown-unknown
cargo clippy -p operations-contract --target wasm32-unknown-unknown -- -D warnings
```

### 3. Local Testing

Test WASM compatibility locally:

```bash
# Install WASM target
rustup target add wasm32-unknown-unknown

# Check all contracts
cargo check -p operations-contract --target wasm32-unknown-unknown
cargo check -p contracts-api-types --target wasm32-unknown-unknown
cargo check -p contracts-config-schema --target wasm32-unknown-unknown
```

## Current Status

### operations-contract ✅

- **Pure types:** Operation enum, request/response structs
- **Helper methods:** `name()`, `hive_id()`, `target_server()` - All WASM-compatible
- **No I/O:** All logic is pure computation

### Implementation Pattern

**❌ WRONG - I/O in contract:**
```rust
// DON'T DO THIS IN CONTRACTS
impl Operation {
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, serde_json::to_string(self)?)
    }
}
```

**✅ RIGHT - Pure logic only:**
```rust
// DO THIS - Pure computation, no I/O
impl Operation {
    pub fn name(&self) -> &'static str {
        match self {
            Operation::Status => "status",
            Operation::Infer(_) => "infer",
            // ... etc
        }
    }
}
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Contracts (WASM-compatible)                     │
│ - Pure types                                    │
│ - Serde traits                                  │
│ - Helper methods (no I/O)                       │
└─────────────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌───────────────┐       ┌──────────────┐
│ Native Rust   │       │ WASM/Browser │
│ - queen-rbee  │       │ - TypeScript │
│ - rbee-hive   │       │ - SDK        │
│ - rbee-keeper │       │ - Edge       │
└───────────────┘       └──────────────┘
```

## Benefits

1. **Single source of truth** - Types defined once, used everywhere
2. **Type safety** - Compile-time guarantees across platforms
3. **Zero runtime overhead** - Direct memory layout in WASM
4. **Future-proof** - Ready for edge computing, plugins, etc.

## FAQ

**Q: Can I use `PathBuf` in operation_impl.rs?**  
A: No. Use `String` for paths in contracts, convert to `PathBuf` in implementation crates.

**Q: What about async operations?**  
A: Contracts define the *types*. Implementation crates (queen-rbee, rbee-hive) handle async execution.

**Q: Can I add a method that reads a file?**  
A: No. Add it to an implementation crate instead. Contracts are pure types.

**Q: How do I test if my change is WASM-compatible?**  
A: Run `cargo check -p operations-contract --target wasm32-unknown-unknown`

## Related

- [Cargo.toml](./operations-contract/Cargo.toml) - WASM feature flag
- [.clippy.toml](./operations-contract/.clippy.toml) - Enforcement rules
- [CI Workflow](../../.github/workflows/contracts-wasm-check.yml) - Automated checks

# Metadata Module

Modularized test metadata system for proof bundles.

## Structure

```
metadata/
├── mod.rs          # Public API and module organization
├── types.rs        # Core TestMetadata struct (70 lines)
├── builder.rs      # Fluent builder API (300 lines)
├── parser.rs       # Doc comment parser (150 lines)
└── helpers.rs      # Query functions (100 lines)
```

## Responsibilities

### `types.rs`
- **TestMetadata** struct definition
- Serde serialization configuration
- Field documentation

### `builder.rs`
- **TestMetadataBuilder** for fluent API
- `test_metadata()` entry point
- All builder methods (priority, spec, team, etc.)
- `record()` and `build()` methods

### `parser.rs`
- `parse_doc_comments()` function
- Annotation extraction logic
- Support for `@key: value` syntax
- Custom fields, lists (requires, tags)

### `helpers.rs`
- `is_critical()` - Check if test is critical priority
- `is_high_priority()` - Check if test is high/critical
- `is_flaky()` - Check if test is marked flaky
- `priority_level()` - Get numeric priority for sorting
- `requires_resource()` - Check resource requirements
- `has_tag()` - Check for specific tags

## Public API

All public items are re-exported from `mod.rs`:

```rust
pub use builder::{test_metadata, TestMetadataBuilder};
pub use helpers::{has_tag, is_critical, is_flaky, is_high_priority, priority_level, requires_resource};
pub use parser::parse_doc_comments;
pub use types::TestMetadata;
```

## Benefits of Modularization

1. **Single Responsibility**: Each file has one clear purpose
2. **Easier Navigation**: ~100-300 lines per file vs 457 lines monolith
3. **Better Testing**: Each module can be tested independently
4. **Clearer Documentation**: Module-level docs explain scope
5. **Maintainability**: Changes are localized to specific files

## Migration

The old `metadata.rs` file has been replaced with the `metadata/` directory.
All public APIs remain unchanged - this is a pure refactoring.

## Tests

Each module includes its own unit tests:
- `parser.rs`: 6 tests for annotation parsing
- `helpers.rs`: 6 tests for query functions
- `mod.rs`: 4 integration tests

Total: 16 tests covering all metadata functionality.

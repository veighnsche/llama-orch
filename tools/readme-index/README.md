# readme-index

**Generates root README index from workspace crate READMEs**

`tools/readme-index` — Scans workspace for README files and generates navigation index.

---

## What This Tool Does

readme-index provides **documentation indexing** for llama-orch:

- **Scan workspace** — Find all README.md files
- **Extract metadata** — Parse titles and descriptions
- **Generate index** — Create navigation table
- **Update root README** — Insert index into root README.md
- **Idempotent** — Deterministic output

**Purpose**: Keep root README synchronized with workspace structure

---

## Usage

### Generate Index

```bash
# Regenerate README index
cargo run -p tools-readme-index --quiet
```

This updates the root `README.md` with:
- List of all crates
- Brief descriptions
- Links to individual READMEs

---

## Index Format

### Generated Section

```markdown
## Workspace Index

### Binaries

- **[orchestratord](./bin/orchestratord/README.md)** — Main orchestrator service
- **[pool-managerd](./bin/pool-managerd/README.md)** — GPU node pool manager

### Libraries

- **[orchestrator-core](./libs/orchestrator-core/README.md)** — Core orchestration logic
- **[catalog-core](./libs/catalog-core/README.md)** — Model catalog and registry
- **[adapter-host](./libs/adapter-host/README.md)** — Worker adapter host

### Test Harness

- **[bdd](./test-harness/bdd/README.md)** — BDD test runner
- **[chaos](./test-harness/chaos/README.md)** — Chaos engineering tests
- **[determinism-suite](./test-harness/determinism-suite/README.md)** — Determinism verification

### Tools

- **[openapi-client](./tools/openapi-client/README.md)** — Generated HTTP client
- **[readme-index](./tools/readme-index/README.md)** — README index generator
- **[spec-extract](./tools/spec-extract/README.md)** — Specification extractor
```

---

## Metadata Extraction

The tool extracts metadata from each README:

```rust
struct CrateMetadata {
    name: String,
    path: String,
    description: String,
    category: Category,
}

enum Category {
    Binary,
    Library,
    TestHarness,
    Tool,
}
```

### Parsing Rules

1. **Name**: First H1 heading (`# name`)
2. **Description**: First paragraph or bold text after title
3. **Category**: Inferred from path (`bin/`, `libs/`, `test-harness/`, `tools/`)

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p tools-readme-index -- --nocapture
```

### Dry Run

```bash
# Preview without writing
cargo run -p tools-readme-index -- --dry-run
```

---

## CI Integration

### GitHub Actions

```yaml
- name: Check README index
  run: |
    cargo run -p tools-readme-index --quiet
    git diff --exit-code README.md
```

This ensures the index is up-to-date in CI.

---

## Dependencies

### Internal

- None (standalone tool)

### External

- `walkdir` — Directory traversal
- `regex` — Markdown parsing

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers

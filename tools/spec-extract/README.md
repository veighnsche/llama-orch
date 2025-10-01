# spec-extract

**Extracts requirements from specification Markdown files**

`tools/spec-extract` — Parses spec files and generates YAML requirement documents.

---

## What This Tool Does

spec-extract provides **requirement extraction** for llama-orch:

- **Parse specs** — Extract requirements from `.specs/*.md` files
- **RFC-2119 keywords** — Identify MUST, SHOULD, MAY requirements
- **Stable IDs** — Preserve requirement identifiers (ORCH-3001, etc.)
- **Generate YAML** — Output structured requirement files
- **Traceability** — Link specs to requirements to tests

**Purpose**: Maintain spec → requirements → tests traceability

---

## Usage

### Extract Requirements

```bash
# Extract requirements from all specs
cargo run -p tools-spec-extract --quiet

# Extract from specific spec
cargo run -p tools-spec-extract -- .specs/00_llama-orch.md
```

This generates:
- `requirements/00_llama-orch.yaml` — Structured requirements
- `requirements/index.yaml` — Requirement index

---

## Spec Format

### Requirement IDs

Requirements use stable IDs in specs:

```markdown
## Queue Management

**ORCH-3001**: The orchestrator MUST maintain a FIFO queue per pool.

**ORCH-3002**: The orchestrator SHOULD prioritize jobs by pool capacity.

**ORCH-3003**: The orchestrator MAY implement priority queues (future).
```

### RFC-2119 Keywords

- **MUST** — Mandatory requirement
- **MUST NOT** — Prohibited behavior
- **SHOULD** — Recommended requirement
- **SHOULD NOT** — Not recommended
- **MAY** — Optional feature

---

## Output Format

### YAML Requirements

```yaml
requirements:
  - id: ORCH-3001
    level: MUST
    description: The orchestrator MUST maintain a FIFO queue per pool
    category: queue
    spec: .specs/00_llama-orch.md
    
  - id: ORCH-3002
    level: SHOULD
    description: The orchestrator SHOULD prioritize jobs by pool capacity
    category: queue
    spec: .specs/00_llama-orch.md
    
  - id: ORCH-3003
    level: MAY
    description: The orchestrator MAY implement priority queues (future)
    category: queue
    spec: .specs/00_llama-orch.md
    status: future
```

---

## Traceability

### Spec → Requirements → Tests

```
.specs/00_llama-orch.md (ORCH-3001)
  ↓
requirements/00_llama-orch.yaml (ORCH-3001)
  ↓
tests/test_queue.rs (test_fifo_queue)
```

### Verification

```bash
# Check all requirements have tests
cargo run -p tools-spec-extract -- --verify

# Output
✅ ORCH-3001: Covered by test_fifo_queue
✅ ORCH-3002: Covered by test_priority
❌ ORCH-3003: No test coverage (MAY requirement)
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p tools-spec-extract -- --nocapture
```

### Validation

```bash
# Validate spec format
cargo run -p tools-spec-extract -- --validate .specs/00_llama-orch.md
```

---

## CI Integration

### GitHub Actions

```yaml
- name: Extract requirements
  run: cargo run -p tools-spec-extract --quiet

- name: Check for changes
  run: |
    git diff --exit-code requirements/
```

This ensures requirements are up-to-date in CI.

---

## Dependencies

### Internal

- None (standalone tool)

### External

- `regex` — Pattern matching
- `serde` — Serialization
- `serde_yaml` — YAML output

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers

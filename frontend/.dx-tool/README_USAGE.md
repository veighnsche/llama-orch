# Frontend DX Tool - Usage Guide

**Created by:** TEAM-DX-001

A Rust-based CLI tool for verifying CSS and HTML changes without browser access.

Tailored for the **Nuxt + Tailwind** setup in `bin/commercial` and `libs/storybook`.

---

## Installation

```bash
cd frontend/.dx-tool
cargo build --release

# Optional: Add to PATH
export PATH="$PATH:$(pwd)/target/release"
```

---

## Quick Start

### Workspace-Aware Commands

The tool knows about your projects! Use `--project` to target commercial or storybook:

```bash
# Check class in commercial frontend (localhost:3000)
dx --project commercial css --class-exists "cursor-pointer"

# Check class in storybook (localhost:6006)
dx --project storybook css --class-exists "hover:bg-blue-500"
```

### Explicit URL

You can still provide explicit URLs:

```bash
dx css --class-exists "cursor-pointer" http://localhost:3000
dx css --class-exists "text-red-500" https://rbee.app
```

---

## Configuration

The tool uses `.dxrc.json` in the `frontend/` directory for workspace defaults:

```json
{
  "workspace": {
    "commercial": {
      "url": "http://localhost:3000",
      "port": 3000
    },
    "storybook": {
      "url": "http://localhost:6006",
      "port": 6006
    }
  }
}
```

---

## Common Workflows

### Verify Tailwind Class in Commercial

```bash
# Start commercial dev server
cd frontend/bin/commercial
pnpm dev

# In another terminal, verify class
cd frontend/.dx-tool
dx --project commercial css --class-exists "cursor-pointer"
```

### Verify Storybook Component Styles

```bash
# Start storybook
cd frontend/libs/storybook
pnpm story:dev

# Verify class
cd frontend/.dx-tool
dx --project storybook css --class-exists "btn-primary"
```

### CI/CD Integration

```bash
# In GitHub Actions or CI pipeline
dx --project commercial css --class-exists "cursor-pointer"
dx --project storybook css --class-exists "text-foreground"
```

---

## Examples

### Success Case

```bash
$ dx --project commercial css --class-exists "cursor-pointer"
✓ Class 'cursor-pointer' found in stylesheet
```

### Failure Case

```bash
$ dx --project commercial css --class-exists "nonexistent-class"
✗ Error: Class 'nonexistent-class' not found in stylesheet
  Possible causes:
    - Class not used in any component
    - Tailwind not scanning source files
    - Class tree-shaken by build tool
```

### Using Explicit URL

```bash
$ dx css --class-exists "cursor-pointer" http://localhost:3000
✓ Class 'cursor-pointer' found in stylesheet
```

---

## Help

```bash
# General help
dx --help

# CSS command help
dx css --help
```

---

## Development

```bash
# Run tests
cargo test

# Build release binary
cargo build --release

# Run with cargo
cargo run -- --project commercial css --class-exists "cursor-pointer"
```

---

## Documentation

- `00_MASTER_PLAN.md` - Overall vision
- `01_CLI_DESIGN.md` - CLI UX design
- `06_IMPLEMENTATION_ROADMAP.md` - Development timeline
- `TEAM_DX_001_HANDOFF.md` - Phase 1 completion summary

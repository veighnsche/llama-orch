# Frontend DX Tool - Planning Documents

A Rust-based CLI tool for verifying CSS and HTML changes without browser access.

Tailored for the **Nuxt + Tailwind** setup in `bin/commercial` and `libs/storybook`.

## Installation

```bash
cd frontend/.dx-tool
cargo build --release

# Optional: Add to PATH
export PATH="$PATH:$(pwd)/target/release"
```

## Quick Start

### Workspace-Aware Commands

The tool knows about your projects! Use `--project` to target commercial or storybook:

```bash
# Check class in commercial frontend (localhost:3000)
dx --project commercial css --class-exists "cursor-pointer"

## Timeline

5 weeks from approval to production-ready tool.

## Engineering Standards

This tool follows the **DX Engineering Rules** (see `/DX_ENGINEERING_RULES.md`):
- ✅ < 2 second response time
- ✅ Deterministic output
- ✅ Clear, actionable errors
- ✅ Single binary distribution
- ✅ Composable with other tools

---

**Next Steps:** Review plans, approve architecture, assign implementation team.

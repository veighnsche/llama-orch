# Scripts Directory

Utility scripts for the llama-orch monorepo.

## Dependency Analysis

### `dependency-graph.py`

Comprehensive dependency graph generator that analyzes both Cargo and pnpm workspaces.

**Quick start:**
```bash
# Show statistics
python scripts/dependency-graph.py

# Generate all formats
./scripts/generate-all-deps.sh
```

**See:** [DEPENDENCY_GRAPH_USAGE.md](./DEPENDENCY_GRAPH_USAGE.md) for full documentation.

### `generate-all-deps.sh`

Convenience script to generate dependency graphs in all formats at once.

```bash
# Generate to default location (.docs/architecture/dependencies/)
./scripts/generate-all-deps.sh

# Generate to custom location
./scripts/generate-all-deps.sh /tmp/deps
```

**Generates:**
- `stats.md` - Human-readable statistics
- `dependencies.json` - Machine-readable data
- `dependencies.mmd` - Mermaid diagram
- `dependencies.dot` - GraphViz DOT format
- `dependencies.png` - PNG image (if graphviz installed)
- `dependencies.svg` - SVG image (if graphviz installed)

### `find-dependency-path.py`

Find dependency paths between two packages (useful for impact analysis).

```bash
# Find how rbee-keeper depends on narration-core
python scripts/find-dependency-path.py rbee-keeper observability-narration-core

# Check for circular dependencies
python scripts/find-dependency-path.py package-a package-b
```

**Shows:**
- Shortest dependency path
- Alternative paths
- Circular dependency detection
- Package types and locations

## Build Scripts

### `build-all.sh` ⭐ NEW

**TEAM-450:** Simple build script for new machines. Just runs `pnpm install && pnpm run build && cargo build --release`.

```bash
./scripts/build-all.sh
```

Turborepo and Cargo workspaces handle all the dependency ordering automatically!

### `quick-start.sh` ⭐ NEW

**TEAM-450:** Quick start for frontend development.

```bash
./scripts/quick-start.sh
```

Installs deps and builds @rbee/ui, then shows dev commands.

## GitHub Setup

### `setup-github-branches.sh` ⭐ NEW

**TEAM-451:** Setup development and production branches.

```bash
# Create branches and set development as default
./scripts/setup-github-branches.sh
```

### `configure-branch-protection.sh` ⭐ NEW

**TEAM-451:** Configure branch protection rules.

```bash
# Protect production, free pushing on development
./scripts/configure-branch-protection.sh
```

**Configuration:**
- `production`: Protected (PR required, CI must pass, triggers releases)
- `development`: Free pushing (no restrictions)

### `release.sh` ⭐ NEW

**TEAM-451:** Bump version using existing tools (Rule Zero).

```bash
# One-time setup
cargo install cargo-workspaces

# Bump version
./scripts/release.sh patch   # 0.1.0 → 0.1.1
./scripts/release.sh minor   # 0.1.0 → 0.2.0
./scripts/release.sh major   # 0.1.0 → 1.0.0
```

**Uses existing tools:**
- `pnpm version` for JavaScript packages
- `cargo workspaces` for Rust crates
- Simple wrapper (not custom bash logic)

See `.docs/VERSION_MANAGEMENT.md` for details.

## Other Scripts

### `check-build-status.sh`

Check build status of all workspace crates.

### `homelab/`

Scripts for homelab deployment and management.

## Requirements

- **Python 3.11+** (for `dependency-graph.py`)
- **GraphViz** (optional, for rendering DOT to images)
  ```bash
  sudo apt install graphviz  # Ubuntu/Debian
  brew install graphviz      # macOS
  ```

## Contributing

When adding new scripts:

1. Add executable permission: `chmod +x scripts/your-script.sh`
2. Include usage documentation in script header
3. Update this README
4. Follow existing patterns (error handling, output formatting)

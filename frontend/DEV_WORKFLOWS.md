# Development Workflows

## Problem
Running `turbo dev` starts ALL workspace packages (10+), which is unnecessary when working on specific areas.

## Solution: Targeted Dev Bundles

### Marketing Bundle (Public Sites)

#### **UI Component Development** (Storybook only)
```bash
pnpm dev:ui
```
**Runs:** `@rbee/ui` (Storybook + TypeScript watch)
**Use when:** Working on design system components in isolation
**Concurrency:** 1 package
**Port:** http://localhost:6006

#### **Marketing Sites** (Commercial + Docs)
```bash
pnpm dev:marketing
```
**Runs:** `@rbee/ui` + `@rbee/commercial` + `@rbee/user-docs`
**Use when:** Working on public-facing marketing/documentation
**Concurrency:** 3 packages
**Ports:**
- Storybook: http://localhost:6006
- Commercial: http://localhost:7822
- Docs: http://localhost:7811

---

### Product Bundle (Internal Apps)

#### **Keeper UI** (Desktop App)
```bash
pnpm dev:keeper
```
**Runs:** `@rbee/keeper-ui`
**Use when:** Working on the Tauri desktop application
**Concurrency:** 1 package
**Dependencies:** `@rbee/ui` (auto-included)

#### **Queen UI** (Control Plane)
```bash
pnpm dev:queen
```
**Runs:** `@rbee/queen-rbee-ui` + `@rbee/queen-rbee-react` + `@rbee/queen-rbee-sdk`
**Use when:** Working on queen-rbee web interface
**Concurrency:** 3 packages
**Port:** http://localhost:7834
**Dependencies:** `@rbee/ui` (auto-included)

#### **Hive UI**
```bash
pnpm dev:hive
```
**Runs:** `@rbee/rbee-hive-ui` + all hive packages
**Use when:** Working on hive management interface
**Concurrency:** ~3 packages

#### **Worker UI**
```bash
pnpm dev:worker
```
**Runs:** `@rbee/llm-worker-ui` + all worker packages
**Use when:** Working on worker management interface
**Concurrency:** ~3 packages

#### **All Product UIs**
```bash
pnpm dev:product
```
**Runs:** All internal application UIs (keeper + queen + hive + worker)
**Use when:** Testing cross-service integration
**Concurrency:** ~10 packages

---

### Everything
```bash
pnpm dev:all
```
**Runs:** ALL workspace packages (marketing + product)
**Use when:** Full system integration testing (rare)
**Concurrency:** 13+ packages

---

## Why This Works

### Automatic Dependency Resolution
Turbo automatically includes dependencies when you filter:
- `--filter='@rbee/commercial'` → automatically includes `@rbee/ui` (dependency)
- `--filter='@rbee/ui'` → includes all shared config packages

### Shared Packages Always Included
These are automatically built/watched when needed:
- `@repo/tailwind-config`
- `@repo/typescript-config`
- `@repo/eslint-config`
- `@repo/vite-config`

### Concurrency Comparison
| Command | Packages | Concurrency | Use Case |
|---------|----------|-------------|----------|
| `dev:ui` | 1 | 1 | Design system only |
| `dev:marketing` | 3 | 3 | Public sites |
| `dev:keeper` | 1 | 1 | Desktop app |
| `dev:queen` | 3 | 3 | Queen control plane |
| `dev:hive` | ~3 | 3 | Hive management |
| `dev:worker` | ~3 | 3 | Worker management |
| `dev:product` | ~10 | 10 | All internal apps |
| `dev:all` | 13+ | 13+ | Everything (rare) |

**Result:** 70-90% reduction in concurrent processes for typical workflows

---

## Quick Reference

```bash
# MARKETING (public sites)
pnpm dev:ui          # Storybook only
pnpm dev:marketing   # Commercial + Docs

# PRODUCT (internal apps)
pnpm dev:keeper      # Desktop app (Tauri)
pnpm dev:queen       # Queen control plane
pnpm dev:hive        # Hive management
pnpm dev:worker      # Worker management
pnpm dev:product     # All internal apps

# EVERYTHING (rare)
pnpm dev:all         # All packages
```

---

## Technical Details

### Turbo Filter Syntax
```bash
# Single package
turbo dev --filter='@rbee/ui'

# Multiple packages (dependencies auto-included)
turbo dev --filter='@rbee/ui' --filter='@rbee/commercial'

# Package + all dependencies (explicit)
turbo dev --filter='@rbee/commercial...'
```

### Why Not Use `...` Syntax?
The `...` syntax includes ALL dependencies recursively, which defeats the purpose of targeted bundles. We use explicit filters to control exactly what runs.

---

## Build Commands (Unchanged)

```bash
# Build UI library
pnpm build:ui

# Build commercial site (includes UI)
pnpm build:commercial

# Build docs site (includes UI)
pnpm build:docs

# Build everything
pnpm build:frontend
```

---

## Troubleshooting

### "Package not found" error
Make sure you're in the root directory (`/home/vince/Projects/llama-orch`)

### Changes not reflecting
1. Check if the right dev server is running
2. Verify port numbers (commercial: 7822, docs: 7811, storybook: 6006)
3. Try `pnpm dev:all` to ensure it's not a dependency issue

### Too many processes
You're probably running `turbo dev` without filters. Use one of the targeted commands above.

---

**Last Updated:** Oct 27, 2025

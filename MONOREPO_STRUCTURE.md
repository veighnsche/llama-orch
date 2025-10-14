# Monorepo Structure & Workflow

## Repository Layout

```
llama-orch/
├── package.json              # Root orchestration (frontend only)
├── pnpm-workspace.yaml       # All workspaces (frontend + consumers)
├── turbo.json                # Turborepo config (frontend only)
│
├── consumers/                # ❌ Independent workspaces
│   ├── llama-orch-utils      # Utility package
│   ├── .examples/*           # Example projects
│   └── llama-orch-sdk/ts     # TypeScript SDK
│
└── frontend/                 # ✅ Orchestrated workspaces
    ├── bin/
    │   ├── commercial        # Marketing site (@rbee/commercial)
    │   └── user-docs         # Documentation (@rbee/user-docs)
    └── libs/
        ├── rbee-ui           # Shared UI components (@rbee/ui)
        └── frontend-tooling  # Shared tooling
```

## Two Separate Worlds

### 1. Frontend Ecosystem (Orchestrated)

**Packages:**
- `@rbee/ui` - Shared UI components
- `@rbee/commercial` - Marketing site
- `@rbee/user-docs` - Documentation site

**Characteristics:**
- Share UI components via `@rbee/ui`
- Same tech stack (Next.js, React, Tailwind CSS)
- Interdependent (apps consume UI library)
- Automatic dev workflow (CSS rebuilds propagate)

**Commands:**
```bash
pnpm run dev:commercial   # UI + Commercial
pnpm run dev:docs         # UI + Docs
pnpm run dev:frontend     # UI + Commercial + Docs
```

### 2. Consumer Ecosystem (Independent)

**Packages:**
- `llama-orch-utils` - Utility functions
- `llama-orch-sdk` - TypeScript SDK
- `.examples/*` - Example projects

**Characteristics:**
- Different purposes (SDK, utils, examples)
- Different tech stacks (varies by package)
- Independent lifecycle (own build/dev scripts)
- No shared UI components

**Commands:**
```bash
pnpm --filter llama-orch-sdk run build
pnpm --filter llama-orch-utils run test
# Each package has its own workflow
```

## Why This Separation?

### Frontend Benefits
✅ **Shared UI**: All apps use same components  
✅ **Automatic updates**: Edit Button → both apps update  
✅ **Single CSS build**: One watcher, multiple consumers  
✅ **Consistent styling**: Design tokens shared across apps  

### Consumer Benefits
✅ **Independence**: Don't need frontend tooling  
✅ **Flexibility**: Each package chooses its own stack  
✅ **Performance**: No unnecessary rebuilds  
✅ **Clarity**: Clear separation of concerns  

## Development Workflows

### Frontend Development

```bash
# Start developing commercial site
pnpm run dev:commercial

# What happens:
# 1. [UI] Tailwind watches @rbee/ui source files
# 2. [UI] Builds dist/index.css (266KB, 9823 lines)
# 3. [APP] Next.js starts on http://localhost:3000
# 4. [APP] Imports @rbee/ui/styles.css → dist/index.css

# Edit a component:
# 1. Edit frontend/libs/rbee-ui/src/atoms/Button/Button.tsx
# 2. [UI] Rebuilds CSS in ~89ms
# 3. [APP] Hot-reloads automatically
# 4. See changes instantly in browser
```

### Consumer Development

```bash
# Work with SDK
cd consumers/llama-orch-sdk/ts
pnpm install
pnpm run build
pnpm run test

# Completely independent from frontend
# No UI components, no CSS, no Next.js
```

## Configuration Files

### pnpm-workspace.yaml (All Packages)
```yaml
packages:
  # Consumers (independent)
  - consumers/llama-orch-utils
  - consumers/.examples/*
  - consumers/llama-orch-sdk/ts
  
  # Frontend (orchestrated)
  - frontend/bin/commercial
  - frontend/bin/user-docs
  - frontend/libs/rbee-ui
  - frontend/libs/frontend-tooling
```

**Purpose:** Defines all workspaces in the monorepo  
**Scope:** Both frontend and consumers  
**Tool:** pnpm workspace management  

### package.json (Root - Frontend Only)
```json
{
  "scripts": {
    "dev:commercial": "concurrently ... @rbee/ui ... @rbee/commercial",
    "dev:docs": "concurrently ... @rbee/ui ... @rbee/user-docs",
    "dev:frontend": "concurrently ... @rbee/ui ... @rbee/commercial ... @rbee/user-docs"
  }
}
```

**Purpose:** Orchestrate frontend development  
**Scope:** Only frontend packages  
**Tool:** concurrently + pnpm filters  

### turbo.json (Frontend Only)
```json
{
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

**Purpose:** Define build pipeline for frontend  
**Scope:** Only packages with `build`/`dev` scripts (frontend)  
**Tool:** Turborepo (optional, can use pnpm filters instead)  

## Dependency Graph

```
Frontend Ecosystem:
┌─────────────────┐
│ @rbee/commercial│──┐
└─────────────────┘  │
                     ├──> ┌──────────┐
┌─────────────────┐  │    │ @rbee/ui │
│ @rbee/user-docs │──┘    └──────────┘
└─────────────────┘            │
                               ├──> src/**/*.tsx (source)
                               └──> dist/index.css (built)

Consumer Ecosystem:
┌──────────────────┐
│ llama-orch-sdk   │  (independent)
└──────────────────┘

┌──────────────────┐
│ llama-orch-utils │  (independent)
└──────────────────┘
```

## Adding New Packages

### Adding a Frontend App

1. Create package in `frontend/bin/my-app`
2. Add to `pnpm-workspace.yaml`
3. Add `@rbee/ui` dependency
4. Import `@rbee/ui/styles.css` in layout
5. Add dev command to root `package.json`

### Adding a Consumer Package

1. Create package in `consumers/my-package`
2. Add to `pnpm-workspace.yaml`
3. Define own build/dev scripts
4. Work independently (no orchestration needed)

## Summary

| Aspect | Frontend | Consumers |
|--------|----------|-----------|
| **Orchestration** | ✅ Yes (concurrently) | ❌ No (independent) |
| **Shared UI** | ✅ Yes (@rbee/ui) | ❌ No |
| **Dev Command** | `pnpm run dev:*` | `pnpm --filter <name> run dev` |
| **Build Pipeline** | ✅ Yes (UI → Apps) | ❌ No (each package builds itself) |
| **Tech Stack** | Next.js, React, Tailwind | Varies by package |
| **Purpose** | User-facing apps | SDKs, utils, examples |

**The frontend is a cohesive ecosystem. Consumers are independent satellites.**

# Frontend Workspace Structure

## Overview

The monorepo contains multiple workspaces, but **only the frontend packages** are orchestrated together for development:

```
rbee/
├── consumers/              # ❌ NOT part of frontend workflow
│   ├── rbee-utils
│   ├── .examples/*
│   └── rbee-sdk/ts
│
└── frontend/              # ✅ Orchestrated together
    ├── apps/
    │   ├── commercial     # Marketing site
    │   └── user-docs      # Documentation site
    └── packages/
        ├── rbee-ui        # Shared UI components
        └── frontend-tooling
```

## Why This Separation?

### Frontend Packages (Orchestrated)
- **Share UI components** via `@rbee/ui`
- **Same tech stack** (Next.js, React, Tailwind)
- **Interdependent** (commercial & docs both use rbee-ui)
- **Automatic dev workflow** (CSS rebuilds propagate to both apps)

### Consumer Packages (Independent)
- **Different purposes** (SDK, utils, examples)
- **Different tech stacks** (TypeScript SDK, Rust bindings, etc.)
- **Independent lifecycle** (don't need UI components)
- **Separate workflows** (own build/dev scripts)

## Development Commands

### Frontend-Only Commands

```bash
# Develop commercial site only
pnpm run dev:commercial
# Runs: @rbee/ui (CSS watcher) + @rbee/commercial (Next.js)

# Develop docs site only
pnpm run dev:docs
# Runs: @rbee/ui (CSS watcher) + @rbee/user-docs (Next.js)

# Develop both sites simultaneously
pnpm run dev:frontend
# Runs: @rbee/ui (CSS watcher) + @rbee/commercial + @rbee/user-docs

# Just UI component development
pnpm run dev:ui
# Runs: @rbee/ui CSS watcher only (for Storybook)
```

### Build Commands

```bash
# Build commercial site
pnpm run build:commercial
# Builds: @rbee/ui CSS → @rbee/commercial

# Build docs site
pnpm run build:docs
# Builds: @rbee/ui CSS → @rbee/user-docs

# Build all frontend
pnpm run build:frontend
# Builds: @rbee/ui CSS → @rbee/commercial + @rbee/user-docs
```

### Consumer Commands (Separate)

```bash
# Work with SDK
pnpm --filter rbee-sdk run build

# Work with utils
pnpm --filter rbee-utils run test

# These are NOT orchestrated with frontend
```

## Workspace Configuration

### pnpm-workspace.yaml
```yaml
packages:
  # Consumers (independent)
  - consumers/rbee-utils
  - consumers/.examples/*
  - consumers/rbee-sdk/ts
  
  # Frontend (orchestrated)
  - frontend/bin/commercial
  - frontend/bin/user-docs
  - frontend/libs/frontend-tooling
  - frontend/libs/rbee-ui
  - frontend/reference/v0
```

### turbo.json (Frontend-focused)
```json
{
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

This configuration only applies to packages with `build` and `dev` scripts (the frontend packages).

## Dependency Graph

```
@rbee/commercial ──┐
                   ├──> @rbee/ui (CSS source)
@rbee/user-docs ───┘         │
                             ├──> dist/index.css (built CSS)
                             │
                             └──> Tailwind scans src/**/*.tsx
```

Both apps consume the same pre-built CSS from `@rbee/ui/styles.css`.

## Why Not Include Consumers?

1. **No shared UI**: Consumers don't use `@rbee/ui` components
2. **Different workflows**: SDK has different build/test needs
3. **Performance**: No need to watch/rebuild when editing frontend
4. **Clarity**: Separate concerns = easier to understand

## Adding New Frontend Apps

To add a new frontend app that uses `@rbee/ui`:

1. **Create the app:**
   ```bash
   mkdir -p frontend/bin/my-new-app
   cd frontend/bin/my-new-app
   pnpm init
   ```

2. **Add to workspace:**
   ```yaml
   # pnpm-workspace.yaml
   packages:
     - frontend/bin/my-new-app  # Add this line
   ```

3. **Install dependencies:**
   ```json
   {
     "dependencies": {
       "@rbee/ui": "workspace:*"
     }
   }
   ```

4. **Import UI styles:**
   ```tsx
   // app/layout.tsx
   import '@rbee/ui/styles.css'
   ```

5. **Add dev command:**
   ```json
   // Root package.json
   {
     "scripts": {
       "dev:my-app": "concurrently \"pnpm --filter @rbee/ui run dev\" \"pnpm --filter @rbee/my-new-app run dev\""
     }
   }
   ```

Done! The new app automatically gets UI component updates.

## Summary

✅ **Frontend packages** (commercial, user-docs, rbee-ui) are orchestrated together  
✅ **Consumer packages** remain independent  
✅ **One command** to develop any frontend app with automatic CSS rebuilds  
✅ **Clear separation** of concerns  
✅ **Scalable** pattern for adding more frontend apps  

**The frontend is a cohesive unit. Consumers are independent satellites.**

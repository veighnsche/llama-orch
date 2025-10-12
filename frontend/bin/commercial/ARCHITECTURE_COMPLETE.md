# Architecture Consolidation Complete

**Date:** 2025-10-12  
**Team:** TEAM-FE-CONSOLIDATE

## Summary

Successfully split the component library between `rbee-storybook` (atoms/molecules) and `commercial` (organisms/templates) with proper workspace dependencies and Histoire instances.

## Final Architecture

```
frontend/
├── libs/
│   └── storybook/                    ← Shared component library
│       ├── stories/
│       │   ├── atoms/                ← 116 atomic components
│       │   ├── molecules/            ← 28 composite components
│       │   └── index.ts              ← Exports atoms & molecules
│       ├── styles/
│       │   ├── tokens-base.css       ← Design tokens (source of truth)
│       │   └── tokens.css            ← Token imports
│       ├── package.json              ← Exports: stories, styles
│       └── histoire.config.ts        ← Port 6006 (default)
│
└── bin/
    └── commercial/                   ← Commercial frontend app
        ├── app/
        │   ├── stories/
        │   │   ├── organisms/        ← 65 page-specific organisms
        │   │   ├── templates/        ← 8 page templates
        │   │   └── index.ts          ← Re-exports atoms/molecules + local organisms
        │   └── assets/css/
        │       └── main.css          ← Imports tokens from workspace
        ├── package.json              ← Depends on rbee-storybook
        └── histoire.config.ts        ← Port 6007
```

## What Was Done

### 1. Removed from Commercial
- ❌ `app/stories/atoms/` (116 items) → Deleted
- ❌ `app/stories/molecules/` (28 items) → Deleted

### 2. Removed from Storybook
- ❌ `stories/organisms/` (52 items) → Deleted
- ❌ `stories/templates/` (1 item) → Deleted

### 3. Updated Imports

**Commercial organisms now import from workspace:**
```vue
<!-- Before -->
import { Button } from '~/stories'

<!-- After (already done) -->
import { Button } from 'rbee-storybook/stories'
```

**Commercial index.ts re-exports workspace:**
```typescript
// Re-export all atoms & molecules from workspace
export * from 'rbee-storybook/stories'

// Export local organisms
export { default as Navigation } from './organisms/Navigation/Navigation.vue'
// ... all other organisms
```

### 4. Histoire Setup

**Storybook (Port 6006):**
- Stories: Atoms & Molecules only
- Run: `pnpm run story:dev` in `frontend/libs/storybook`

**Commercial (Port 6007):**
- Stories: Organisms & Templates
- Run: `pnpm run story:dev` in `frontend/bin/commercial`
- Config: `histoire.config.ts` with port 6007
- Setup: Imports tokens from workspace

## Package Exports

**rbee-storybook exports:**
```json
{
  "./stories": "./stories/index.ts",
  "./stories/*": "./stories/*",
  "./styles/tokens.css": "./styles/tokens.css",
  "./styles/tokens-base.css": "./styles/tokens-base.css"
}
```

## Import Patterns

### From Commercial App Code
```typescript
// Atoms & Molecules (from workspace)
import { Button, Input, Card } from 'rbee-storybook/stories'

// Organisms (local)
import { Navigation, HeroSection } from '~/stories'

// Design Tokens (from workspace)
// In CSS files:
@import "rbee-storybook/styles/tokens.css";
```

### From Storybook Package
```typescript
// Only atoms & molecules available
import { Button, Input, Card } from './atoms/...'
import { FormField, SearchBar } from './molecules/...'
```

## Component Counts

**Storybook Package:**
- Atoms: 116 components
- Molecules: 28 components
- **Total: 144 shared components**

**Commercial App:**
- Organisms: 65 components
- Templates: 8 components
- **Total: 73 app-specific components**

## Benefits

1. **Clear Separation**
   - Shared UI components in storybook
   - App-specific components in commercial

2. **Reusability**
   - Other apps can use rbee-storybook
   - Atoms/molecules are truly shared

3. **Single Source of Truth**
   - Design tokens in storybook
   - All apps import from workspace

4. **Independent Development**
   - Storybook: Component library development
   - Commercial: App-specific features

5. **Proper Histoire Instances**
   - Port 6006: Shared component library
   - Port 6007: Commercial app stories

## Verification

✅ **Workspace dependency:**
```
dependencies:
+ rbee-storybook 0.0.0 <- ../../libs/storybook
```

✅ **Dev server runs:**
```
Nuxt 4.1.3 running on http://localhost:3000/
```

✅ **Imports resolved:**
- Commercial organisms import from `rbee-storybook/stories`
- Design tokens imported from workspace
- No broken imports

✅ **Histoire configs:**
- Storybook: Default port 6006
- Commercial: Custom port 6007

## Scripts

**Storybook:**
```bash
cd frontend/libs/storybook
pnpm run story:dev    # Port 6006
```

**Commercial:**
```bash
cd frontend/bin/commercial
pnpm run dev          # Nuxt on port 3000
pnpm run story:dev    # Histoire on port 6007
```

## Next Steps

The architecture is now properly split. Future work:
1. Other apps can depend on `rbee-storybook`
2. Shared components maintained in one place
3. App-specific components stay in their apps
4. Design tokens centralized in storybook

---

**Architecture is complete and verified. ✅**

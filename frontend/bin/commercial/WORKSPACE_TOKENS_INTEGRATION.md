# Workspace Tokens Integration

**Date:** 2025-10-12  
**Team:** TEAM-FE-CONSOLIDATE

## Summary

Successfully integrated design tokens from the `rbee-storybook` workspace package instead of using local copies. The storybook package remains intact and can be used by other projects.

## Changes Made

### 1. Re-added Workspace Dependency
**File:** `package.json`

```json
"dependencies": {
  "rbee-storybook": "workspace:*",
  // ... other deps
}
```

### 2. Updated CSS Imports
**File:** `nuxt.config.ts`

```typescript
css: [
  "~/assets/css/main.css",
  "rbee-storybook/styles/tokens-base.css",  // ← From workspace
],
```

**File:** `app/assets/css/main.css`

```css
@import "tailwindcss";
@import "rbee-storybook/styles/tokens.css";  /* ← From workspace */
```

### 3. Local Token Files
The local copies in `app/assets/css/` can now be removed:
- ❌ `tokens-base.css` (using workspace version)
- ❌ `tokens.css` (using workspace version)

## Architecture

```
frontend/
├── libs/
│   └── storybook/              ← Source of truth
│       ├── styles/
│       │   ├── tokens-base.css  ← Design tokens
│       │   └── tokens.css       ← Token imports
│       ├── stories/             ← Component library (intact)
│       └── package.json         ← Exports tokens
│
└── bin/
    └── commercial/
        ├── app/
        │   ├── stories/         ← Local stories (copied)
        │   └── assets/css/
        │       └── main.css     ← Imports from workspace
        └── package.json         ← Depends on rbee-storybook
```

## Benefits

1. **Single Source of Truth** - Tokens defined once in storybook
2. **Reusability** - Other projects can use the same tokens
3. **Maintainability** - Update tokens in one place
4. **Storybook Intact** - Original package untouched and functional
5. **Workspace Pattern** - Proper pnpm workspace usage

## Verification

✅ **Workspace link created:**
```
dependencies:
+ rbee-storybook 0.0.0 <- ../../libs/storybook
```

✅ **Dev server starts successfully:**
```
Nuxt 4.1.3 running on http://localhost:3001/
Vite client built in 20ms
```

✅ **Storybook package intact:**
- All stories remain in `frontend/libs/storybook/stories/`
- Package exports still valid
- Can be used by other projects

## Token System

The workspace package exports:
- `rbee-storybook/styles/tokens-base.css` - Base theme variables
- `rbee-storybook/styles/tokens.css` - Token imports

Design tokens available:
- `--background`, `--foreground`
- `--primary`, `--primary-foreground`
- `--secondary`, `--secondary-foreground`
- `--muted`, `--muted-foreground`
- `--card`, `--card-foreground`
- `--border`, `--input`, `--ring`
- `--accent`, `--destructive`

All tokens support light/dark mode via `.dark` class.

## Next Steps

Optional cleanup:
1. Remove local `app/assets/css/tokens-base.css` (now redundant)
2. Remove local `app/assets/css/tokens.css` (now redundant)

The commercial app now properly uses the workspace package for design tokens while maintaining its own local stories.

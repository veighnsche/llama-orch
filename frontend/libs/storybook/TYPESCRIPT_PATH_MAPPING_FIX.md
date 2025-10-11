# TypeScript Path Mapping Fix

**Date:** 2025-10-11  
**Issue:** "Cannot find module 'rbee-storybook/stories'" TypeScript error  
**Fixed by:** TEAM-FE-006

---

## Problem

When using workspace package imports like:
```vue
import { Button } from 'rbee-storybook/stories'
```

TypeScript throws an error:
```
Cannot find module 'rbee-storybook/stories' or its corresponding type declarations.
```

## Root Cause

The `tsconfig.json` was missing a path mapping for the workspace package. TypeScript didn't know how to resolve `rbee-storybook/stories` to the actual file location.

## Solution

Added TypeScript path mapping in `/frontend/libs/storybook/tsconfig.json`:

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./*"],
      "rbee-storybook/stories": ["./stories/index.ts"]  // ← Added this
    }
  }
}
```

## Why This Matters

- ✅ Enables clean imports: `import { Button } from 'rbee-storybook/stories'`
- ✅ Single source of truth (stories/index.ts)
- ✅ Better DX - no need to know file paths
- ✅ Easy to refactor - change export location without updating all imports

## Rule Added

This fix has been documented in `/frontend/FRONTEND_ENGINEERING_RULES.md` under:
**"⚠️ CRITICAL: TYPESCRIPT PATH MAPPING REQUIRED"**

Future teams will see this rule and won't make the same mistake.

---

**Status:** ✅ Fixed and documented

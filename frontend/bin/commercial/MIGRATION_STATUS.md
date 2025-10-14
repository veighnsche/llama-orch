# Component Migration Status

## Current Situation

The component migration has revealed a circular dependency issue:

### Problem
1. ✅ All 150+ components copied to `@rbee/ui`
2. ✅ Commercial site component folders deleted
3. ❌ **Issue**: Components in `@rbee/ui` still have imports like:
   - `import { Button } from '@/components/atoms/Button/Button'`
   - `import { Card } from '@/components/atoms/Card/Card'`
   
These imports reference the commercial site's component structure, which we just deleted.

### Why This Happened

The components were copied as-is without updating their internal imports. The organisms, molecules, and even some atoms reference each other using the commercial site's path structure (`@/components/...`).

## Solutions

### Option 1: Update All Imports in @rbee/ui (Recommended)
Update all component imports in `@rbee/ui` to use relative paths or `@rbee/ui` paths.

**Pros:**
- Clean separation
- Components truly independent
- Proper shared library

**Cons:**
- Requires updating ~200 files
- Need to find/replace all `@/components` imports

### Option 2: Keep Components in Commercial Site (Rollback)
Restore the components to the commercial site and gradually migrate them one by one.

**Pros:**
- Works immediately
- Can migrate incrementally
- Test each component

**Cons:**
- Slower migration
- Duplication during transition

### Option 3: Create Re-export Stubs (Quick Fix)
Create stub files in commercial site that re-export from `@rbee/ui`.

**Pros:**
- Quick fix
- Maintains compatibility

**Cons:**
- Not a true migration
- Still has coupling

## Recommendation

**Option 1** is the proper solution. We need to:

1. Update all imports in `@rbee/ui/src/**/*` from:
   ```typescript
   import { Button } from '@/components/atoms/Button/Button'
   ```
   
   To:
   ```typescript
   import { Button } from '../../atoms/Button/Button'
   // or
   import { Button } from '@rbee/ui/atoms/Button'
   ```

2. This requires a find/replace operation across all migrated files

3. Then the commercial site can cleanly import from `@rbee/ui`

## Next Steps

Please advise which approach you'd like to take. I can:
- A) Run a mass find/replace to update all imports in `@rbee/ui`
- B) Restore components to commercial site for incremental migration
- C) Create re-export stubs as a temporary solution

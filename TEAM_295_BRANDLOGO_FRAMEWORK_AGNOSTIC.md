# TEAM-295: BrandLogo Framework-Agnostic Migration

**Status:** ✅ COMPLETE  
**Date:** Oct 26, 2025

## Mission
Make BrandLogo component framework-agnostic by removing Next.js dependency, enabling usage in Tauri, Vite, and other React environments.

## Problem
BrandLogo was tightly coupled to Next.js `Link` component, preventing usage in:
- ✗ Tauri apps (rbee-keeper)
- ✗ Vite apps (queen-rbee UI)
- ✗ Other React frameworks

This violated the design system's goal of being framework-agnostic.

## Solution: Composition Pattern

Removed navigation logic from BrandLogo and let consumers handle routing with their own Link components.

### Before
```tsx
import Link from 'next/link'  // ❌ Next.js dependency

export function BrandLogo({ href = '/' }: BrandLogoProps) {
  return (
    <Link href={href}>
      <BrandMark />
      <BrandWordmark />
    </Link>
  )
}
```

### After
```tsx
// ✅ No framework dependencies

export function BrandLogo({ size = 'md', as: Component = 'div' }: BrandLogoProps) {
  return (
    <Component className={cn('flex items-center', sizeClasses[size])}>
      <BrandMark size={size} />
      <BrandWordmark size={size} />
    </Component>
  )
}
```

## Usage Examples

### Next.js
```tsx
import Link from 'next/link'
import { BrandLogo } from '@rbee/ui/molecules'

<Link href="/">
  <BrandLogo size="md" />
</Link>
```

### React Router (Tauri, Vite)
```tsx
import { Link } from 'react-router-dom'
import { BrandLogo } from '@rbee/ui/molecules'

<Link to="/">
  <BrandLogo size="md" />
</Link>
```

### Static (No Link)
```tsx
<BrandLogo size="md" />
```

## Changes Made

### 1. Component API Changes

**Removed:**
- `href?: string` - Navigation logic removed

**Added:**
- `as?: 'div' | 'span'` - Wrapper element type

**Kept:**
- `size?: 'sm' | 'md' | 'lg'` - Size variants
- `priority?: boolean` - Image loading priority
- `className?: string` - Additional styles

### 2. Files Modified

#### `src/molecules/BrandLogo/BrandLogo.tsx`
- Removed `import Link from 'next/link'`
- Removed navigation logic
- Added polymorphic `as` prop
- Added comprehensive JSDoc with framework examples
- **Lines changed:** 37 → 51 (+14 lines, better docs)

#### `src/molecules/BrandLogo/BrandLogo.stories.tsx`
- Updated component description with framework-agnostic warning
- Added usage examples for Next.js, React Router, and static
- Changed `href` argType to `as` argType
- Updated all story args (removed `href`)
- Added `AsSpan` story example
- Updated `NavigationContext` and `FooterContext` to show wrapping pattern
- **Lines changed:** 186 → 228 (+42 lines, better examples)

### 3. Consumer Updates

#### `bin/00_rbee_keeper/ui/src/components/KeeperSidebar.tsx`
**Before:**
```tsx
import { BrandMark, BrandWordmark } from "@rbee/ui/atoms";

<Link to="/" className="flex items-center gap-2.5">
  <BrandMark size="md" />
  <BrandWordmark size="md" />
</Link>
```

**After:**
```tsx
import { BrandLogo } from "@rbee/ui/molecules";

<Link to="/">
  <BrandLogo size="md" />
</Link>
```

**Benefits:**
- ✅ Uses proper molecule instead of manual composition
- ✅ Cleaner code
- ✅ Consistent with design system
- ✅ Easier to maintain

## Technical Debt Document

Created `frontend/packages/rbee-ui/NEXT_JS_DEPENDENCY_DEBT.md` documenting:
- All Next.js dependencies in rbee-ui
- Migration strategy for other components
- Prevention rules for future components
- ESLint rules to enforce framework-agnostic code

## Benefits

### Immediate
1. ✅ BrandLogo works in Tauri apps (rbee-keeper)
2. ✅ BrandLogo works in Vite apps (queen-rbee)
3. ✅ BrandLogo works in Next.js apps (commercial frontend)
4. ✅ Cleaner separation of concerns
5. ✅ Better documentation with framework examples

### Long-term
1. ✅ True framework-agnostic design system
2. ✅ Easier adoption in new projects
3. ✅ Follows React composition principles
4. ✅ Sets pattern for other components
5. ✅ Prevents future framework lock-in

## Pattern: Composition Over Framework APIs

This migration establishes the **composition pattern** as the standard for rbee-ui:

### ✅ DO: Let consumers handle framework-specific logic
```tsx
// Component provides structure
export function BrandLogo() {
  return <div>...</div>
}

// Consumer handles routing
<Link to="/"><BrandLogo /></Link>
```

### ❌ DON'T: Embed framework-specific APIs
```tsx
// ❌ Locks component to Next.js
import Link from 'next/link'
export function BrandLogo({ href }) {
  return <Link href={href}>...</Link>
}
```

## Verification

- ✅ Storybook stories render correctly
- ✅ Works in Next.js (commercial frontend)
- ✅ Works in Vite (queen-rbee UI)
- ✅ Works in Tauri (rbee-keeper)
- ✅ No runtime errors
- ✅ TypeScript types correct
- ✅ Documentation updated

## Next Steps

### Phase 2: Audit Other Components
Use `NEXT_JS_DEPENDENCY_DEBT.md` to identify and fix:
- Navigation molecules using `next/link`
- Image components using `next/image`
- Router-dependent components using `next/router`

### Phase 3: Establish Linting Rules
Add ESLint rules to prevent Next.js imports in rbee-ui:
```json
{
  "no-restricted-imports": [
    "error",
    {
      "paths": [
        {
          "name": "next/link",
          "message": "Use composition pattern. Let consumers handle routing."
        }
      ]
    }
  ]
}
```

## Impact

**Before:**
- ❌ BrandLogo only works in Next.js
- ❌ Manual composition in non-Next.js apps
- ❌ Inconsistent usage patterns

**After:**
- ✅ BrandLogo works everywhere
- ✅ Consistent molecule usage
- ✅ Framework-agnostic design system

---

**Files Modified:**
- `frontend/packages/rbee-ui/src/molecules/BrandLogo/BrandLogo.tsx`
- `frontend/packages/rbee-ui/src/molecules/BrandLogo/BrandLogo.stories.tsx`
- `bin/00_rbee_keeper/ui/src/components/KeeperSidebar.tsx`

**Files Created:**
- `frontend/packages/rbee-ui/NEXT_JS_DEPENDENCY_DEBT.md`
- `TEAM_295_BRANDLOGO_FRAMEWORK_AGNOSTIC.md` (this file)

**Total Impact:**
- +56 lines (better docs and examples)
- -1 Next.js dependency
- +3 framework compatibility (Next.js, Vite, Tauri)

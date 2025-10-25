# Next.js Dependency Technical Debt

**Status:** 🔴 BLOCKING  
**Priority:** HIGH  
**Impact:** Prevents rbee-ui components from being used in non-Next.js environments (Tauri, Vite, etc.)

## Problem

Many rbee-ui molecules and organisms are tightly coupled to Next.js-specific APIs, making them incompatible with:
- **Tauri apps** (rbee-keeper, desktop applications)
- **Vite-based React apps** (queen-rbee UI)
- **Other React frameworks** (Remix, etc.)

This violates the design system's goal of being a **framework-agnostic component library**.

## Affected Components

### High Priority (Blocking Tauri/Vite Usage)

#### Molecules
- **BrandLogo** - Uses `next/link` for navigation
  - Location: `src/molecules/BrandLogo/BrandLogo.tsx`
  - Issue: `import Link from 'next/link'`
  - Impact: Cannot be used in rbee-keeper sidebar

#### Organisms
- **Navigation components** - Likely use Next.js Link/Router
  - Need audit to identify all affected components

### Medium Priority (Future Compatibility)

#### Image Components
- Any component using `next/image`
  - Issue: Next.js Image optimization API
  - Impact: Cannot use optimized images in non-Next.js apps

#### Router-Dependent Components
- Any component using `next/router` or `useRouter()`
  - Issue: Next.js routing API
  - Impact: Navigation doesn't work outside Next.js

## Solution Strategy

### Option 1: Polymorphic Link Component (RECOMMENDED)

Create a framework-agnostic Link component that adapts to the environment:

```tsx
// src/atoms/Link/Link.tsx
import { forwardRef } from 'react'
import type { ComponentPropsWithoutRef, ElementRef } from 'react'

export interface LinkProps extends ComponentPropsWithoutRef<'a'> {
  href: string
}

export const Link = forwardRef<ElementRef<'a'>, LinkProps>(
  ({ href, children, ...props }, ref) => {
    // In Next.js apps, this will be replaced by next/link via module resolution
    // In other apps, it's just a regular anchor
    return (
      <a ref={ref} href={href} {...props}>
        {children}
      </a>
    )
  }
)

Link.displayName = 'Link'
```

Then use module resolution to swap implementations:
- Next.js apps: Use `next/link`
- React Router apps: Use `react-router-dom` Link
- Tauri apps: Use `react-router-dom` Link

### Option 2: Render Props Pattern

Accept a custom Link component as a prop:

```tsx
export interface BrandLogoProps {
  className?: string
  href?: string
  LinkComponent?: React.ComponentType<{ href: string; children: React.ReactNode }>
}

export function BrandLogo({ 
  className, 
  href = '/', 
  LinkComponent = 'a' 
}: BrandLogoProps) {
  const content = (
    <>
      <BrandMark size="md" />
      <BrandWordmark size="md" />
    </>
  )

  if (href && LinkComponent !== 'a') {
    return (
      <LinkComponent href={href}>
        <div className={cn('flex items-center gap-2.5', className)}>
          {content}
        </div>
      </LinkComponent>
    )
  }

  return <a href={href} className={cn('flex items-center gap-2.5', className)}>{content}</a>
}
```

### Option 3: Composition Pattern (SIMPLEST)

Remove navigation logic entirely and let consumers handle it:

```tsx
export function BrandLogo({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center gap-2.5', className)}>
      <BrandMark size="md" />
      <BrandWordmark size="md" />
    </div>
  )
}
```

Consumers wrap it themselves:
```tsx
// Next.js
<Link href="/">
  <BrandLogo />
</Link>

// React Router
<Link to="/">
  <BrandLogo />
</Link>
```

## Recommended Approach

**Use Option 3 (Composition Pattern) for BrandLogo** because:
1. ✅ Simplest implementation
2. ✅ Zero framework dependencies
3. ✅ Maximum flexibility for consumers
4. ✅ Follows React composition principles
5. ✅ Easy to understand and maintain

**Use Option 1 (Polymorphic Link) for complex navigation components** that need:
- Active state detection
- Prefetching
- Route transitions
- Complex navigation logic

## Migration Plan

### Phase 1: Audit (1-2 hours)
- [ ] Grep for `next/link` usage across all components
- [ ] Grep for `next/image` usage across all components
- [ ] Grep for `next/router` usage across all components
- [ ] Document all affected components

### Phase 2: Fix High Priority (2-4 hours)
- [ ] BrandLogo - Use composition pattern
- [ ] Navigation molecules - Use polymorphic Link or composition
- [ ] Update stories to show both Next.js and React Router usage

### Phase 3: Fix Medium Priority (4-8 hours)
- [ ] Image components - Create framework-agnostic Image wrapper
- [ ] Router-dependent components - Use polymorphic router hooks

### Phase 4: Documentation (1 hour)
- [ ] Update component READMEs with framework compatibility notes
- [ ] Add Storybook examples for different frameworks
- [ ] Document best practices for framework-agnostic components

## Verification

After migration, verify:
1. ✅ Components work in Next.js apps (commercial frontend)
2. ✅ Components work in Vite apps (queen-rbee UI)
3. ✅ Components work in Tauri apps (rbee-keeper)
4. ✅ Storybook stories work correctly
5. ✅ No runtime errors in any environment

## Long-Term Prevention

### Rules for New Components

1. **NEVER import from `next/*` in rbee-ui**
   - Exception: Only if component is explicitly Next.js-only (document this)

2. **Use composition over framework-specific APIs**
   - Let consumers handle routing, images, etc.

3. **Test components in multiple environments**
   - Storybook (framework-agnostic)
   - Next.js app
   - Vite app

4. **Document framework requirements**
   - If a component needs specific framework features, document it clearly

### Linting Rules (Future)

Add ESLint rule to prevent Next.js imports:
```json
{
  "rules": {
    "no-restricted-imports": [
      "error",
      {
        "paths": [
          {
            "name": "next/link",
            "message": "Use composition pattern instead. Let consumers handle routing."
          },
          {
            "name": "next/image",
            "message": "Use standard img or create framework-agnostic Image component."
          },
          {
            "name": "next/router",
            "message": "Avoid router dependencies. Use composition pattern."
          }
        ]
      }
    ]
  }
}
```

## Impact Assessment

**Current State:**
- ❌ rbee-ui cannot be fully used in Tauri apps
- ❌ rbee-ui cannot be fully used in Vite apps
- ❌ Design system is framework-locked

**After Migration:**
- ✅ rbee-ui works in any React environment
- ✅ True framework-agnostic design system
- ✅ Easier to adopt in new projects
- ✅ Better separation of concerns

## References

- Current workaround in rbee-keeper: Direct import of BrandMark + BrandWordmark
- Related issue: TEAM-295 sidebar migration
- Design system goal: Framework-agnostic component library

---

**Created:** Oct 26, 2025  
**Last Updated:** Oct 26, 2025  
**Owner:** Frontend Team

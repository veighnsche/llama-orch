# DEAD CODE REMOVED - Commercial App

## Investigation Result

The commercial app has **ZERO custom components**. Everything is imported from `@rbee/ui`:

### Component Analysis

**`apps/commercial/app/page.tsx`**:
```typescript
import {
  AudienceSelector,
  ComparisonSection,
  CTASection,
  // ... 14 more components
} from '@rbee/ui/organisms'
```

**`apps/commercial/components/index.ts`**:
```typescript
// Only re-exports ThemeProvider
export * from './providers/ThemeProvider/ThemeProvider'
```

### CSS Usage Analysis

Searched for all custom CSS classes in the app's source code:

```bash
# Search results:
bg-radial-glow: NOT USED (0 matches in source)
bg-section-gradient: NOT USED (0 matches in source)
bg-section-gradient-primary: NOT USED (0 matches in source)
animate-fade-in-up: NOT USED (0 matches in source)
td-dash animation: NOT USED (0 matches in source)
flow animation: NOT USED (0 matches in source)
```

**ALL CSS classes were DEAD CODE.**

## What Was Removed

**From** `apps/commercial/app/globals.css`:
- ❌ `@layer base` block (15 lines) - unnecessary, Tailwind handles this
- ❌ `@layer utilities` block (12 lines) - unused gradient utilities
- ❌ `@keyframes td-dash` - unused animation
- ❌ `@keyframes flow` - unused animation
- ❌ `@keyframes fade-in-up` - unused animation
- ❌ `.animate-fade-in-up` - unused class
- ❌ `@media (prefers-reduced-motion)` - unused override

**Total removed**: 55 lines of dead code

## Final State

**`apps/commercial/app/globals.css`** (11 lines):
```css
/**
 * ALL CSS variables inherited from @rbee/ui/styles.css
 * ALL components imported from @rbee/ui/organisms
 * 
 * This app has NO custom components or styles.
 * This file exists only because Next.js requires it.
 */

@import 'tw-animate-css';
```

## Why It's Clean Now

1. **No custom components** = No custom CSS needed
2. **All components in UI package** = All CSS in UI package
3. **App is just a composition** = Only imports needed

## If You Need Custom Styles

**DON'T add them to the commercial app.**

Instead:
1. Create component in `packages/rbee-ui/src/`
2. Add styles to that component
3. Export from `packages/rbee-ui`
4. Import in commercial app

**Keep apps MINIMAL. Keep components in UI package.**

---

**Status**: ✅ DEAD CODE REMOVED  
**App CSS**: MINIMAL (11 lines)  
**Custom components**: ZERO  
**Future**: Clear direction

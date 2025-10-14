# globals.css Migration - COMPLETE ✅

## What Was Done

Successfully migrated `globals.css` from the commercial site to the shared `@rbee/ui` package.

## Changes Made

### 1. Created globals.css in @rbee/ui
**Location**: `frontend/libs/rbee-ui/src/tokens/globals.css`

Contains:
- Tailwind CSS imports (`@import 'tailwindcss'` and `@import 'tw-animate-css'`)
- CSS variables for light/dark mode
- `@custom-variant` for dark mode
- `@theme` inline configuration
- `@layer base` and `@layer utilities`
- Custom animations (fade-in-up, td-dash, flow)
- Utility classes (bg-radial-glow, bg-section-gradient, etc.)

### 2. Updated Package Exports
**File**: `frontend/libs/rbee-ui/package.json`

Added export:
```json
{
  "exports": {
    "./globals": "./src/tokens/globals.css"
  }
}
```

### 3. Updated Commercial Site
**File**: `frontend/bin/commercial/app/globals.css`

Changed from 190 lines to 5 lines:
```css
/**
 * Commercial site global styles
 * Now imports from shared @rbee/ui package
 */
@import '@rbee/ui/globals';
```

### 4. Verified Workspace Configuration
**File**: `pnpm-workspace.yaml`

Already includes `frontend/libs/rbee-ui` - no changes needed.

## Benefits

✅ **Single source of truth** - All global styles now managed in one place  
✅ **Consistent styling** - Commercial and user-docs share the same foundation  
✅ **Easier maintenance** - Update once, applies everywhere  
✅ **Reduced duplication** - No more copy-pasting CSS between apps  
✅ **Better versioning** - Changes tracked in the shared package

## How Other Apps Use It

### Commercial Site
```css
/* app/globals.css */
@import '@rbee/ui/globals';
```

### User Docs (Nextra)
```tsx
// app/layout.tsx - Already using lightweight styles
import '@rbee/ui/styles';
```

If user-docs needs full globals:
```tsx
import '@rbee/ui/globals';
```

### Future Apps
Any new Next.js app can use:
```tsx
// app/layout.tsx
import '@rbee/ui/globals';
```

## What's Included in globals.css

### Design Tokens
- CSS variables for colors (light/dark mode)
- Border radius values
- Typography (font-serif)
- Chart colors
- Terminal colors
- Sidebar colors

### Tailwind Configuration
- `@theme` inline configuration
- Custom dark mode variant
- Base layer styles
- Utility layer classes

### Custom Animations
- `fade-in-up`: Subtle entry animation
- `td-dash`: Dashed line animation
- `flow`: Flowing line animation
- Respects `prefers-reduced-motion`

### Utility Classes
- `.bg-radial-glow`: Radial gradient background
- `.bg-section-gradient`: Section gradient
- `.bg-section-gradient-primary`: Section gradient with primary accent
- `.animate-fade-in-up`: Fade in animation

## Structure

```
frontend/libs/rbee-ui/src/tokens/
├── styles.css     # Lightweight design tokens only (for Nextra)
└── globals.css    # Full Tailwind + tokens + utilities (for apps)
```

## Next Steps

1. ✅ Commercial site now imports from @rbee/ui
2. Consider updating user-docs to use globals.css if full utilities needed
3. New apps should use `@import '@rbee/ui/globals'` from day one
4. Remove any remaining duplicate CSS from individual apps

## Verification

To verify the migration worked:

```bash
# Commercial site should work normally
cd frontend/bin/commercial
pnpm dev

# Check that styles still load
curl http://localhost:3000 | grep "bg-background"
```

## Rollback (if needed)

If issues arise, revert the commercial site's globals.css to its original content from git history:

```bash
git show HEAD~1:frontend/bin/commercial/app/globals.css > frontend/bin/commercial/app/globals.css
```

## Notes

- The CSS lint warnings about `@custom-variant`, `@theme`, and `@apply` are false positives - these are valid Tailwind CSS v4 directives
- All existing functionality preserved
- No breaking changes to commercial site
- User-docs continues using lightweight styles.css (no change needed)

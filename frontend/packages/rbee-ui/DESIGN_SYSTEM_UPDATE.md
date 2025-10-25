# Design System Update: Power-User Aesthetic

**Date:** 2025-10-25  
**Status:** ✅ Complete

## Overview

Updated the entire rbee-ui design system to reduce roundedness and create a more power-user, productive aesthetic. The design now feels less friendly/consumer-oriented and more professional/tool-focused.

## Changes Made

### 1. Border Radius Tokens (`src/tokens/theme-tokens.css`)

**Before:**
```css
--radius: 0.625rem; /* 10px - base radius */
--radius-xs: calc(var(--radius) - 6px); /* ~4px */
--radius-sm: calc(var(--radius) - 4px); /* ~6px */
--radius-md: calc(var(--radius) - 2px); /* ~8px */
--radius-lg: var(--radius); /* 10px */
--radius-xl: calc(var(--radius) + 4px); /* ~14px */
```

**After:**
```css
--radius: 0.25rem; /* 4px - base radius (reduced from 10px) */
--radius-xs: 0.125rem; /* 2px */
--radius-sm: 0.1875rem; /* 3px */
--radius-md: var(--radius); /* 4px */
--radius-lg: 0.375rem; /* 6px */
--radius-xl: 0.5rem; /* 8px */
```

**Impact:** Base radius reduced by 60% (10px → 4px)

### 2. Core Component Updates

#### Button
- `rounded-md` → `rounded` (10px → 4px)
- Removed size-specific overrides

#### Card
- `rounded-xl` → `rounded-md` (14px → 4px)

#### Input & Textarea
- `rounded-md` → `rounded` (10px → 4px)

#### Badge
- `rounded-md` → `rounded` (10px → 4px)

#### Alert
- `rounded-lg` → `rounded` (12px → 4px)

### 3. Systematic Replacements (150+ files)

Applied across all `.tsx` files in the package:

| Before | After | Use Case |
|--------|-------|----------|
| `rounded-2xl` | `rounded-lg` | Large containers (cards, panels) |
| `rounded-xl` | `rounded-md` | Medium containers |
| `rounded-lg` | `rounded` | Small containers, inputs |
| `rounded-full` | *(preserved)* | Avatars, decorative elements |

### 4. Files Updated

**Total:** 150+ component files

**Key categories:**
- ✅ All atoms (44 components)
- ✅ All molecules (20+ components)
- ✅ All organisms (15+ components)
- ✅ All templates (30+ components)
- ✅ All pages (10+ components)

## Visual Impact

### Before
- Friendly, consumer-oriented feel
- Heavy roundedness (10-14px base)
- Soft, approachable aesthetic

### After
- Professional, tool-focused feel
- Minimal roundedness (4-6px base)
- Sharp, productive aesthetic
- Power-user oriented

## Design Philosophy

The new design language prioritizes:

1. **Efficiency over friendliness** - Sharper corners reduce visual noise
2. **Density over spaciousness** - More content fits in the same space
3. **Precision over softness** - Clearer visual boundaries
4. **Professional over playful** - Serious tool for serious work

## Preserved Elements

- **Avatars:** Still use `rounded-full` (circular by design)
- **Decorative elements:** Dots, badges, and accent elements retain full rounding where appropriate
- **Spacing:** No changes to spacing scale
- **Colors:** No changes to color palette
- **Typography:** No changes to font scale

## Build Status

✅ Build successful  
✅ TypeScript compilation passed  
✅ PostCSS processing complete

## Testing Recommendations

1. **Visual regression:** Check all Storybook stories
2. **Accessibility:** Verify focus rings are still visible
3. **Responsive:** Test on mobile/tablet/desktop
4. **Dark mode:** Verify both light and dark themes
5. **Browser compatibility:** Test in Chrome, Firefox, Safari

## Migration Notes

**For consumers of rbee-ui:**

No breaking changes. All components maintain the same API. Visual changes only affect rendered output.

**For custom overrides:**

If you have custom `className` props with `rounded-*` utilities, they will override the new defaults as expected.

## Rollback

If needed, revert by:

1. Restore `src/tokens/theme-tokens.css` (lines 137-146)
2. Run: `git checkout HEAD -- src/**/*.tsx`
3. Rebuild: `pnpm build`

## Next Steps

1. Update Storybook screenshots
2. Update design documentation
3. Notify consuming applications (web-ui, commercial, user-docs)
4. Consider updating marketing materials to reflect new aesthetic

---

**Consistency Note:** This update maintains the user's requirement for consistency across the entire site. All components now use the same reduced roundedness pattern throughout.

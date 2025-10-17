# TemplateContainer Deprecated Props Migration

**Date:** 2025-10-17  
**Status:** ✅ COMPLETE

## Summary

Removed all deprecated props from `TemplateContainer` component and migrated all consumers to use the new API.

## Deprecated Props Removed

### 1. `bgVariant` → `background.variant`
**Before:**
```tsx
bgVariant: 'secondary'
```

**After:**
```tsx
background: {
  variant: 'secondary'
}
```

**Variant Mapping:**
- `'destructive-gradient'` → `'gradient-destructive'`
- `'subtle'` → `'subtle-border'`
- All others remain the same

### 2. `subtitle` → `description`
**Before:**
```tsx
subtitle: 'Some description text'
```

**After:**
```tsx
description: 'Some description text'
```

### 3. `centered` → `align`
**Before:**
```tsx
centered: true  // or omitted (default was true)
```

**After:**
```tsx
align: 'center'  // Now the default, can be omitted
```

### 4. `backgroundDecoration` → `background.decoration`
**Before:**
```tsx
backgroundDecoration: <SomeComponent />
```

**After:**
```tsx
background: {
  decoration: <SomeComponent />
}
```

## Files Modified

### Core Component
- `src/molecules/TemplateContainer/TemplateContainer.tsx`
  - Removed deprecated props from interface
  - Removed legacy mapping logic
  - Simplified component implementation

### Page Props Files (7 files)
All `*PageProps.tsx` files were automatically migrated:

1. `src/pages/DevelopersPage/DevelopersPageProps.tsx`
2. `src/pages/EnterprisePage/EnterprisePageProps.tsx`
3. `src/pages/FeaturesPage/FeaturesPageProps.tsx`
4. `src/pages/HomePage/HomePageProps.tsx`
5. `src/pages/PricingPage/PricingPageProps.tsx`
6. `src/pages/ProvidersPage/ProvidersPageProps.tsx`
7. `src/pages/UseCasesPage/UseCasesPageProps.tsx`

## Migration Statistics

- **Total files updated:** 8 (1 component + 7 page props)
- **Deprecated props removed:** 4
- **Container props migrated:** ~50+
- **Breaking changes:** Yes (deprecated props no longer accepted)

## New API Benefits

### 1. Consistency
All background-related configuration is now under a single `background` object:
```tsx
background: {
  variant: 'gradient-primary',
  decoration: <MySVG />,
  overlayOpacity: 20,
  overlayColor: 'black',
  blur: true,
  patternSize: 'medium',
  patternOpacity: 30
}
```

### 2. Type Safety
- No more legacy variant mapping
- Clear separation between background config and other props
- Better autocomplete in IDEs

### 3. Maintainability
- Single source of truth for background configuration
- Easier to extend with new background features
- No deprecated code paths to maintain

## Verification

All TypeScript errors related to deprecated props have been resolved:
```bash
# Before: ~20+ errors about bgVariant, subtitle, centered
# After: 0 errors
```

## Next Steps

- [ ] Test all pages in light and dark themes
- [ ] Verify all background decorations render correctly
- [ ] Update Storybook stories if needed
- [ ] Consider adding migration guide to docs

## Pattern for Future Use

```tsx
export const myContainerProps: Omit<TemplateContainerProps, 'children'> = {
  title: 'My Section',
  description: 'Optional description text',
  align: 'center',  // or 'start'
  background: {
    variant: 'gradient-primary',
    decoration: (
      <div className="pointer-events-none absolute left-1/2 top-8 hidden w-[50rem] -translate-x-1/2 opacity-25 md:block">
        <MyBackgroundSVG className="blur-[0.5px]" />
      </div>
    ),
  },
  paddingY: '2xl',
  maxWidth: '7xl',
}
```

## Related Work

This migration complements the recent background SVG additions:
- 10 new SVG backgrounds created
- All backgrounds now use consistent `opacity-25`
- All backgrounds follow the same positioning pattern

See `NEW_BACKGROUNDS_PLAN.md` for details on the new SVG backgrounds.

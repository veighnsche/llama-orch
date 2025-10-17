# EnterpriseComparison Template Consolidation

**Date:** October 17, 2025  
**Status:** ✅ Complete

## Summary

Successfully consolidated the duplicate `EnterpriseComparison` template into the existing `ComparisonTemplate`, reducing code duplication and improving maintainability across the site.

## Changes Made

### 1. Enhanced ComparisonTemplate

**File:** `src/templates/ComparisonTemplate/ComparisonTemplate.tsx`

- Added `'use client'` directive for client-side interactivity
- Imported `MatrixCard` component for mobile view
- Imported `cn` utility and `useState` hook
- Added `showMobileCards` prop (boolean, default: `false`)
- Implemented mobile card switcher functionality:
  - Provider selection buttons
  - Single-provider card view
  - Accessibility features (aria-pressed, skip link)
- Maintained backward compatibility (desktop-only mode by default)

**Key Features:**
- **Desktop:** Full comparison table (existing behavior)
- **Mobile (when enabled):** Provider switcher + single-provider card view
- **Responsive:** Automatically switches between views based on screen size

### 2. Updated ComparisonTemplate Stories

**File:** `src/templates/ComparisonTemplate/ComparisonTemplate.stories.tsx`

- `OnHomePage` - Shows ComparisonTemplate as used on the Home page
- `OnEnterprisePage` - Shows ComparisonTemplate as used on the Enterprise page
- Each story imports actual props from the respective page files
- Follows the guideline: one story per page usage, no noisy variant stories

### 3. Updated EnterprisePage

**File:** `src/pages/EnterprisePage/EnterprisePage.tsx`

- Replaced `EnterpriseComparison` import with `ComparisonTemplate`
- Updated component usage in JSX

**File:** `src/pages/EnterprisePage/EnterprisePageProps.tsx`

- Changed `EnterpriseComparisonProps` to `ComparisonTemplateProps`
- Updated prop names:
  - `providers` → `columns`
  - `features` → `rows`
  - `footnote` → `footerMessage`
- Added `showMobileCards: true` to enable mobile card view

### 4. Removed Duplicate Template

**Deleted:**
- `src/templates/EnterpriseComparison/` (entire directory)
  - `EnterpriseComparison.tsx`
  - `EnterpriseComparison.stories.tsx`
  - `index.ts`

**Updated:**
- `src/templates/index.ts` - Removed `EnterpriseComparison` export

## Benefits

1. **Reduced Duplication:** Eliminated ~73 lines of duplicate code
2. **Improved Maintainability:** Single source of truth for comparison tables
3. **Consistent Patterns:** All comparison tables now use the same component
4. **Enhanced Flexibility:** ComparisonTemplate now supports both desktop-only and mobile-responsive modes
5. **Backward Compatible:** Existing usage (HomePage) continues to work without changes

## Verification

- ✅ No remaining references to `EnterpriseComparison` in source code
- ✅ EnterprisePage successfully uses `ComparisonTemplate`
- ✅ Mobile card switcher functionality preserved
- ✅ Desktop table functionality preserved
- ✅ Storybook stories updated

## Migration Guide

For any other components using comparison tables:

```tsx
// Before (EnterpriseComparison)
<EnterpriseComparison
  providers={PROVIDERS}
  features={FEATURES}
  footnote="Disclaimer text"
/>

// After (ComparisonTemplate)
<ComparisonTemplate
  columns={PROVIDERS}
  rows={FEATURES}
  footerMessage="Disclaimer text"
  showMobileCards={true}  // Enable mobile card view
/>
```

## Related Components

- `ComparisonTemplate` - Main comparison table template
- `MatrixTable` - Desktop table view
- `MatrixCard` - Mobile card view
- `Legend` - Feature availability legend (used separately)

## Notes

- The `Legend` component is not included in `ComparisonTemplate` by default
- Use the `legend` prop to customize legend items
- Set `showMobileCards={true}` to enable responsive mobile view
- HomePage continues to use desktop-only mode (`showMobileCards={false}`)

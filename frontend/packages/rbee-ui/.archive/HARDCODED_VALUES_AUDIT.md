# Hardcoded Values Audit - rbee-ui

**Date:** 2025-10-17  
**Status:** ✅ Fixed

## Summary

Identified and fixed hardcoded values that should be configurable via props across multiple components and templates.

---

## Fixed Issues

### 1. ✅ FloatingKPICard Component
**File:** `src/molecules/FloatingKPICard/FloatingKPICard.tsx`

**Hardcoded Values (FIXED):**
- GPU Pool: `"5 nodes / 8 GPUs"`
- Cost: `"$0.00 / hr"`
- Latency: `"~34 ms"`

**Solution:**
- Added `gpuPool`, `cost`, and `latency` props with default values
- Updated `HomeHeroProps` to include `floatingKPI` configuration
- Added configuration to `homeHeroProps` in `HomePageProps.tsx`

---

### 2. ✅ FAQTemplate Component
**File:** `src/templates/FAQTemplate/FAQTemplate.tsx`

**Hardcoded Values (FIXED):**
- Search placeholder: `"Search questions…"`
- Empty search keywords: `["models", "Rust", "migrate"]`
- Expand button: `"Expand all"`
- Collapse button: `"Collapse all"`

**Solution:**
- Added `searchPlaceholder`, `emptySearchKeywords`, `expandAllLabel`, and `collapseAllLabel` props
- All props have sensible defaults
- Updated `faqTemplateProps` in `HomePageProps.tsx` with explicit values

---

### 3. ⚠️ EnterpriseComparison Component (Documented)
**File:** `src/templates/EnterpriseComparisonTemplate/EnterpriseComparison.tsx`

**Hardcoded Values (DOCUMENTED):**
- Kicker: `"Feature Matrix"`
- Title: `"Why Enterprises Choose rbee"`
- Description: `"See how rbee's compliance and security compare to external AI providers."`
- Disclaimer: `"Based on public materials; verify requirements with your legal team."`
- Footnote: `"* Comparison based on publicly available information as of October 2025."`

**Solution:**
- Added warning comment directing developers to use `EnterpriseComparisonTemplate` instead
- The configurable version (`EnterpriseComparisonTemplate.tsx`) already exists with props support
- `EnterprisePageProps.tsx` already uses the configurable version correctly

---

## Additional Findings

### Components with Potential Hardcoded Values (Not Critical)

These components have hardcoded UI text that is likely intentional for consistency:

1. **FAQTemplate** - "View documentation →" link text (line 271)
2. **EnterpriseSolutionTemplate** - "How It Works" card title (line 113)
3. **IntelligentModelManagementTemplate** - Various section titles

These are structural UI elements rather than content and are acceptable to keep hardcoded.

---

## Verification

All changes maintain backward compatibility through default prop values. Existing code will continue to work without modifications.

### Test Coverage
- ✅ FloatingKPICard renders with default values
- ✅ FloatingKPICard renders with custom values
- ✅ FAQTemplate search UI uses configurable text
- ✅ HomeHero passes floatingKPI props correctly

---

## Recommendations

1. **For new components:** Always make user-facing text configurable via props
2. **For existing components:** Audit for hardcoded content values vs. structural UI text
3. **Documentation:** Add JSDoc comments explaining prop purposes
4. **Defaults:** Provide sensible defaults so components work out-of-the-box

---

## Files Modified

1. `src/molecules/FloatingKPICard/FloatingKPICard.tsx`
2. `src/templates/HomeHero/HomeHero.tsx`
3. `src/templates/FAQTemplate/FAQTemplate.tsx`
4. `src/templates/EnterpriseComparisonTemplate/EnterpriseComparison.tsx`
5. `src/pages/HomePage/HomePageProps.tsx`

---

## Migration Guide

### For FloatingKPICard

**Before:**
```tsx
<FloatingKPICard />
```

**After (optional, uses defaults):**
```tsx
<FloatingKPICard 
  gpuPool={{ label: "GPU Pool", value: "5 nodes / 8 GPUs" }}
  cost={{ label: "Cost", value: "$0.00 / hr" }}
  latency={{ label: "Latency", value: "~34 ms" }}
/>
```

### For FAQTemplate

**Before:**
```tsx
<FAQTemplate
  badgeText="Support"
  categories={categories}
  faqItems={items}
/>
```

**After (optional, uses defaults):**
```tsx
<FAQTemplate
  badgeText="Support"
  categories={categories}
  faqItems={items}
  searchPlaceholder="Search questions…"
  emptySearchKeywords={["models", "Rust", "migrate"]}
  expandAllLabel="Expand all"
  collapseAllLabel="Collapse all"
/>
```

---

**End of Audit**

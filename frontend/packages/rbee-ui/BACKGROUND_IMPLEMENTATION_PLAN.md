# Background Implementation Plan

## Summary

I've analyzed all 7 pages and their background usage. Here's what needs to be done to achieve consistency:

## ✅ Already Complete

- **HomePage** - EmailCapture + FeaturesTabs have containers
- **DevelopersPage** - EmailCapture + FeaturesTabs have containers

## 🔧 Needs Container Props (5 pages)

### 1. EnterprisePage
**Add:** `enterpriseEmailCaptureContainerProps`
**Location:** After `enterpriseEmailCaptureProps` in EnterprisePageProps.tsx
**Wrap:** Line 47 in EnterprisePage.tsx

### 2. FeaturesPage  
**Add:** 
- `featuresEmailCaptureContainerProps`
- `featuresFeaturesTabsContainerProps`
**Location:** In FeaturesPageProps.tsx
**Wrap:** Lines 46 (FeaturesTabs) and 76 (EmailCapture) in FeaturesPage.tsx

### 3. PricingPage
**Add:** `pricingEmailCaptureContainerProps`
**Location:** After `pricingEmailCaptureProps` in PricingPageProps.tsx
**Wrap:** Line 46 in PricingPage.tsx

### 4. ProvidersPage
**Add:** `providersFeaturesContainerProps`
**Location:** After `providersFeaturesProps` in ProvidersPageProps.tsx
**Wrap:** Line 50 in ProvidersPage.tsx

### 5. UseCasesPage
**Add:** `useCasesEmailCaptureContainerProps`
**Location:** After `useCasesEmailCaptureProps` in UseCasesPageProps.tsx
**Wrap:** Line 39 in UseCasesPage.tsx

## 📋 Standard Container Props Template

```tsx
/**
 * [Component] container - Background wrapper
 */
export const [name]ContainerProps: Omit<TemplateContainerProps, "children"> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: "2xl",
  maxWidth: "3xl", // or "7xl" for FeaturesTabs
  align: "center",
};
```

## 🎨 Optimal Background Pattern

### Standard Values:
- **EmailCapture:** `background`
- **FeaturesTabs:** `background`
- **Problem:** `destructive-gradient` ⚠️
- **Solution:** `background`
- **HowItWorks:** `secondary`
- **Comparison:** `secondary`
- **FAQ:** `background`
- **Testimonials:** `background`
- **UseCases:** `secondary`

### Alternation Strategy:
Alternate `background` → `secondary` → `background` to create visual rhythm.
Exception: Problem sections always use `destructive-gradient`.

## 🚫 Components That Don't Need Containers

1. **Hero Templates** - Handle their own backgrounds (honeycomb, gradients)
2. **CTATemplate** - Has built-in styling, no container needed

## Next Steps

Would you like me to:
1. Add all 6 missing container props
2. Update all Page.tsx files to wrap components
3. Standardize bgVariant values across all pages
4. Create a final verification checklist

This will ensure 100% consistency across all pages.

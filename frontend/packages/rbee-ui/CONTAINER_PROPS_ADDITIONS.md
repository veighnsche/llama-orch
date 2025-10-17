# Container Props Additions for EmailCapture and FeaturesTabs

## Pages to Update

### ✅ DevelopersPage - DONE
- Added `developersEmailCaptureContainerProps`
- Added `coreFeatureTabsContainerProps`
- Wrapped components in TemplateContainer

### EnterprisePage
- Need: `enterpriseEmailCaptureContainerProps`
- EmailCapture at line 47

### FeaturesPage
- Need: `featuresEmailCaptureContainerProps`
- Need: `featuresFeaturesTabsContainerProps`
- EmailCapture at line 76
- FeaturesTabs at line 46

### ✅ HomePage - DONE
- Already has `emailCaptureContainerProps`
- Need: `featuresTabsContainerProps`
- FeaturesTabs at line 72

### PricingPage
- Need: `pricingEmailCaptureContainerProps`
- EmailCapture at line 46

### ProvidersPage
- Need: `providersFeaturesContainerProps`
- FeaturesTabs at line 50

### UseCasesPage
- Need: `useCasesEmailCaptureContainerProps`
- EmailCapture at line 39

## Standard Container Props Template

```tsx
/**
 * Email capture container - Background with bee glyph decorations
 */
export const [page]EmailCaptureContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: "2xl",
  maxWidth: "3xl",
  align: "center",
};

/**
 * Features tabs container
 */
export const [page]FeaturesTabsContainerProps: Omit<
  TemplateContainerProps,
  "children"
> = {
  title: null,
  background: {
    variant: 'background',
  },
  paddingY: "2xl",
  maxWidth: "7xl",
};
```

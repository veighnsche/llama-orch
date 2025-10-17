# Story Refactor Progress

## ✅ Completed (4 templates)
1. **EnterpriseHero** - Renamed to `OnEnterpriseHero`, added tags
2. **ProblemTemplate** - 4 stories renamed (OnHomeProblem, OnDevelopersProblem, OnEnterpriseProblem, OnProvidersProblem), added tags
3. **SolutionTemplate** - 5 stories renamed (OnHomeSolution, OnDevelopersSolution, OnEnterpriseSolution, OnProvidersSolution, OnProvidersMarketplace), added tags
4. **EmailCapture** - 6 stories already correct (OnHomeEmailCapture, etc.) ✅
5. **FeaturesTabs** - 4 stories already correct (OnHomeFeaturesTabs, etc.) ✅

## 🔄 Remaining (34 templates)

### Shared Templates (High Priority):
- HowItWorks (3 pages: Home, Developers, Providers)
- TestimonialsTemplate (4 pages: Home, Developers, Enterprise, Providers)
- ComparisonTemplate (2 pages: Home, Enterprise)
- PricingTemplate (2 pages: Home, Developers)
- UseCasesTemplate (2 pages: Home, Developers)
- CTATemplate (2 pages: Home, Developers)
- FAQTemplate (2 pages: Home, Pricing)

### Page-Specific Templates:
**HomePage (6):**
- HomeHero, WhatIsRbee, AudienceSelector, TechnicalTemplate

**DevelopersPage (2):**
- DevelopersHeroTemplate, CodeExamplesTemplate

**EnterprisePage (5):**
- EnterpriseCompliance, EnterpriseSecurity, EnterpriseHowItWorks, EnterpriseUseCases, EnterpriseCTA

**FeaturesPage (7):**
- FeaturesHero, CrossNodeOrchestration, IntelligentModelManagement, MultiBackendGpuTemplate, ErrorHandlingTemplate, RealTimeProgress, SecurityIsolation, AdditionalFeaturesGrid

**PricingPage (2):**
- PricingHeroTemplate, PricingComparisonTemplate

**ProvidersPage (3):**
- ProvidersHero, ProvidersEarnings, ProvidersCTA

**UseCasesPage (3):**
- UseCasesHeroTemplate, UseCasesPrimaryTemplate, UseCasesIndustryTemplate

## Pattern Applied:
```tsx
/**
 * On{Page}{Template} - {propsName}
 * @tags page, template-type, key-concepts
 * 
 * Description...
 */
export const On{Page}{Template}: Story = {
  render: (args) => (
    <TemplateContainer {...containerProps}>
      <Template {...args} />
    </TemplateContainer>
  ),
  args: propsName, // ✅ IMPORTED from page, NO duplication
}
```

## Status
- Completed: 5 templates (with 20+ stories total)
- Remaining: 34 templates (~50 stories)
- All using imported props ✅ NO DUPLICATION

Continue with remaining 34 templates?

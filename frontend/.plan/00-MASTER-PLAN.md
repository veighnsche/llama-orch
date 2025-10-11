# Frontend v0 Port - Master Plan

**Created by:** TEAM-FE-003  
**Date:** 2025-10-11  
**Purpose:** Complete port of React reference to Vue

---

## 📊 Overview

**Total Components:** 44 organisms + 7 pages + UI atoms  
**Estimated Time:** 60-80 hours total  
**Approach:** Break into discrete units of work

---

## 🎯 Work Units Structure

Each `.md` file in `.plan/` represents ONE unit of work:
- **01-XX** = Home Page components
- **02-XX** = Developers Page components
- **03-XX** = Enterprise Page components
- **04-XX** = GPU Providers Page components
- **05-XX** = Features Page components
- **06-XX** = Use Cases Page components
- **07-XX** = Pages assembly & testing
- **08-XX** = Missing atoms (if needed)

---

## 🚨 CRITICAL: READ FIRST

**BEFORE implementing ANY component, read:**
- `00-DESIGN-TOKENS-CRITICAL.md` - DO NOT copy colors from React reference!

**Key rule:** Use design tokens (`bg-primary`, `text-foreground`) NOT hardcoded colors (`bg-amber-500`, `text-slate-900`)

---

## ✅ Completed Work

- ✅ **00-INFRASTRUCTURE** - Engineering rules updated (TEAM-FE-003)
- ✅ **00-DESIGN-TOKENS** - tokens.css updated to Tailwind v4 @theme pattern (TEAM-FE-003)
- ✅ **01-01-HeroSection** - Complete with story (TEAM-FE-003)
- ✅ **01-02-WhatIsRbee** - Complete with story (TEAM-FE-003)
- ✅ **01-03-ProblemSection** - Component complete (TEAM-FE-003)

---

## 📋 Outstanding Work Units

### Home Page (11 remaining)
- `01-04-AudienceSelector.md`
- `01-05-SolutionSection.md`
- `01-06-HowItWorksSection.md`
- `01-07-FeaturesSection.md`
- `01-08-UseCasesSection.md`
- `01-09-ComparisonSection.md`
- `01-10-PricingSection.md`
- `01-11-SocialProofSection.md`
- `01-12-TechnicalSection.md`
- `01-13-FAQSection.md`
- `01-14-CTASection.md`

### Developers Page (10 components)
- `02-01-DevelopersHero.md`
- `02-02-DevelopersProblem.md`
- `02-03-DevelopersSolution.md`
- `02-04-DevelopersHowItWorks.md`
- `02-05-DevelopersFeatures.md`
- `02-06-DevelopersCodeExamples.md`
- `02-07-DevelopersUseCases.md`
- `02-08-DevelopersPricing.md`
- `02-09-DevelopersTestimonials.md`
- `02-10-DevelopersCTA.md`

### Enterprise Page (11 components)
- `03-01-EnterpriseHero.md`
- `03-02-EnterpriseProblem.md`
- `03-03-EnterpriseSolution.md`
- `03-04-EnterpriseHowItWorks.md`
- `03-05-EnterpriseFeatures.md`
- `03-06-EnterpriseSecurity.md`
- `03-07-EnterpriseCompliance.md`
- `03-08-EnterpriseComparison.md`
- `03-09-EnterpriseUseCases.md`
- `03-10-EnterpriseTestimonials.md`
- `03-11-EnterpriseCTA.md`

### GPU Providers Page (11 components)
- `04-01-ProvidersHero.md`
- `04-02-ProvidersProblem.md`
- `04-03-ProvidersSolution.md`
- `04-04-ProvidersHowItWorks.md`
- `04-05-ProvidersFeatures.md`
- `04-06-ProvidersMarketplace.md`
- `04-07-ProvidersEarnings.md`
- `04-08-ProvidersSecurity.md`
- `04-09-ProvidersUseCases.md`
- `04-10-ProvidersTestimonials.md`
- `04-11-ProvidersCTA.md`

### Features Page (8 components)
- `05-01-FeaturesHero.md`
- `05-02-CoreFeaturesTabs.md`
- `05-03-MultiBackendGPU.md`
- `05-04-CrossNodeOrchestration.md`
- `05-05-IntelligentModelManagement.md`
- `05-06-RealTimeProgress.md`
- `05-07-ErrorHandling.md`
- `05-08-SecurityIsolation.md`
- `05-09-AdditionalFeaturesGrid.md`

### Use Cases Page (3 components)
- `06-01-UseCasesHero.md`
- `06-02-UseCasesGrid.md`
- `06-03-IndustryUseCases.md`

### Pages Assembly (7 units)
- `07-01-HomeView.md` - Assemble Home page
- `07-02-DevelopersView.md` - Assemble Developers page
- `07-03-EnterpriseView.md` - Assemble Enterprise page
- `07-04-ProvidersView.md` - Assemble Providers page
- `07-05-FeaturesView.md` - Assemble Features page
- `07-06-UseCasesView.md` - Assemble Use Cases page
- `07-07-Testing.md` - Test all pages

### Missing Atoms (if needed)
- `08-01-Tabs.md` - For AudienceSelector (check if exists)
- `08-02-Accordion.md` - For FAQSection (check if exists)

---

## 📈 Progress Tracking

**Total Units:** 61  
**Completed:** 3 (4.9%)  
**Remaining:** 58 (95.1%)

---

## 🎯 Recommended Team Distribution

- **TEAM-FE-004:** Complete Home Page (11 units) + HomeView
- **TEAM-FE-005:** Developers Page (10 units) + DevelopersView
- **TEAM-FE-006:** Enterprise Page (11 units) + EnterpriseView
- **TEAM-FE-007:** GPU Providers Page (11 units) + ProvidersView
- **TEAM-FE-008:** Features Page (9 units) + FeaturesView
- **TEAM-FE-009:** Use Cases Page (3 units) + UseCasesView + Final Testing

---

## 📝 Unit of Work Template

Each unit file contains:
1. **Component Name** & React reference location
2. **Estimated Time** (1-2 hours)
3. **Dependencies** (atoms/molecules needed)
4. **Implementation Checklist**
5. **Testing Checklist**
6. **Completion Criteria**

---

**Next:** Start with `01-04-AudienceSelector.md`

---

## 📚 Required Reading

**BEFORE starting this unit, read:**

1. **Design Tokens (CRITICAL):** `00-DESIGN-TOKENS-CRITICAL.md`
   - DO NOT copy colors from React reference
   - Use design tokens: `bg-primary`, `text-foreground`, etc.
   - Translation guide: React colors → Vue tokens

2. **Engineering Rules:** `/frontend/FRONTEND_ENGINEERING_RULES.md`
   - Section 2: Design tokens requirement
   - Section 3: Histoire `.story.vue` format
   - Section 8: Port vs create distinction

3. **Examples:** Look at completed components
   - HeroSection: `/frontend/libs/storybook/stories/organisms/HeroSection/`
   - WhatIsRbee: `/frontend/libs/storybook/stories/organisms/WhatIsRbee/`
   - ProblemSection: `/frontend/libs/storybook/stories/organisms/ProblemSection/`

**Key Rules:**
- ✅ Use `.story.vue` format (NOT `.story.ts`)
- ✅ Use design tokens (NOT hardcoded colors like `bg-amber-500`)
- ✅ Import from workspace: `import { Button } from 'rbee-storybook/stories'`
- ✅ Add team signature: `<!-- TEAM-FE-XXX: Implemented ComponentName -->`
- ✅ Export in `stories/index.ts`

---


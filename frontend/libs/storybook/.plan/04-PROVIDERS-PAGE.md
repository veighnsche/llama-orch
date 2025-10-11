# GPU Providers Page - Units 04-01 through 04-11

**Total Units:** 11 organisms + 1 page assembly  
**Estimated Time:** 13-16 hours  
**React Reference:** `/frontend/reference/v0/app/gpu-providers/page.tsx`

---

## 📋 Units

### Organisms (11)
- `04-01-ProvidersHero.md`
- `04-02-ProvidersProblem.md`
- `04-03-ProvidersSolution.md`
- `04-04-ProvidersHowItWorks.md`
- `04-05-ProvidersFeatures.md`
- `04-06-ProvidersMarketplace.md` - Marketplace features
- `04-07-ProvidersEarnings.md` - Earnings calculator/info
- `04-08-ProvidersSecurity.md`
- `04-09-ProvidersUseCases.md`
- `04-10-ProvidersTestimonials.md`
- `04-11-ProvidersCTA.md`

### Page Assembly
- `07-04-ProvidersView.md` - Assemble page + route `/gpu-providers`

---

## 📦 React Component Locations

All in `/frontend/reference/v0/components/providers/`

---

**Assigned To:** TEAM-FE-007

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


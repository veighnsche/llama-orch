# Developers Page - Units 02-01 through 02-10

**Total Units:** 10 organisms + 1 page assembly  
**Estimated Time:** 12-15 hours  
**React Reference:** `/frontend/reference/v0/app/developers/page.tsx`

---

## 📋 Units

### Organisms (10)
- `02-01-DevelopersHero.md` - Hero section for developers
- `02-02-DevelopersProblem.md` - Developer pain points
- `02-03-DevelopersSolution.md` - How rbee solves it
- `02-04-DevelopersHowItWorks.md` - Developer workflow
- `02-05-DevelopersFeatures.md` - Developer-focused features
- `02-06-DevelopersCodeExamples.md` - Code snippets/examples
- `02-07-DevelopersUseCases.md` - Developer use cases
- `02-08-DevelopersPricing.md` - Pricing for developers
- `02-09-DevelopersTestimonials.md` - Developer testimonials
- `02-10-DevelopersCTA.md` - Developer CTA

### Page Assembly
- `07-02-DevelopersView.md` - Assemble page + route `/developers`

---

## 📦 React Component Locations

All in `/frontend/reference/v0/components/developers/`:
- `developers-hero.tsx`
- `developers-problem.tsx`
- `developers-solution.tsx`
- `developers-how-it-works.tsx`
- `developers-features.tsx`
- `developers-code-examples.tsx`
- `developers-use-cases.tsx`
- `developers-pricing.tsx`
- `developers-testimonials.tsx`
- `developers-cta.tsx`

---

## 📝 Notes

**Start After:** Home page complete  
**Assigned To:** TEAM-FE-005  
**Pattern:** Similar to Home page organisms

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


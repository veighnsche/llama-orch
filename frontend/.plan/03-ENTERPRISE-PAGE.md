# Enterprise Page - Units 03-01 through 03-11

**Total Units:** 11 organisms + 1 page assembly  
**Estimated Time:** 13-16 hours  
**React Reference:** `/frontend/reference/v0/app/enterprise/page.tsx`

---

## 📋 Units

### Organisms (11)
- `03-01-EnterpriseHero.md`
- `03-02-EnterpriseProblem.md`
- `03-03-EnterpriseSolution.md`
- `03-04-EnterpriseHowItWorks.md`
- `03-05-EnterpriseFeatures.md`
- `03-06-EnterpriseSecurity.md` - Security features
- `03-07-EnterpriseCompliance.md` - Compliance info
- `03-08-EnterpriseComparison.md` - Enterprise comparison
- `03-09-EnterpriseUseCases.md`
- `03-10-EnterpriseTestimonials.md`
- `03-11-EnterpriseCTA.md`

### Page Assembly
- `07-03-EnterpriseView.md` - Assemble page + route `/enterprise`

---

## 📦 React Component Locations

All in `/frontend/reference/v0/components/enterprise/`

---

**Assigned To:** TEAM-FE-006

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


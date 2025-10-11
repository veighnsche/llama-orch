# Use Cases Page - Units 06-01 through 06-03

**Total Units:** 3 organisms + 1 page assembly  
**Estimated Time:** 4-5 hours  
**React Reference:** `/frontend/reference/v0/app/use-cases/page.tsx`

---

## 📋 Units

### Organisms (3)
- `06-01-UseCasesHero.md`
- `06-02-UseCasesGrid.md` - Grid of use cases
- `06-03-IndustryUseCases.md` - Industry-specific cases

### Page Assembly
- `07-06-UseCasesView.md` - Assemble page + route `/use-cases`

---

## 📦 React Component Locations

In `/frontend/reference/v0/components/` (no subdirectory for these)

---

**Assigned To:** TEAM-FE-009  
**Note:** Smallest page, good for final team

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


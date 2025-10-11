# Unit 01-09: ComparisonSection Component

**Status:** 🔴 Not Started  
**Estimated Time:** 2 hours  
**React Reference:** `/frontend/reference/v0/components/comparison-section.tsx`  
**Vue Location:** `/frontend/libs/storybook/stories/organisms/ComparisonSection/ComparisonSection.vue`

---

## 🎯 Description
Comparison table: rbee vs Cloud Providers vs Self-Hosted

## 📦 Dependencies
- Table/grid layout
- Check/X icons from `lucide-vue-next`

## ✅ Checklist
- [ ] Read React reference
- [ ] Implement comparison table
- [ ] Create story
- [ ] Test in Histoire
- [ ] Verify responsive table

**Next:** `01-10-PricingSection.md`

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


# Unit 01-13: FAQSection Component

**Status:** 🔴 Not Started  
**Estimated Time:** 2 hours  
**React Reference:** `/frontend/reference/v0/components/faq-section.tsx`  
**Vue Location:** `/frontend/libs/storybook/stories/organisms/FAQSection/FAQSection.vue`

---

## 🎯 Description
FAQ accordion with expandable questions/answers

## 📦 Dependencies
- ⚠️ **Accordion atom** - Check if exists, may need to port

## ✅ Checklist
- [ ] Check if Accordion atom exists
- [ ] If missing, create `08-02-Accordion.md` and implement first
- [ ] Read React reference
- [ ] Implement component
- [ ] Create story
- [ ] Test accordion expand/collapse
- [ ] Test in Histoire

**Next:** `01-14-CTASection.md`

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


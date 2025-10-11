# Unit 07-01: HomeView Page Assembly

**Status:** 🔴 Not Started  
**Estimated Time:** 1 hour  
**Priority:** High (after all Home organisms complete)

---

## 📍 Page Info

**Name:** HomeView  
**Route:** `/`  
**React Reference:** `/frontend/reference/v0/app/page.tsx`  
**Vue Location:** `/frontend/bin/commercial-frontend/src/views/HomeView.vue`

---

## 🎯 What This Does

Assembles all Home page organisms into the main landing page.

---

## 📦 Dependencies

**All Home Page Organisms Must Be Complete:**
- ✅ HeroSection (TEAM-FE-003)
- ✅ WhatIsRbee (TEAM-FE-003)
- ✅ ProblemSection (TEAM-FE-003)
- ⏳ AudienceSelector (01-04)
- ⏳ SolutionSection (01-05)
- ⏳ HowItWorksSection (01-06)
- ⏳ FeaturesSection (01-07)
- ⏳ UseCasesSection (01-08)
- ⏳ ComparisonSection (01-09)
- ⏳ PricingSection (01-10)
- ⏳ SocialProofSection (01-11)
- ⏳ TechnicalSection (01-12)
- ⏳ FAQSection (01-13)
- ⏳ CTASection (01-14)
- EmailCapture (scaffolded)
- Footer (already exists)

---

## ✅ Implementation Checklist

### Step 1: Read React Reference
- [ ] Read `/frontend/reference/v0/app/page.tsx`
- [ ] Note component order (lines 20-36)

### Step 2: Implement Page
- [ ] Open `/frontend/bin/commercial-frontend/src/views/HomeView.vue`
- [ ] Add TEAM signature
- [ ] Import all organisms:
  ```vue
  <script setup lang="ts">
  import {
    HeroSection,
    WhatIsRbee,
    AudienceSelector,
    EmailCapture,
    ProblemSection,
    SolutionSection,
    HowItWorksSection,
    FeaturesSection,
    UseCasesSection,
    ComparisonSection,
    PricingSection,
    SocialProofSection,
    TechnicalSection,
    FAQSection,
    CTASection,
    Footer,
  } from 'rbee-storybook/stories'
  </script>
  ```
- [ ] Add template with all sections in correct order
- [ ] Add `pt-16` class to main for navigation spacing

### Step 3: Verify Route
- [ ] Check `/frontend/bin/commercial-frontend/src/router/index.ts`
- [ ] Verify `/` route points to HomeView
- [ ] Route should already be configured

### Step 4: Test
- [ ] Run `pnpm dev` in commercial-frontend
- [ ] Navigate to `http://localhost:5173/`
- [ ] Verify all sections render
- [ ] Scroll through entire page
- [ ] Check responsive design
- [ ] Compare with React reference at `http://localhost:3000/`

---

## 🧪 Testing Checklist

- [ ] Page loads without errors
- [ ] All 14+ sections render
- [ ] Sections in correct order
- [ ] Navigation works (if implemented)
- [ ] Footer displays at bottom
- [ ] Responsive on mobile
- [ ] Responsive on tablet
- [ ] Responsive on desktop
- [ ] No console errors
- [ ] Visual parity with React reference

---

## ✅ Completion Criteria

- [ ] Page assembled
- [ ] All organisms imported
- [ ] Route verified
- [ ] Page tested in browser
- [ ] Visual parity confirmed
- [ ] Team signature added
- [ ] No errors in console

---

## 📝 Notes

**Blockers:** All Home page organisms (01-04 through 01-14) must be complete first  
**Time:** 1 hour  
**Next:** `02-01-DevelopersHero.md` (Start Developers page)

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


# Unit 01-05: SolutionSection Component

**Status:** 🔴 Not Started  
**Assigned To:** TEAM-FE-004  
**Estimated Time:** 2 hours  
**Priority:** High

---

## 📍 Component Info

**Name:** SolutionSection  
**Type:** Organism  
**React Reference:** `/frontend/reference/v0/components/solution-section.tsx`  
**Vue Location:** `/frontend/libs/storybook/stories/organisms/SolutionSection/SolutionSection.vue`  
**Story Location:** `/frontend/libs/storybook/stories/organisms/SolutionSection/SolutionSection.story.vue`

---

## 🎯 What This Component Does

Shows the rbee solution with:
- Title and description
- "Bee Architecture" diagram (Queen, Hive Managers, Workers)
- 4 key benefits cards with icons

**Complexity:** High - Has custom architecture diagram

---

## 📦 Dependencies

### Required Atoms:
- ✅ Card (already exists)

### Icons Needed:
- From `lucide-vue-next`: Anchor, DollarSign, Laptop, Shield

---

## ✅ Implementation Checklist

### Step 1: Read React Reference
- [ ] Read `/frontend/reference/v0/components/solution-section.tsx`
- [ ] Note architecture diagram structure (lines 17-71)
- [ ] Note 4 benefits cards (lines 74-114)

### Step 2: Implement Component
- [ ] Open `/frontend/libs/storybook/stories/organisms/SolutionSection/SolutionSection.vue`
- [ ] Add TEAM signature
- [ ] Import icons
- [ ] Define Props interface (optional - mostly static content)
- [ ] Implement architecture diagram section
- [ ] Implement 4 benefits cards
- [ ] Use emoji: 👑 🍯 🐝

### Step 3: Create Story
- [ ] Create `SolutionSection.story.vue`
- [ ] Default variant

### Step 4: Test
- [ ] Test in Histoire
- [ ] Verify architecture diagram displays correctly
- [ ] Verify all 4 benefit cards show
- [ ] Check responsive grid (2 cols on md, 4 on lg)

---

## ✅ Completion Criteria

- [ ] Component implemented
- [ ] Story created
- [ ] Architecture diagram renders correctly
- [ ] All 4 benefit cards display
- [ ] Responsive design works
- [ ] Team signature added

---

**Time:** 2 hours  
**Next:** `01-06-HowItWorksSection.md`

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


# Unit 01-04: AudienceSelector Component

**Status:** 🔴 Not Started  
**Assigned To:** TEAM-FE-004  
**Estimated Time:** 2 hours  
**Priority:** High

---

## 📍 Component Info

**Name:** AudienceSelector  
**Type:** Organism  
**React Reference:** `/frontend/reference/v0/components/audience-selector.tsx`  
**Vue Location:** `/frontend/libs/storybook/stories/organisms/AudienceSelector/AudienceSelector.vue`  
**Story Location:** `/frontend/libs/storybook/stories/organisms/AudienceSelector/AudienceSelector.story.vue`

---

## 🎯 What This Component Does

Interactive tabs component that shows different content for three audiences:
- Developers
- GPU Providers
- Enterprises

Uses tabs to switch between audience-specific content.

---

## 📦 Dependencies

### Required Atoms:
- ✅ Button (already exists)
- ⚠️ **Tabs** - Check if exists, may need to port from React

### Icons Needed:
- From `lucide-vue-next`: Code, Server, Building2

---

## ✅ Implementation Checklist

### Step 1: Check Dependencies
- [ ] Verify Tabs atom exists in `/frontend/libs/storybook/stories/atoms/Tabs/`
- [ ] If missing, create unit `08-01-Tabs.md` and implement first
- [ ] Verify Tabs is exported in `stories/index.ts`

### Step 2: Read React Reference
- [ ] Read `/frontend/reference/v0/components/audience-selector.tsx`
- [ ] Note tab structure and content
- [ ] Identify props needed

### Step 3: Implement Component
- [ ] Open `/frontend/libs/storybook/stories/organisms/AudienceSelector/AudienceSelector.vue`
- [ ] Add TEAM signature: `<!-- TEAM-FE-004: Implemented AudienceSelector -->`
- [ ] Import dependencies:
  ```vue
  import { Button, Tabs } from 'rbee-storybook/stories'
  import { Code, Server, Building2 } from 'lucide-vue-next'
  ```
- [ ] Define Props interface (if needed for configurability)
- [ ] Port template from React
- [ ] Convert JSX to Vue template syntax
- [ ] Use Tailwind classes from React reference

### Step 4: Create Story
- [ ] Create `AudienceSelector.story.vue`
- [ ] Add TEAM signature
- [ ] Create Default variant
- [ ] Test all three tabs work

### Step 5: Test
- [ ] Run `pnpm story:dev` in storybook directory
- [ ] Verify component renders in Histoire
- [ ] Test tab switching works
- [ ] Verify responsive design
- [ ] Check Tailwind styling applied
- [ ] No console errors

---

## 🧪 Testing Checklist

- [ ] Component renders without errors
- [ ] All three tabs display correctly
- [ ] Tab switching works smoothly
- [ ] Content changes when tabs switch
- [ ] Icons display correctly
- [ ] Buttons are clickable
- [ ] Responsive on mobile (stacked layout)
- [ ] Responsive on tablet
- [ ] Responsive on desktop
- [ ] Tailwind classes applied
- [ ] No console errors or warnings

---

## ✅ Completion Criteria

- [ ] Component implemented in Vue
- [ ] Story file created
- [ ] Component tested in Histoire
- [ ] All tabs functional
- [ ] Responsive design works
- [ ] Team signature added
- [ ] No TODO comments
- [ ] Visual parity with React reference

---

## 📝 Notes

**Complexity:** Medium (requires Tabs atom)  
**Blockers:** May need to port Tabs atom first  
**Time Estimate:** 2 hours (includes Tabs port if needed)

---

**Next Unit:** `01-05-SolutionSection.md`

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


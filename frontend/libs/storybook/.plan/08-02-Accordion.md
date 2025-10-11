# Unit 08-02: Accordion Atom (If Needed)

**Status:** ⚠️ Conditional - Check if exists first  
**Estimated Time:** 2 hours  
**Priority:** Required for FAQSection (01-13)

---

## 🎯 Check First

Before implementing, verify:
```bash
ls /home/vince/Projects/llama-orch/frontend/libs/storybook/stories/atoms/Accordion/
```

If Accordion.vue exists and is implemented, **skip this unit**.

---

## 📍 Component Info

**Name:** Accordion  
**Type:** Atom  
**React Reference:** `/frontend/reference/v0/components/ui/accordion.tsx`  
**Vue Location:** `/frontend/libs/storybook/stories/atoms/Accordion/Accordion.vue`

---

## 🎯 What This Does

Expandable/collapsible accordion component using Radix Vue

---

## 📦 Dependencies

- `radix-vue` - AccordionRoot, AccordionItem, AccordionTrigger, AccordionContent

---

## ✅ Implementation Checklist

### Step 1: Read React Reference
- [ ] Read `/frontend/reference/v0/components/ui/accordion.tsx`
- [ ] Note Radix UI Accordion components used

### Step 2: Port to Vue
- [ ] Create Accordion.vue using Radix Vue
- [ ] Import from `radix-vue`:
  ```vue
  import { AccordionRoot, AccordionItem, AccordionTrigger, AccordionContent } from 'radix-vue'
  ```
- [ ] Port Tailwind classes
- [ ] Add chevron icon animation

### Step 3: Create Story
- [ ] Create Accordion.story.vue
- [ ] Show multiple items example
- [ ] Test expand/collapse

### Step 4: Export
- [ ] Add to `stories/index.ts`:
  ```typescript
  export { default as Accordion } from './atoms/Accordion/Accordion.vue'
  ```

### Step 5: Test
- [ ] Test in Histoire
- [ ] Verify expand/collapse works
- [ ] Verify animations work
- [ ] Verify styling applied

---

## ✅ Completion Criteria

- [ ] Accordion component works
- [ ] Story created
- [ ] Exported in index.ts
- [ ] Expand/collapse functional
- [ ] Ready for use in FAQSection

---

**Blocker For:** 01-13-FAQSection.md

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


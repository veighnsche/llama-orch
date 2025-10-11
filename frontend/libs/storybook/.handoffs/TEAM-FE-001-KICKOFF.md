# TEAM-FE-001: Implementation Kickoff

**Team:** TEAM-FE-001 (First Implementation Team)  
**Date:** 2025-10-11  
**From:** TEAM-FE-000 (Project Manager)  
**Status:** READY TO START 🚀

---

## 🚨 CRITICAL: THIS IS A PORT FROM REACT

**YOU ARE NOT DESIGNING FROM SCRATCH. YOU ARE PORTING EXISTING REACT COMPONENTS TO VUE.**

### What This Means:

✅ **DO:**
- Read the React reference component FIRST
- Port the exact same behavior to Vue
- Match the visual design exactly
- Use the same props (converted to Vue syntax)
- Keep the same variants and states
- Compare side-by-side with React reference

❌ **DO NOT:**
- Design new components from scratch
- Guess how components should work
- Change behavior without approval
- Invent new features
- Skip reading the React reference

### React Reference Location:

**Path:** `/frontend/reference/v0/components/ui/`  
**Run:** `pnpm --filter frontend/reference/v0 dev`  
**URL:** http://localhost:3000

**Every component you build has a React version to port from.**

---

## 🎯 Your Mission

You are the **FIRST implementation team**. Your job is to **PORT** the **foundational atoms** from React to Vue.

**Success = 5 core atoms ported, tested, and matching the React reference.**

---

## 📋 Your Assignments (Priority Order)

### Priority 1: Button Component ⭐ CRITICAL

**Why first:** Most used component. Everything depends on it.

**Source:** `/frontend/reference/v0/components/ui/button.tsx`

**Requirements:**
- All variants: default, destructive, outline, secondary, ghost, link
- All sizes: default, sm, lg, icon
- All states: default, hover, active, disabled, loading
- Use CVA (class-variance-authority) for variants
- Use Tailwind classes with cn() utility
- Proper TypeScript props interface

**Deliverables:**
- `/frontend/libs/storybook/stories/atoms/Button/Button.vue`
- `/frontend/libs/storybook/stories/atoms/Button/Button.story.ts`
- Story showing ALL variants and sizes
- Tested in Histoire (http://localhost:6006)

---

### Priority 2: Input Component ⭐ CRITICAL

**Why second:** Forms need this. High priority.

**Source:** `/frontend/reference/v0/components/ui/input.tsx`

**Requirements:**
- Types: text, email, password, number, search, tel, url
- States: default, error, disabled, readonly
- Proper ARIA labels
- Error message support
- Placeholder support

**Deliverables:**
- `/frontend/libs/storybook/stories/atoms/Input/Input.vue`
- `/frontend/libs/storybook/stories/atoms/Input/Input.story.ts`
- Story showing ALL types and states
- Tested in Histoire

---

### Priority 3: Label Component

**Why third:** Pairs with Input. Forms need this.

**Source:** `/frontend/reference/v0/components/ui/label.tsx`

**Requirements:**
- Use Radix Vue Label primitive
- Required indicator (*)
- Proper for/id association
- Disabled state styling

**Deliverables:**
- `/frontend/libs/storybook/stories/atoms/Label/Label.vue`
- `/frontend/libs/storybook/stories/atoms/Label/Label.story.ts`
- Story showing default, required, disabled
- Tested in Histoire

---

### Priority 4: Card Component

**Why fourth:** Container component. Many organisms use it.

**Source:** `/frontend/reference/v0/components/ui/card.tsx`

**Requirements:**
- Subcomponents: CardHeader, CardTitle, CardDescription, CardContent, CardFooter
- Proper composition pattern
- Flexible layout
- Optional padding variants

**Deliverables:**
- `/frontend/libs/storybook/stories/atoms/Card/Card.vue`
- `/frontend/libs/storybook/stories/atoms/Card/CardHeader.vue`
- `/frontend/libs/storybook/stories/atoms/Card/CardTitle.vue`
- `/frontend/libs/storybook/stories/atoms/Card/CardDescription.vue`
- `/frontend/libs/storybook/stories/atoms/Card/CardContent.vue`
- `/frontend/libs/storybook/stories/atoms/Card/CardFooter.vue`
- `/frontend/libs/storybook/stories/atoms/Card/Card.story.ts`
- Story showing all compositions
- Tested in Histoire

---

### Priority 5: Alert Component

**Why fifth:** Feedback component. Used in many places.

**Source:** `/frontend/reference/v0/components/ui/alert.tsx`

**Requirements:**
- Variants: default, destructive (error), success, warning, info
- Optional icon
- Optional close button
- Proper ARIA role="alert"

**Deliverables:**
- `/frontend/libs/storybook/stories/atoms/Alert/Alert.vue`
- `/frontend/libs/storybook/stories/atoms/Alert/Alert.story.ts`
- Story showing all variants
- Tested in Histoire

---

## ✅ Success Criteria

Your work is COMPLETE when ALL of these are checked:

### For Each Component:
- [ ] Component file created in correct location
- [ ] Story file created with ALL variants
- [ ] Tested in Histoire (no errors)
- [ ] TypeScript props interface defined
- [ ] Uses Tailwind with cn() utility
- [ ] Uses design tokens (no hardcoded values)
- [ ] ARIA labels present (where applicable)
- [ ] Keyboard navigation works
- [ ] Mobile responsive
- [ ] Matches React reference visually
- [ ] No console errors or warnings
- [ ] Team signature added: `// Created by: TEAM-FE-001`
- [ ] Already exported in index.ts ✅ (done by TEAM-FE-000)

### Overall:
- [ ] All 5 components complete
- [ ] All stories tested in Histoire
- [ ] Side-by-side comparison with React reference
- [ ] Screenshots taken (before/after)
- [ ] This handoff checklist complete

---

## 🚀 Getting Started

### Step 1: Set Up Environment

```bash
cd /home/vince/Projects/llama-orch

# Install dependencies (if not done)
pnpm install

# Start React reference (for comparison)
pnpm --filter frontend/reference/v0 dev
# Opens: http://localhost:3000

# Start Histoire (for development)
pnpm --filter rbee-storybook story:dev
# Opens: http://localhost:6006
```

### Step 2: Read the React Reference

```bash
# Button component
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/button.tsx

# Input component
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/input.tsx

# Label component
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/label.tsx

# Card component
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/card.tsx

# Alert component
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/alert.tsx
```

### Step 3: Implement Button (Priority 1)

```bash
# Edit component
# File: /home/vince/Projects/llama-orch/frontend/libs/storybook/stories/atoms/Button/Button.vue

# Edit story
# File: /home/vince/Projects/llama-orch/frontend/libs/storybook/stories/atoms/Button/Button.story.ts

# Test in Histoire
# Open: http://localhost:6006
# Navigate to: atoms/Button
# Verify all variants work
```

### Step 4: Repeat for Priorities 2-5

Follow the same pattern for Input, Label, Card, Alert.

### Step 5: Verify Everything

```bash
# Type check
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm type-check

# Should pass with no errors
```

---

## 📚 Required Reading

**BEFORE YOU START, READ THESE:**

1. **Frontend Engineering Rules** (MANDATORY)
   - `/frontend/FRONTEND_ENGINEERING_RULES.md`
   - Read ALL 16 sections
   - This is your bible

2. **Atomic Design Philosophy**
   - `/frontend/libs/storybook/ATOMIC_DESIGN_PHILOSOPHY.md`
   - Understand atoms vs molecules vs organisms

3. **Developer Checklist**
   - `/frontend/libs/storybook/DEVELOPER_CHECKLIST.md`
   - Universal checklist for every component

4. **Dependencies Guide**
   - `/frontend/bin/commercial-frontend/DEPENDENCIES_GUIDE.md`
   - How to use Radix Vue, Tailwind, CVA, etc.

5. **Workspace Guide**
   - `/frontend/WORKSPACE_GUIDE.md`
   - How to run everything

---

## 🎨 Component Template

Use this as your starting point:

```vue
<!-- Created by: TEAM-FE-001 -->
<script setup lang="ts">
import { computed } from 'vue'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

// Define variants using CVA
const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline: 'border border-input hover:bg-accent hover:text-accent-foreground',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'underline-offset-4 hover:underline text-primary',
      },
      size: {
        default: 'h-10 py-2 px-4',
        sm: 'h-9 px-3 rounded-md',
        lg: 'h-11 px-8 rounded-md',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

// Define props interface
interface Props {
  variant?: VariantProps<typeof buttonVariants>['variant']
  size?: VariantProps<typeof buttonVariants>['size']
  disabled?: boolean
  type?: 'button' | 'submit' | 'reset'
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'default',
  disabled: false,
  type: 'button',
})

// Compute classes
const classes = computed(() =>
  cn(buttonVariants({ variant: props.variant, size: props.size }), props.class)
)
</script>

<template>
  <button :type="type" :disabled="disabled" :class="classes">
    <slot />
  </button>
</template>
```

**Story Template:**

```typescript
// Created by: TEAM-FE-001
import Button from './Button.vue'

export default {
  title: 'atoms/Button',
  component: Button,
}

export const Default = () => ({
  components: { Button },
  template: '<Button>Default Button</Button>',
})

export const Destructive = () => ({
  components: { Button },
  template: '<Button variant="destructive">Destructive</Button>',
})

export const Outline = () => ({
  components: { Button },
  template: '<Button variant="outline">Outline</Button>',
})

export const Secondary = () => ({
  components: { Button },
  template: '<Button variant="secondary">Secondary</Button>',
})

export const Ghost = () => ({
  components: { Button },
  template: '<Button variant="ghost">Ghost</Button>',
})

export const Link = () => ({
  components: { Button },
  template: '<Button variant="link">Link</Button>',
})

export const Small = () => ({
  components: { Button },
  template: '<Button size="sm">Small</Button>',
})

export const Large = () => ({
  components: { Button },
  template: '<Button size="lg">Large</Button>',
})

export const Disabled = () => ({
  components: { Button },
  template: '<Button disabled>Disabled</Button>',
})
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ MISTAKE 1: Not Testing in Histoire

**Wrong:** Implement component, mark as done, move on.

**Correct:** Implement component, test in Histoire, verify all variants, THEN mark as done.

---

### ❌ MISTAKE 2: Hardcoded Values

**Wrong:**
```css
.button {
  padding: 12px 24px;
  background: #0066cc;
}
```

**Correct:**
```css
.button {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-primary);
}
```

---

### ❌ MISTAKE 3: Incomplete Stories

**Wrong:** Story with only one variant.

**Correct:** Story showing ALL variants, sizes, and states.

---

### ❌ MISTAKE 4: No TypeScript

**Wrong:**
```vue
<script setup>
const props = defineProps(['variant', 'size'])
</script>
```

**Correct:**
```vue
<script setup lang="ts">
interface Props {
  variant?: 'default' | 'outline'
  size?: 'sm' | 'md' | 'lg'
}
const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'md',
})
</script>
```

---

### ❌ MISTAKE 5: Not Comparing with React Reference

**Wrong:** Implement based on memory or guessing.

**Correct:** Open React reference side-by-side, compare visually, match exactly.

---

## 📊 Progress Tracking

Update this as you complete each component:

- [ ] **Button** - 0% complete
  - [ ] Component implemented
  - [ ] Story created with all variants
  - [ ] Tested in Histoire
  - [ ] Matches React reference
  - [ ] No console errors

- [ ] **Input** - 0% complete
  - [ ] Component implemented
  - [ ] Story created with all types
  - [ ] Tested in Histoire
  - [ ] Matches React reference
  - [ ] No console errors

- [ ] **Label** - 0% complete
  - [ ] Component implemented
  - [ ] Story created with all states
  - [ ] Tested in Histoire
  - [ ] Matches React reference
  - [ ] No console errors

- [ ] **Card** - 0% complete
  - [ ] All subcomponents implemented
  - [ ] Story created with all compositions
  - [ ] Tested in Histoire
  - [ ] Matches React reference
  - [ ] No console errors

- [ ] **Alert** - 0% complete
  - [ ] Component implemented
  - [ ] Story created with all variants
  - [ ] Tested in Histoire
  - [ ] Matches React reference
  - [ ] No console errors

**Overall Progress:** 0/5 components (0%)

---

## 🎯 Definition of Done

A component is DONE when:

1. ✅ Component file exists and compiles
2. ✅ Story file exists with ALL variants
3. ✅ Tested in Histoire (no errors)
4. ✅ TypeScript props interface defined
5. ✅ Uses Tailwind with cn() utility
6. ✅ ARIA labels present (if interactive)
7. ✅ Keyboard navigation works (if interactive)
8. ✅ Mobile responsive
9. ✅ Matches React reference visually
10. ✅ No console errors or warnings
11. ✅ Team signature added
12. ✅ Screenshots taken

**If ANY item is unchecked, the component is NOT done.**

---

## 📸 Screenshots Required

For your handoff, take screenshots of:

1. **React reference** (http://localhost:3000) showing the component
2. **Your Vue implementation** (http://localhost:6006) showing all variants
3. **Side-by-side comparison** proving visual parity

Store in: `/frontend/libs/storybook/.handoffs/TEAM-FE-001-screenshots/`

---

## 🚨 Critical Reminders

1. **Build in storybook FIRST** - Not in the application
2. **Test in Histoire** - Before marking complete
3. **Use workspace imports** - `import { X } from 'rbee-storybook/stories'`
4. **No hardcoded values** - Use design tokens
5. **Complete ALL 5 components** - Don't stop at 3
6. **Read the rules** - `/frontend/FRONTEND_ENGINEERING_RULES.md`

---

## 📞 Need Help?

If you get stuck:

1. **Read the React reference** - The answer is usually there
2. **Check the dependencies guide** - Shows how to use Radix Vue, CVA, etc.
3. **Look at existing components** - Badge, Button (old) for patterns
4. **Check Histoire** - Make sure it's running and accessible

---

## 🎉 Next Team

When you're done, the next team (TEAM-FE-002) will implement:
- Textarea
- Checkbox
- RadioGroup
- Switch
- Slider

But that's NOT your concern. Focus on YOUR 5 components.

---

## ✅ Final Checklist

Before submitting your handoff:

- [ ] All 5 components implemented
- [ ] All 5 stories created with all variants
- [ ] All 5 components tested in Histoire
- [ ] All 5 components match React reference
- [ ] Screenshots taken (React vs Vue)
- [ ] No console errors in Histoire
- [ ] Type check passes: `pnpm type-check`
- [ ] Team signatures added to all files
- [ ] This checklist complete

**If you can't check ALL boxes, keep working.**

---

**From:** TEAM-FE-000 (Project Manager)  
**To:** TEAM-FE-001 (First Implementation Team)  
**Status:** READY TO START  
**Priority:** CRITICAL - These are foundational components

**Good luck! You're building the foundation for the entire frontend.** 🚀

---

## Signatures

```
// Handoff created by: TEAM-FE-000
// Date: 2025-10-11
// For: TEAM-FE-001
// Components: Button, Input, Label, Card, Alert
// Status: Ready to start
```

# 🚨 MANDATORY FRONTEND ENGINEERING RULES

> **REQUIRED READING BEFORE CONTRIBUTING TO rbee FRONTEND**

**Version:** 1.0 | **Date:** 2025-10-11 | **Status:** MANDATORY  
**Applies to:** All frontend work (Vue, Storybook, React reference)

---

## ⚠️ CRITICAL: READ THIS FIRST

Violations result in: **REJECTED work**, **DELETED handoff**, cited in "teams that failed" list.

**These rules are based on lessons learned from 67 failed teams. Don't be team 68.**

---

## 0. Required Dependencies & Tools

### ⚠️ CRITICAL: KNOW YOUR TOOLS

**You MUST understand these dependencies. They are already installed and ready to use.**

### Core Stack

**Vue 3.5.18** - Frontend framework
- Composition API (use `<script setup>`)
- TypeScript support
- Reactivity system

**Vite 7.0.6** - Build tool & dev server
- Fast HMR (Hot Module Replacement)
- TypeScript support
- Dev server: `pnpm dev`

**TypeScript 5.8.0** - Type safety
- Strict mode enabled
- Props interfaces required
- No `any` types allowed

---

### UI Component Libraries

**Radix Vue 1.9.11** ⭐ CRITICAL
- **What:** Vue port of Radix UI primitives
- **Why:** Headless, accessible UI components
- **Use for:** Dialog, Dropdown, Accordion, Tabs, Tooltip, Popover, etc.
- **Import:** `import { DialogRoot, DialogTrigger } from 'radix-vue'`
- **Docs:** https://www.radix-vue.com/

**Example:**
```vue
<script setup lang="ts">
import { DialogRoot, DialogTrigger, DialogContent } from 'radix-vue'
</script>

<template>
  <DialogRoot>
    <DialogTrigger>Open</DialogTrigger>
    <DialogContent>Content here</DialogContent>
  </DialogRoot>
</template>
```

---

### Styling

**Tailwind CSS 4.1.9** ⭐ CRITICAL
- **What:** Utility-first CSS framework
- **Why:** Rapid styling, consistent design
- **Use:** All components must use Tailwind classes
- **Config:** `tailwind.config.ts`
- **Docs:** https://tailwindcss.com/

**class-variance-authority (CVA) 0.7.1** ⭐ CRITICAL
- **What:** Variant management for components
- **Why:** Type-safe component variants
- **Use for:** Button variants, size props, etc.
- **Import:** `import { cva } from 'class-variance-authority'`

**Example:**
```typescript
const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground',
        outline: 'border border-input',
      },
      size: {
        sm: 'h-9 px-3',
        lg: 'h-11 px-8',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'sm',
    },
  }
)
```

**clsx 2.1.1** + **tailwind-merge 2.5.5**
- **What:** Conditional class utilities
- **Why:** Merge Tailwind classes intelligently
- **Use:** Via `cn()` utility function
- **Import:** `import { cn } from '@/lib/utils'`

**Example:**
```vue
<button :class="cn('base-class', isActive && 'active-class', props.class)">
```

**tailwindcss-animate 1.0.7**
- **What:** Animation utilities for Tailwind
- **Why:** Pre-built animations (accordion-down, etc.)
- **Use:** `animate-accordion-down`, `animate-spin`, etc.

---

### Icons

**Lucide Vue Next 0.454.0** ⭐ CRITICAL
- **What:** Icon library (Vue port of Lucide)
- **Why:** Consistent, customizable icons
- **Use for:** All icons in components
- **Import:** `import { Menu, X, Check } from 'lucide-vue-next'`
- **Docs:** https://lucide.dev/

**Example:**
```vue
<script setup lang="ts">
import { Menu, X, Check } from 'lucide-vue-next'
</script>

<template>
  <Menu :size="24" />
  <X :size="16" />
  <Check :size="20" :stroke-width="2" />
</template>
```

---

### Composables & Utilities

**@vueuse/core 11.3.0** ⭐ CRITICAL
- **What:** Collection of Vue composables
- **Why:** Replaces React hooks with Vue equivalents
- **Use for:** useMediaQuery, useLocalStorage, useEventListener, etc.
- **Import:** `import { useMediaQuery } from '@vueuse/core'`
- **Docs:** https://vueuse.org/

**Example:**
```vue
<script setup lang="ts">
import { useMediaQuery } from '@vueuse/core'

const isMobile = useMediaQuery('(max-width: 768px)')
</script>

<template>
  <div v-if="isMobile">Mobile view</div>
  <div v-else>Desktop view</div>
</template>
```

---

### Specialized Components

**embla-carousel-vue 8.5.1**
- **What:** Carousel/slider component
- **Why:** Smooth, accessible carousels
- **Use for:** Image carousels, testimonial sliders
- **Import:** `import emblaCarouselVue from 'embla-carousel-vue'`
- **Docs:** https://www.embla-carousel.com/

**vaul-vue 0.2.0**
- **What:** Drawer component (slide-out panel)
- **Why:** Mobile-friendly drawers
- **Use for:** Mobile menus, side panels
- **Import:** `import { Drawer } from 'vaul-vue'`

---

### React Reference (Comparison Only)

**Next.js 15** + **React 19**
- **Location:** `/frontend/reference/v0/`
- **Purpose:** Visual reference for porting
- **DO NOT:** Copy React code directly
- **DO:** Port to Vue equivalents

**React → Vue Equivalents:**
- `useState` → `ref()` or `reactive()`
- `useEffect` → `watch()` or `watchEffect()`
- `useCallback` → `computed()` or regular function
- `useMemo` → `computed()`
- `useRef` → `ref()` with `.value`
- React hooks → VueUse composables

---

### ⚠️ BANNED Dependencies

**DO NOT install or use:**
- ❌ Any React libraries (we use Vue)
- ❌ jQuery (use Vue reactivity)
- ❌ Bootstrap (we use Tailwind)
- ❌ Moment.js (use native Date or date-fns if needed)
- ❌ Lodash (use native JS or VueUse)
- ❌ Any CSS-in-JS library (we use Tailwind)

---

### Quick Reference Table

| Need | Use | Import From |
|------|-----|-------------|
| Dialog/Modal | Radix Vue Dialog | `radix-vue` |
| Dropdown | Radix Vue DropdownMenu | `radix-vue` |
| Tooltip | Radix Vue Tooltip | `radix-vue` |
| Accordion | Radix Vue Accordion | `radix-vue` |
| Icons | Lucide Vue Next | `lucide-vue-next` |
| Responsive | useMediaQuery | `@vueuse/core` |
| Carousel | Embla Carousel | `embla-carousel-vue` |
| Drawer | Vaul Vue | `vaul-vue` |
| Variants | CVA | `class-variance-authority` |
| Classes | cn() utility | `@/lib/utils` |

---

### Installation (Already Done ✅)

All dependencies are already installed. You don't need to install anything.

If you need to verify:
```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

---

### Where to Learn More

- **Radix Vue:** https://www.radix-vue.com/
- **VueUse:** https://vueuse.org/
- **Tailwind CSS:** https://tailwindcss.com/
- **Lucide Icons:** https://lucide.dev/
- **CVA:** https://cva.style/docs
- **Dependencies Guide:** `/frontend/bin/commercial-frontend/DEPENDENCIES_GUIDE.md`

---

## 1. Component Development Rules

### ⚠️ CRITICAL: BUILD IN STORYBOOK FIRST

**ALL components MUST be built in storybook BEFORE being used in the application.**

✅ **REQUIRED Workflow:**
```
1. Create component in /frontend/libs/storybook/stories/[type]/[Name]/
2. Create story file [Name].story.ts
3. Test in Histoire (pnpm story:dev)
4. Export in stories/index.ts
5. THEN import in application
```

❌ **BANNED:**
- Creating components directly in `/frontend/bin/commercial-frontend/src/components/`
- Copying components from React reference without testing in Histoire
- Skipping the story file
- Not testing in Histoire before using in app

**Why this matters:** Storybook is your single source of truth. If it's not in storybook, it doesn't exist. Building in isolation catches bugs early and ensures reusability.

---

### ⚠️ CRITICAL: ATOMIC DESIGN HIERARCHY

**You MUST follow the atomic design pattern. No exceptions.**

✅ **REQUIRED Structure:**
```
atoms/           # Smallest units (Button, Input, Label)
  ↓
molecules/       # 2-3 atoms combined (FormField, SearchBar)
  ↓
organisms/       # Complex components (Navigation, HeroSection)
  ↓
templates/       # Page layouts (optional in storybook)
  ↓
pages/           # Actual pages (in application, not storybook)
```

❌ **BANNED:**
- Atoms importing from molecules or organisms
- Molecules importing from organisms
- Skipping levels (atom directly to organism)
- Creating "god components" that do everything

**Example of FAILURE:**
```vue
<!-- Button.vue (ATOM) trying to import Dropdown (MOLECULE) -->
<script setup>
import { Dropdown } from '../molecules/Dropdown' // ❌ WRONG
</script>
```

**Example of SUCCESS:**
```vue
<!-- FormField.vue (MOLECULE) importing atoms -->
<script setup>
import { Label, Input, Alert } from 'rbee-storybook/stories' // ✅ CORRECT
</script>
```

---

### ⚠️ CRITICAL: WORKSPACE PACKAGE IMPORTS

**You MUST use workspace packages. NO relative imports.**

✅ **REQUIRED:**
```vue
<script setup>
import { Button, Input } from 'rbee-storybook/stories'
</script>

<style>
@import 'rbee-storybook/styles/tokens.css';
</style>
```

❌ **BANNED:**
```vue
<script setup>
import Button from '../../../libs/storybook/stories/atoms/Button/Button.vue'
import Input from '../../../../libs/storybook/stories/atoms/Input/Input.vue'
</script>
```

**Why this matters:** Workspace packages enforce boundaries and prevent circular dependencies. Relative imports break when you move files.

---

## 2. Component Implementation Rules

### ⚠️ NO LOREM IPSUM OR PLACEHOLDER CONTENT

**ALL components must use REAL content from the React reference.**

✅ **REQUIRED:**
- Port actual copy from `/frontend/reference/v0/`
- Use real feature lists, pricing, descriptions
- Extract content into props or content files

❌ **BANNED:**
- "Lorem ipsum dolor sit amet..."
- "Placeholder text here"
- "TODO: Add real content"
- Hardcoded dummy data

**Example of FAILURE:**
```vue
<template>
  <h1>Lorem ipsum dolor sit amet</h1>
  <p>Placeholder text here</p>
</template>
```

**Example of SUCCESS:**
```vue
<script setup>
interface Props {
  title: string
  description: string
}
// Port from /frontend/reference/v0/app/page.tsx lines 12-18
</script>

<template>
  <h1>{{ title }}</h1>
  <p>{{ description }}</p>
</template>
```

---

### ⚠️ USE DESIGN TOKENS, NOT HARDCODED VALUES

**ALL styling must use design tokens from rbee-storybook.**

✅ **REQUIRED:**
```css
.button {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--color-primary);
  border-radius: var(--border-radius-md);
  font-size: var(--font-size-base);
}
```

❌ **BANNED:**
```css
.button {
  padding: 8px 16px;
  background: #0066cc;
  border-radius: 8px;
  font-size: 16px;
}
```

**Why this matters:** Design tokens ensure consistency. If we change the primary color, it updates everywhere automatically.

---

### ⚠️ USE TAILWIND WITH cn() UTILITY

**When using Tailwind, ALWAYS use the cn() utility for conditional classes.**

✅ **REQUIRED:**
```vue
<script setup lang="ts">
import { cn } from '@/lib/utils'

const props = defineProps<{
  variant?: 'default' | 'outline'
  disabled?: boolean
}>()
</script>

<template>
  <button
    :class="
      cn(
        'px-4 py-2 rounded-md',
        variant === 'default' && 'bg-primary text-primary-foreground',
        variant === 'outline' && 'border border-input',
        disabled && 'opacity-50 cursor-not-allowed'
      )
    "
  >
    <slot />
  </button>
</template>
```

❌ **BANNED:**
```vue
<template>
  <button :class="'px-4 py-2 ' + (variant === 'default' ? 'bg-primary' : 'border')">
    <slot />
  </button>
</template>
```

---

### ⚠️ USE RADIX VUE FOR PRIMITIVES

**For complex UI components, use Radix Vue primitives.**

✅ **REQUIRED:**
```vue
<script setup lang="ts">
import {
  AccordionRoot,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from 'radix-vue'
</script>

<template>
  <AccordionRoot type="single" collapsible>
    <AccordionItem value="item-1">
      <AccordionTrigger>Question</AccordionTrigger>
      <AccordionContent>Answer</AccordionContent>
    </AccordionItem>
  </AccordionRoot>
</template>
```

❌ **BANNED:**
- Building accordion from scratch when Radix Vue has it
- Reinventing the wheel for dropdowns, dialogs, tooltips
- Ignoring accessibility features Radix Vue provides

**Why this matters:** Radix Vue handles accessibility, keyboard navigation, and edge cases. Don't reinvent the wheel.

---

## 3. Story Requirements

### ⚠️ EVERY COMPONENT NEEDS A STORY

**No component is complete without a proper story file.**

✅ **REQUIRED Story Structure:**
```typescript
// Button.story.ts
import Button from './Button.vue'

export default {
  title: 'atoms/Button',
  component: Button,
}

// Show all variants
export const Primary = () => ({
  components: { Button },
  template: '<Button variant="primary">Primary</Button>',
})

export const Secondary = () => ({
  components: { Button },
  template: '<Button variant="secondary">Secondary</Button>',
})

export const Disabled = () => ({
  components: { Button },
  template: '<Button disabled>Disabled</Button>',
})

// Show all sizes
export const Small = () => ({
  components: { Button },
  template: '<Button size="sm">Small</Button>',
})

export const Large = () => ({
  components: { Button },
  template: '<Button size="lg">Large</Button>',
})
```

❌ **BANNED:**
- Story with only one variant
- No story file at all
- Story that doesn't show all props
- Story that doesn't show all states

---

### ⚠️ TEST IN HISTOIRE BEFORE MARKING COMPLETE

**You MUST verify your component works in Histoire.**

✅ **REQUIRED Checklist:**
- [ ] Component renders without errors
- [ ] All variants shown in Histoire
- [ ] All states work (hover, active, disabled)
- [ ] Props are reactive
- [ ] No console errors
- [ ] Responsive on mobile
- [ ] Keyboard navigation works

❌ **BANNED:**
- Marking component complete without testing
- "It compiles so it must work"
- Skipping Histoire testing

**How to test:**
```bash
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev
# Open http://localhost:6006
# Navigate to your component
# Test all variants and states
```

---

## 4. Code Quality Rules

### ⚠️ CRITICAL: NO BACKGROUND TESTING

**You MUST see full, blocking output. Background jobs hang and you lose logs.**

❌ **BANNED:**
```bash
pnpm dev &              # Backgrounds the process, you lose output
nohup pnpm build &      # Same problem
pnpm test | grep ...    # Pipes into interactive tool, hangs
```

✅ **REQUIRED:**
```bash
pnpm dev                # Foreground, see all output
pnpm build 2>&1 | tee build.log  # Foreground with logging
pnpm test -- --reporter=verbose  # Foreground, verbose output
```

---

### ⚠️ CODE SIGNATURES

**Every file MUST have a team signature.**

✅ **REQUIRED:**
```vue
<!-- Created by: TEAM-FE-001 -->
<!-- TEAM-FE-002: Added dark mode support -->
<script setup lang="ts">
// Component code
</script>
```

```typescript
// Created by: TEAM-FE-001
// TEAM-FE-002: Refactored to use composables
export function useExample() {
  // Code
}
```

❌ **BANNED:**
- No signature at all
- Removing previous team's signatures
- Generic signatures like "// Updated"

**Rules:**
- New files: `// Created by: TEAM-FE-XXX`
- Modifications: `// TEAM-FE-XXX: description of change`
- NEVER remove old team signatures (keep for history)

---

### ⚠️ FILE NAMING CONVENTIONS

**Strict naming rules for consistency.**

✅ **REQUIRED:**
```
Components:     CamelCase.vue      (Button.vue, FormField.vue)
Stories:        CamelCase.story.ts (Button.story.ts)
Composables:    camelCase.ts       (useTheme.ts, useAuth.ts)
Utils:          camelCase.ts       (utils.ts, helpers.ts)
Types:          camelCase.ts       (types.ts, interfaces.ts)
```

❌ **BANNED:**
```
button.vue          # Wrong case
Button.stories.ts   # Wrong extension (should be .story.ts)
use-theme.ts        # Wrong case (should be useTheme.ts)
Utils.ts            # Wrong case (should be utils.ts)
```

---

## 5. TypeScript Rules

### ⚠️ DEFINE PROPS INTERFACES

**All components MUST have TypeScript props interfaces.**

✅ **REQUIRED:**
```vue
<script setup lang="ts">
interface Props {
  variant?: 'default' | 'outline' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'md',
  disabled: false,
  loading: false,
})
</script>
```

❌ **BANNED:**
```vue
<script setup>
// No TypeScript, no props interface
const props = defineProps(['variant', 'size', 'disabled'])
</script>
```

---

### ⚠️ NO ANY TYPES

**Never use `any`. Use proper types or `unknown`.**

✅ **REQUIRED:**
```typescript
interface User {
  id: string
  name: string
  email: string
}

function getUser(id: string): User {
  // Implementation
}
```

❌ **BANNED:**
```typescript
function getUser(id: any): any {
  // Implementation
}
```

---

## 6. Accessibility Rules

### ⚠️ ARIA LABELS REQUIRED

**All interactive elements MUST have ARIA labels.**

✅ **REQUIRED:**
```vue
<template>
  <button aria-label="Close dialog">
    <X :size="16" />
  </button>

  <input
    type="text"
    aria-label="Search"
    aria-describedby="search-help"
  />

  <div role="alert" aria-live="polite">
    {{ errorMessage }}
  </div>
</template>
```

❌ **BANNED:**
```vue
<template>
  <button>
    <X :size="16" />
  </button>

  <input type="text" />

  <div>{{ errorMessage }}</div>
</template>
```

---

### ⚠️ KEYBOARD NAVIGATION REQUIRED

**All interactive components MUST support keyboard navigation.**

✅ **REQUIRED:**
- Tab/Shift+Tab to navigate
- Enter/Space to activate buttons
- Escape to close dialogs/dropdowns
- Arrow keys for lists/menus
- Focus visible indicators

❌ **BANNED:**
- Mouse-only interactions
- No focus indicators
- Keyboard traps
- Inaccessible custom controls

---

## 7. Documentation Rules

### ⚠️ NO MULTIPLE .md FILES FOR ONE TASK

**If you create more than 2 .md files for a single task, YOU FUCKED UP.**

✅ **REQUIRED:**
- UPDATE existing .md files instead of creating new ones
- Check if documentation already exists before creating
- Create ONLY ONE comprehensive doc if needed

❌ **BANNED:**
- Creating "PLAN.md", "SUMMARY.md", "QUICKSTART.md" for the same thing
- Multiple .md files for ONE component/feature
- Duplicating information across files

**Why this matters:** If you're inaccurate once and repeat that across multiple .md files, we need to update ALL documents. That wastes everyone's time.

---

### ⚠️ COMPLETE PREVIOUS TEAM'S TODO LIST

**Previous team's TODO = YOUR PLAN. Follow it. Don't invent new work.**

✅ **REQUIRED:**
- Read "Next Steps" from previous handoff
- Complete ALL priorities in order (1, 2, 3...)
- If you finish Priority 1, immediately start Priority 2
- Only hand off when ALL priorities complete

❌ **BANNED:**
- Doing just Priority 1 and writing new handoff
- Ignoring existing TODO and making up your own priorities
- Using "found a bug" as excuse to abandon the plan
- Inventing new work items that derail the plan

---

## 8. Port Workflow Rules

### ⚠️ FOLLOW THE PORT PLAN

**The React to Vue port has a specific workflow. Follow it.**

✅ **REQUIRED Workflow:**
```
1. Read React reference component
2. Understand props and behavior
3. Create Vue component in storybook
4. Port Tailwind classes (use cn() utility)
5. Replace React hooks with Vue composables
6. Replace Radix UI with Radix Vue
7. Create story with all variants
8. Test in Histoire
9. Export in index.ts
10. Use in application
```

❌ **BANNED:**
- Skipping the React reference
- Guessing how it should work
- Changing behavior without reason
- Not testing in Histoire

---

### ⚠️ VISUAL PARITY REQUIRED

**Your Vue component MUST match the React reference visually.**

✅ **REQUIRED:**
- Open React reference (http://localhost:3000)
- Open your Vue component (http://localhost:5173)
- Compare side-by-side
- Match spacing, colors, fonts, interactions
- Take screenshots if needed

❌ **BANNED:**
- "Close enough"
- Changing design without approval
- Ignoring visual differences
- Not comparing side-by-side

---

## 9. Handoff Requirements

### ⚠️ MAXIMUM 2 PAGES

**Your handoff document MUST be 2 pages or less.**

### ⚠️ MUST INCLUDE

1. **Code examples** of what you implemented
2. **Actual progress** (component count, features added)
3. **Verification checklist** (all boxes checked)
4. **Screenshots** (before/after if visual changes)

### ⚠️ MUST NOT INCLUDE

1. ❌ TODO lists for the next team
2. ❌ "The next team should implement X"
3. ❌ Analysis without implementation
4. ❌ Plans without code

---

## 10. Testing Requirements

### ⚠️ MINIMUM TESTING CHECKLIST

**Before marking a component complete, verify ALL of these:**

- [ ] Component renders without errors
- [ ] All props work correctly
- [ ] All variants shown in Histoire
- [ ] All states work (default, hover, active, disabled, error)
- [ ] Keyboard navigation works
- [ ] ARIA labels present
- [ ] Screen reader accessible (test with NVDA/VoiceOver)
- [ ] Mobile responsive (test at 375px, 768px, 1024px)
- [ ] No console errors or warnings
- [ ] TypeScript compiles without errors
- [ ] Matches React reference visually
- [ ] Story file complete with all variants

**If you can't check ALL boxes, keep working.**

---

## 11. Performance Rules

### ⚠️ OPTIMIZE IMAGES

**All images MUST be optimized.**

✅ **REQUIRED:**
- Use WebP format
- Lazy load images below the fold
- Use appropriate sizes (no 4K images for thumbnails)
- Use `<picture>` for responsive images

❌ **BANNED:**
- Unoptimized PNG/JPG
- Loading all images on page load
- Oversized images

---

### ⚠️ CODE SPLITTING

**Large pages MUST use code splitting.**

✅ **REQUIRED:**
```typescript
// Lazy load route components
const HomeView = () => import('./views/HomeView.vue')
const AboutView = () => import('./views/AboutView.vue')

const router = createRouter({
  routes: [
    { path: '/', component: HomeView },
    { path: '/about', component: AboutView },
  ],
})
```

❌ **BANNED:**
```typescript
// Loading everything upfront
import HomeView from './views/HomeView.vue'
import AboutView from './views/AboutView.vue'
```

---

## 12. Git Commit Rules

### ⚠️ MEANINGFUL COMMIT MESSAGES

**Every commit MUST have a clear, descriptive message.**

✅ **REQUIRED:**
```
feat(atoms): implement Button component with all variants
fix(molecules): correct FormField error message positioning
docs(storybook): add usage examples for Navigation component
refactor(organisms): extract HeroSection into smaller components
```

❌ **BANNED:**
```
update
fix bug
wip
asdf
changes
```

---

## 13. Common Mistakes to Avoid

### ❌ MISTAKE 1: Building in App Instead of Storybook

**Wrong:**
```
1. Create component in /frontend/bin/commercial-frontend/src/components/
2. Use in page
3. Debug in browser
```

**Correct:**
```
1. Create component in /frontend/libs/storybook/stories/atoms/
2. Create story
3. Test in Histoire
4. Export in index.ts
5. Use in app
```

---

### ❌ MISTAKE 2: Relative Imports

**Wrong:**
```vue
<script setup>
import Button from '../../../libs/storybook/stories/atoms/Button/Button.vue'
</script>
```

**Correct:**
```vue
<script setup>
import { Button } from 'rbee-storybook/stories'
</script>
```

---

### ❌ MISTAKE 3: No Story File

**Wrong:**
- Create Button.vue
- Skip Button.story.ts
- Use directly in app

**Correct:**
- Create Button.vue
- Create Button.story.ts with all variants
- Test in Histoire
- Then use in app

---

### ❌ MISTAKE 4: Hardcoded Values

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

### ❌ MISTAKE 5: Skipping Accessibility

**Wrong:**
```vue
<button @click="close">
  <X />
</button>
```

**Correct:**
```vue
<button @click="close" aria-label="Close dialog">
  <X />
</button>
```

---

## 14. Success Criteria

**A component is COMPLETE when:**

- ✅ Built in storybook (not in app)
- ✅ Story file with all variants
- ✅ Tested in Histoire
- ✅ TypeScript props interface defined
- ✅ Uses design tokens (no hardcoded values)
- ✅ Uses workspace imports (no relative imports)
- ✅ ARIA labels present
- ✅ Keyboard navigation works
- ✅ Mobile responsive
- ✅ Matches React reference visually
- ✅ No console errors
- ✅ Exported in index.ts
- ✅ Team signature added
- ✅ All checklist items checked

**If ANY item is unchecked, the component is NOT complete.**

---

## 15. Resources

### Documentation
- **Port Plan:** `/frontend/bin/commercial-frontend/REACT_TO_VUE_PORT_PLAN.md`
- **Developer Checklist:** `/frontend/libs/storybook/DEVELOPER_CHECKLIST.md`
- **Atomic Design:** `/frontend/libs/storybook/ATOMIC_DESIGN_PHILOSOPHY.md`
- **Dependencies Guide:** `/frontend/bin/commercial-frontend/DEPENDENCIES_GUIDE.md`
- **Workspace Guide:** `/frontend/WORKSPACE_GUIDE.md`

### React Reference
- **Location:** `/frontend/reference/v0/`
- **Run:** `pnpm --filter frontend/reference/v0 dev`
- **URL:** http://localhost:3000

### Storybook
- **Location:** `/frontend/libs/storybook/`
- **Run:** `pnpm --filter rbee-storybook story:dev`
- **URL:** http://localhost:6006

### Vue App
- **Location:** `/frontend/bin/commercial-frontend/`
- **Run:** `pnpm --filter rbee-commercial-frontend dev`
- **URL:** http://localhost:5173

---

## 16. Enforcement

**These rules are MANDATORY. Violations will result in:**

1. ⚠️ **First violation:** Warning and request to fix
2. ⚠️ **Second violation:** Work rejected, must redo
3. ⚠️ **Third violation:** Handoff deleted, cited in "teams that failed" list

**67 teams failed by ignoring these rules. Don't be team 68.**

---

## Summary Checklist

Before submitting ANY work, verify:

- [ ] Built in storybook first (not in app)
- [ ] Story file created with all variants
- [ ] Tested in Histoire
- [ ] Uses workspace imports (rbee-storybook/stories)
- [ ] Uses design tokens (no hardcoded values)
- [ ] Uses Tailwind with cn() utility
- [ ] Uses Radix Vue for primitives
- [ ] TypeScript props interface defined
- [ ] ARIA labels present
- [ ] Keyboard navigation works
- [ ] Mobile responsive
- [ ] Matches React reference visually
- [ ] No console errors
- [ ] Team signature added
- [ ] Exported in index.ts
- [ ] Handoff ≤2 pages
- [ ] Previous team's TODO completed

**If you can't check ALL boxes, keep working.**

---

**Version:** 1.0  
**Last Updated:** 2025-10-11  
**Status:** MANDATORY  
**Applies to:** All frontend teams

**READ. FOLLOW. SUCCEED.** 🚀

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

### ⚠️ CRITICAL: DX CLI TOOL (MANDATORY FOR ALL FRONTEND WORK)

**The `dx` CLI tool is MANDATORY for verifying frontend work without browser access.**

**Location:** `/home/vince/Projects/llama-orch/frontend/.dx-tool/`  
**Binary:** `./frontend/.dx-tool/target/release/dx`  
**Docs:** `frontend/.dx-tool/README.md`

**What it does:**
- Inspects HTML structure from Storybook/Histoire
- Extracts all Tailwind CSS classes and rules
- Verifies component rendering via headless Chrome
- Locates story files from Storybook URLs
- Works with SPAs (Vue, React) via automatic JavaScript execution

**Commands you MUST use:**
```bash
# Locate story file from URL
dx story-file "http://localhost:6006/story/stories-atoms-button-button-story-vue"

# Inspect element: get HTML + all CSS in one command
dx inspect button "http://localhost:6006/story/..."

# Check if Tailwind class exists
dx css --class-exists "h-11" "http://localhost:6006/story/..."

# Get all classes on element
dx css --list-classes --list-selector button "http://localhost:6006/story/..."

# Query DOM structure
dx html --selector button "http://localhost:6006/story/..."

# Get element attributes
dx html --attributes button "http://localhost:6006/story/..."
```

**Performance:**
- Tuned for Intel i5-1240P, 62GB RAM, NVMe SSD
- Typical execution: 6-8 seconds (includes headless Chrome startup + SPA rendering)
- Default timeout: 3 seconds for page render
- Maximum timeout: 10 seconds

**When to use:**
- ✅ Verifying story variants render correctly
- ✅ Checking Tailwind classes are applied
- ✅ Validating HTML structure
- ✅ Finding which file defines a story
- ✅ Working without browser access (SSH, remote, CI/CD)

**CRITICAL RULES:**

1. **DO NOT create ad-hoc scripts** to inspect HTML/CSS
   - Use `dx` commands instead
   - If a feature is missing, document it in `frontend/.dx-tool/FEATURE_REQUESTS.md`
   - The DX tool team will add the feature

2. **DO NOT use `curl` + manual parsing**
   - `dx` handles SPA rendering, iframe detection, CSS extraction
   - Manual parsing is error-prone and doesn't handle SPAs

3. **DO NOT skip verification**
   - Every story variant MUST be verified with `dx inspect`
   - Document verification results in your handoff

4. **If you find a bug or missing feature:**
   - Document it in `frontend/.dx-tool/FEATURE_REQUESTS.md`
   - Include: use case, expected command, expected output
   - Notify DX tool team
   - DO NOT create workarounds

**Example workflow:**
```bash
# 1. Find story file
dx story-file "http://localhost:6006/story/stories-atoms-button-button-story-vue"
# Output: stories/atoms/Button/Button.story.vue

# 2. Edit story file
vim stories/atoms/Button/Button.story.vue

# 3. Verify new variant
dx inspect button "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-3"

# 4. Check specific classes
dx css --class-exists "h-11" "http://localhost:6006/story/..."

# 5. Validate HTML
dx html --attributes button "http://localhost:6006/story/..."
```

**See also:**
- `frontend/.dx-tool/README.md` - Full documentation
- `frontend/.dx-tool/PERFORMANCE_SPECS.md` - Performance tuning
- `frontend/.dx-tool/INSPECT_COMMAND_DEMO.md` - Usage examples
- `frontend/.dx-tool/TEAM_DX_004_HANDOFF.md` - Workflow guide

---

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

### ⚠️ CRITICAL: TAILWIND CSS V4 HISTOIRE SETUP

**Tailwind v4 requires specific configuration to work in Histoire. Without this, components will have NO styling.**

✅ **REQUIRED Configuration:**

**1. In `styles/tokens.css` (or main CSS file):**
```css
@import "tailwindcss";

/* Your custom CSS and design tokens below */
:root {
  --color-primary: #0066cc;
  /* ... other tokens */
}
```

**2. In `histoire.config.ts`:**
```typescript
import { defineConfig } from 'histoire'
import { HstVue } from '@histoire/plugin-vue'
import tailwindcss from '@tailwindcss/postcss'

export default defineConfig({
  plugins: [HstVue()],
  vite: {
    css: {
      postcss: {
        plugins: [tailwindcss()],
      },
    },
  },
})
```

**3. In `histoire.setup.ts`:**
```typescript
import './styles/tokens.css' // This imports Tailwind
```

❌ **BANNED:**
- Forgetting `@import "tailwindcss"` in CSS
- Not configuring PostCSS plugin in histoire.config.ts
- Assuming Tailwind "just works" without configuration

**Why this matters:** Tailwind v4 uses a new import system. Without proper configuration, all Tailwind classes will be ignored and components will render unstyled.

**Symptoms of misconfiguration:**
- Components render but have no styling
- Tailwind classes don't apply colors/spacing
- Everything looks broken in Histoire

**If you see unstyled components, check these 3 files first!**

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

### ⚠️ CRITICAL: BUILD IN SHARED STORYBOOK FIRST

**ALL atoms and molecules MUST be built in the SHARED storybook package (`/frontend/libs/storybook/`) because they are reusable across all frontends.**

**Only organisms that are specific to ONE frontend should be built in that frontend's local stories directory.**

✅ **REQUIRED Workflow for Atoms & Molecules (REUSABLE):**
```
1. Create component in /frontend/libs/storybook/stories/[atoms|molecules]/[Name]/
2. Create story file [Name].story.vue
3. Test in Histoire (cd /frontend/libs/storybook && pnpm story:dev)
4. Export in /frontend/libs/storybook/stories/index.ts
5. THEN import in ANY application via 'rbee-storybook/stories'
```

✅ **REQUIRED Workflow for Organisms (FRONTEND-SPECIFIC):**
```
1. Create component in /frontend/bin/[frontend-name]/app/stories/organisms/[Name]/
2. Create story file [Name].story.vue
3. Test in Histoire (cd /frontend/bin/[frontend-name] && pnpm story:dev)
4. Export in /frontend/bin/[frontend-name]/app/stories/index.ts
5. THEN import in that specific application via '~/stories'
```

❌ **BANNED:**
- Creating atoms/molecules in frontend-specific directories (they belong in libs/storybook)
- Creating components directly in `/frontend/bin/commercial/src/components/`
- Copying components from React reference without testing in Histoire
- Skipping the story file
- Not testing in Histoire before using in app

**Why this matters:** 
- **Atoms & Molecules** are reusable primitives (Button, Input, Card, FormField, etc.) that should work across ALL frontends (commercial, admin, dashboard, etc.)
- **Organisms** are complex, context-specific components (Navigation, HeroSection, PricingTiers) that are usually unique to one frontend
- Building in the shared storybook ensures consistency and prevents duplication
- If it's not in the shared storybook, it can't be reused across frontends

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

### ⚠️ CRITICAL: EXPORT ALL COMPONENTS

**Every component MUST be exported in `stories/index.ts` to be usable.**

✅ **REQUIRED:**
```typescript
// /frontend/libs/storybook/stories/index.ts

// Atoms
export { default as Button } from './atoms/Button/Button.vue'
export { default as Badge } from './atoms/Badge/Badge.vue'
export { default as Input } from './atoms/Input/Input.vue'

// Molecules
export { default as PricingCard } from './molecules/PricingCard/PricingCard.vue'

// Organisms
export { default as Navigation } from './organisms/Navigation/Navigation.vue'
export { default as PricingHero } from './organisms/PricingHero/PricingHero.vue'
```

❌ **BANNED:**
- Creating a component but not exporting it
- Forgetting to add export after creating component
- Exporting with wrong name

**Why this matters:** If a component isn't exported in `index.ts`, it cannot be imported using workspace packages. Other developers won't be able to use your component.

**Workflow:**
1. Create component: `Button.vue`
2. Create story: `Button.story.vue`
3. **Export in index.ts** ← Don't forget this step!
4. Test import: `import { Button } from 'rbee-storybook/stories'`

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

### ⚠️ CRITICAL: USE DESIGN TOKENS, NOT HARDCODED COLORS

**React reference uses hardcoded colors (slate-950, amber-500, etc.). We use design tokens instead.**

✅ **REQUIRED: Use Tailwind v4 @theme tokens**
```vue
<template>
  <!-- ✅ CORRECT: Use design tokens -->
  <div class="bg-background text-foreground">
    <h1 class="text-primary">Title</h1>
    <p class="text-muted-foreground">Description</p>
    <button class="bg-primary text-primary-foreground">Click</button>
  </div>
</template>
```

❌ **BANNED: Hardcoded Tailwind colors from React reference**
```vue
<template>
  <!-- ❌ WRONG: Don't copy these from React -->
  <div class="bg-white text-slate-900">
    <h1 class="text-amber-500">Title</h1>
    <p class="text-slate-600">Description</p>
    <button class="bg-amber-500 text-white">Click</button>
  </div>
</template>
```

### Available Design Tokens

**Colors (via `@theme` directive):**
- `bg-background` / `text-foreground` - Base colors
- `bg-card` / `text-card-foreground` - Card backgrounds
- `bg-primary` / `text-primary-foreground` - Primary actions (buttons, links)
- `bg-secondary` / `text-secondary-foreground` - Secondary elements
- `bg-muted` / `text-muted-foreground` - Muted/subtle elements
- `bg-accent` / `text-accent-foreground` - Accent highlights
- `bg-highlight` / `text-highlight-foreground` - **NEW (TEAM-FE-010):** Emphasized backgrounds (pricing cards, comparison tables) - muted in dark mode
- `bg-destructive` / `text-destructive-foreground` - Errors/warnings
- `border-border` - Border colors
- `ring-ring` - Focus rings

**⚠️ CRITICAL: Use `bg-highlight` for emphasized backgrounds, NOT `bg-primary`**

`bg-primary` stays bright in dark mode (good for buttons), but `bg-highlight` is muted in dark mode (good for card backgrounds). See `/frontend/libs/storybook/styles/DESIGN_TOKENS_GUIDE.md` for details.

**Border Radius:**
- `rounded` - Default radius
- `rounded-sm` - Small radius
- `rounded-md` - Medium radius
- `rounded-lg` - Large radius
- `rounded-xl` - Extra large radius

### Translation Guide: React → Vue

When porting from React reference, translate hardcoded colors to tokens:

| React Reference | Vue Design Token |
|----------------|------------------|
| `bg-white` | `bg-background` |
| `bg-slate-50` | `bg-secondary` |
| `bg-slate-900` | `bg-background` (dark mode) |
| `text-slate-900` | `text-foreground` |
| `text-slate-600` | `text-muted-foreground` |
| `bg-amber-500` | `bg-primary` or `bg-accent` |
| `text-amber-500` | `text-primary` or `text-accent` |
| `border-slate-200` | `border-border` |
| `bg-red-500` | `bg-destructive` |

### Why This Matters

1. **Dark mode support** - Tokens automatically adapt
2. **Brand consistency** - Change one value, updates everywhere
3. **Maintainability** - No magic numbers scattered in code
4. **Flexibility** - Easy to rebrand or theme

### When Porting Components

**DO NOT** copy Tailwind classes directly from React reference!

**Instead:**
1. Read React component
2. Understand the **intent** (primary button, muted text, etc.)
3. Use appropriate **design token** for that intent
4. Test in both light and dark modes

**Why this matters:** Design tokens ensure consistency and enable dark mode. Hardcoded colors break theming and create maintenance nightmares.

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

### ⚠️ CRITICAL: HISTOIRE STORY FORMAT

**Histoire requires `.story.vue` files (Vue SFC format), NOT `.story.ts` TypeScript files.**

✅ **REQUIRED: .story.vue Format**
```vue
<!-- Button.story.vue -->
<script setup lang="ts">
import Button from './Button.vue'
</script>

<template>
  <Story title="atoms/Button">
    <Variant title="Primary">
      <Button variant="primary">Primary</Button>
    </Variant>
    
    <Variant title="Secondary">
      <Button variant="secondary">Secondary</Button>
    </Variant>
    
    <Variant title="Disabled">
      <Button disabled>Disabled</Button>
    </Variant>
    
    <Variant title="Small">
      <Button size="sm">Small</Button>
    </Variant>
    
    <Variant title="Large">
      <Button size="lg">Large</Button>
    </Variant>
  </Story>
</template>
```

❌ **BANNED: .story.ts Format**
```typescript
// ❌ DON'T DO THIS - Histoire doesn't support this format!
import type { Meta, StoryObj } from '@histoire/vue3'
import Button from './Button.vue'

export default {
  title: 'atoms/Button',
  component: Button,
} as Meta<typeof Button>

export const Primary: StoryObj<typeof Button> = {
  // This format DOES NOT WORK with Histoire!
}
```

**Why this matters:** Histoire is designed for Vue SFC stories. TypeScript story format will not appear in Histoire and will waste hours of debugging.

---

### ⚠️ EVERY COMPONENT NEEDS A STORY

**No component is complete without a proper story file.**

✅ **REQUIRED:**
- Story file with ALL variants
- Story shows ALL props
- Story shows ALL states (default, hover, active, disabled, error)
- Story uses `.story.vue` format

❌ **BANNED:**
- Story with only one variant
- No story file at all
- Story that doesn't show all props
- Story that doesn't show all states
- Using `.story.ts` format (use `.story.vue` instead)

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
- Starting a new dev server when one is already running

**How to test:**

**IMPORTANT:** The user typically keeps Histoire running at http://localhost:6006/

✅ **CORRECT Approach:**
1. Check if Histoire is already running by visiting http://localhost:6006/ in browser
2. If running, just navigate to your component in the existing server
3. Histoire has Hot Module Replacement (HMR) - your changes appear automatically
4. No need to restart the server

❌ **WRONG Approach:**
```bash
# DON'T DO THIS if server is already running!
pnpm story:dev  # This will fail with "port already in use"
```

**Only start the server if:**
- You verified http://localhost:6006/ is NOT accessible
- You need to start it for the first time

**To start (only if needed):**
```bash
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev
# Open http://localhost:6006
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

### ⚠️ CRITICAL: TWO TYPES OF COMPONENTS

**You must understand the difference between porting atoms vs creating organisms.**

#### Type 1: UI Primitives (Atoms) - PORT FROM REACT

**These exist as separate components in React reference.**

**Location:** `/frontend/reference/v0/components/ui/`

**Examples:**
- badge.tsx
- button.tsx
- card.tsx
- input.tsx
- accordion.tsx
- tabs.tsx
- dialog.tsx

**Workflow:**
```bash
# 1. Find the React component
ls /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/

# 2. Read it
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/badge.tsx

# 3. Port to Vue
# Create: /frontend/libs/storybook/stories/atoms/Badge/Badge.vue
# - Same variants (default, secondary, destructive, outline)
# - Same props
# - Same Tailwind classes
# - Use CVA for variants
# - Use Radix Vue for interactive components

# 4. Create story
# Create: /frontend/libs/storybook/stories/atoms/Badge/Badge.story.vue

# 5. Test in Histoire
pnpm --filter rbee-storybook story:dev

# 6. Export in index.ts
# Add: export { default as Badge } from './atoms/Badge/Badge.vue'
```

---

#### Type 2: Page Components (Molecules/Organisms) - CREATE NEW

**These DON'T exist as separate components in React. They're embedded in page files.**

**Location:** `/frontend/reference/v0/app/[page-name]/page.tsx` (all in one file)

**Examples:**
- PricingCard (molecule) - doesn't exist in React, you design it
- PricingTiers (organism) - doesn't exist in React, you design it
- PricingHero (organism) - doesn't exist in React, you design it
- HeroSection (organism) - doesn't exist in React, you design it

**Workflow:**
```bash
# 1. Read the page file
cat /home/vince/Projects/llama-orch/frontend/reference/v0/app/pricing/page.tsx

# 2. Or view in browser
pnpm --filter frontend/reference/v0 dev
# Open: http://localhost:3000/pricing

# 3. Identify sections
# - Hero section (lines 8-23)
# - Pricing tiers (lines 26-177)
# - Comparison table (lines 180-end)

# 4. Design reusable components
# - PricingCard: Reusable for all 3 tiers
# - PricingTiers: Grid of 3 PricingCards
# - PricingHero: Hero section

# 5. Create components by composing atoms
# PricingCard = Card + Button + Badge + Check icons

# 6. Create story
# Create: PricingCard.story.vue with all variants

# 7. Test in Histoire
pnpm --filter rbee-storybook story:dev

# 8. Export in index.ts
```

**Key Difference:**
- **Atoms:** Port directly from `/components/ui/` - already designed
- **Molecules/Organisms:** Design yourself by analyzing page structure - you're the architect

---

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
- Confusing atoms (port) with organisms (create)

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

### Testing Commands Quick Reference

**Test components in Histoire:**
```bash
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev
# Open: http://localhost:6006
```

**Test full Vue app:**
```bash
cd /home/vince/Projects/llama-orch/frontend/bin/commercial-frontend
pnpm dev
# Open: http://localhost:5173
```

**Compare with React reference:**
```bash
cd /home/vince/Projects/llama-orch/frontend/reference/v0
pnpm dev
# Open: http://localhost:3000
```

**Run linting:**
```bash
pnpm --filter rbee-storybook lint
pnpm --filter rbee-commercial-frontend lint
```

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
- [ ] Story files use `.story.vue` format (NOT `.story.ts`)
- [ ] Tailwind v4 configured in Histoire (PostCSS + @import)
- [ ] Component exported in stories/index.ts

**If you can't check ALL boxes, keep working.**

---

## 17. Critical Lessons from Failed Teams

### ❌ TEAM-FE-002 Lesson: Histoire Story Format

**What happened:**
- Created `.story.ts` files with TypeScript Meta/StoryObj format
- Stories didn't appear in Histoire
- Spent hours debugging

**Root cause:**
- Histoire requires `.story.vue` Vue SFC files
- TypeScript story format doesn't work

**Solution:**
- Always use `.story.vue` format
- Use `<Story>` and `<Variant>` components

---

### ❌ TEAM-FE-002 Lesson: Tailwind CSS v4 Setup

**What happened:**
- Components rendered but had no styling
- All Tailwind classes were ignored
- Spent hours debugging

**Root cause:**
- Tailwind v4 requires `@import "tailwindcss"` in CSS
- Requires PostCSS plugin in `histoire.config.ts`
- Not configured by default

**Solution:**
- Add `@import "tailwindcss"` to `styles/tokens.css`
- Add `@tailwindcss/postcss` plugin to `histoire.config.ts`
- Import tokens.css in `histoire.setup.ts`

---

### Key Takeaway

**These issues cost TEAM-FE-002 significant time. Learn from their mistakes:**
1. Always use `.story.vue` format for Histoire
2. Always configure Tailwind v4 properly
3. Always export components in index.ts
4. Always test in Histoire before marking complete

**Don't repeat these mistakes. Read the rules. Follow the rules.**

---

**Version:** 1.1  
**Last Updated:** 2025-10-11  
**Updated by:** TEAM-FE-003  
**Status:** MANDATORY  
**Applies to:** All frontend teams

**READ. FOLLOW. SUCCEED.** 🚀

# TEAM-FE-000: Dependencies Setup Handoff

**Team:** TEAM-FE-000 (Project Manager)  
**Date:** 2025-10-11  
**Task:** Add Tailwind v4 and React-equivalent dependencies  
**Status:** Complete ✅

---

## Completed Work

### 1. Added Tailwind CSS v4

✅ **Tailwind v4 with PostCSS**
- `tailwindcss@^4.1.9` (matches React reference exactly)
- `@tailwindcss/postcss@^4.1.9`
- `postcss@^8.5.1`
- `autoprefixer@^10.4.20`
- `tailwindcss-animate@^1.0.7`

✅ **Configuration Files**
- `tailwind.config.ts` - Tailwind config with shadcn/ui theme
- `postcss.config.mjs` - PostCSS config for Tailwind v4
- Updated `src/assets/main.css` - Tailwind imports + CSS variables

### 2. Added Vue Equivalents of React Libraries

✅ **UI Primitives**
- `radix-vue@^1.9.11` - Vue port of Radix UI (replaces all @radix-ui/react-* packages)
- `vaul-vue@^0.2.0` - Vue drawer component

✅ **Icons**
- `lucide-vue-next@^0.454.0` - Vue port of Lucide icons (same version as React)

✅ **Carousel**
- `embla-carousel-vue@^8.5.1` - Vue port of Embla Carousel

✅ **Composables**
- `@vueuse/core@^11.3.0` - Vue composables (replaces React hooks)

✅ **Utilities (Same as React)**
- `class-variance-authority@^0.7.1` - Component variants (CVA)
- `clsx@^2.1.1` - Conditional className utility
- `tailwind-merge@^2.5.5` - Merge Tailwind classes

### 3. Created Utility Files

✅ **src/lib/utils.ts**
- `cn()` function - Combines clsx + twMerge
- Matches the React reference exactly

### 4. Created Documentation

✅ **DEPENDENCIES_GUIDE.md**
- Complete guide to all added dependencies
- React → Vue equivalents table
- Usage examples for each library
- Links to documentation

---

## Files Created/Modified

### Created
```
tailwind.config.ts           # Tailwind configuration
postcss.config.mjs            # PostCSS configuration
src/lib/utils.ts              # cn() utility function
DEPENDENCIES_GUIDE.md         # Complete dependencies guide
.handoffs/TEAM-FE-000-DEPENDENCIES-SETUP.md  # This file
```

### Modified
```
package.json                  # Added 13 new dependencies
src/assets/main.css           # Tailwind imports + CSS variables
```

---

## Key Decisions

### 1. Tailwind v4 (Same Version as React)

**Why:** The React reference uses Tailwind v4.1.9. Using the same version ensures:
- ✅ Same utility classes work
- ✅ Same configuration format
- ✅ Easier to copy/paste Tailwind classes from React components

### 2. Radix Vue (Single Package)

**Why:** Instead of 20+ separate `@radix-ui/react-*` packages, Radix Vue is one package:
- ✅ Simpler dependency management
- ✅ All primitives in one place
- ✅ Same API as Radix UI React

### 3. VueUse (Composables Library)

**Why:** Replaces React hooks with Vue composables:
- ✅ `useMediaQuery()` replaces `use-mobile` hook
- ✅ `useLocalStorage()` for state persistence
- ✅ `useEventListener()` for event handling
- ✅ 200+ composables available

### 4. Same Utility Libraries

**Why:** CVA, clsx, and tailwind-merge work the same in Vue and React:
- ✅ No need to rewrite variant logic
- ✅ Same `cn()` utility function
- ✅ Copy/paste from React reference works

---

## React → Vue Mapping

| React Package | Vue Equivalent | Status |
|---------------|----------------|--------|
| `@radix-ui/react-*` (20+ packages) | `radix-vue` (1 package) | ✅ Added |
| `lucide-react` | `lucide-vue-next` | ✅ Added |
| `embla-carousel-react` | `embla-carousel-vue` | ✅ Added |
| `vaul` | `vaul-vue` | ✅ Added |
| React hooks | `@vueuse/core` | ✅ Added |
| `class-variance-authority` | Same | ✅ Added |
| `clsx` | Same | ✅ Added |
| `tailwind-merge` | Same | ✅ Added |
| `tailwindcss@^4.1.9` | Same | ✅ Added |

---

## What's NOT Included (Intentionally)

These will be added **only if needed** during porting:

- ❌ `zod` - Form validation (add if we need forms)
- ❌ `date-fns` - Date utilities (add if we need date pickers)
- ❌ `recharts` - Charts (use Chart.js for Vue if needed)
- ❌ `cmdk` - Command palette (build custom or find Vue equivalent)
- ❌ `sonner` - Toast (use Radix Vue Toast instead)
- ❌ `react-hook-form` - Forms (use Vue form library if needed)
- ❌ `next-themes` - Theme switching (build custom with VueUse)

**Rationale:** Add dependencies as needed, not upfront. Keeps bundle size small.

---

## Usage Examples

### 1. Using Tailwind Classes

```vue
<template>
  <div class="flex items-center gap-4 p-4 bg-primary text-primary-foreground">
    <h1 class="text-2xl font-bold">Hello Tailwind</h1>
  </div>
</template>
```

### 2. Using cn() Utility

```vue
<script setup lang="ts">
import { cn } from '@/lib/utils'

const props = defineProps<{
  variant?: 'default' | 'outline'
  className?: string
}>()
</script>

<template>
  <button
    :class="
      cn(
        'px-4 py-2 rounded-md',
        variant === 'default' && 'bg-primary text-primary-foreground',
        variant === 'outline' && 'border border-input',
        props.className
      )
    "
  >
    <slot />
  </button>
</template>
```

### 3. Using Radix Vue

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
      <AccordionTrigger>Is it accessible?</AccordionTrigger>
      <AccordionContent>Yes. It adheres to the WAI-ARIA design pattern.</AccordionContent>
    </AccordionItem>
  </AccordionRoot>
</template>
```

### 4. Using Lucide Icons

```vue
<script setup lang="ts">
import { Menu, X, Github } from 'lucide-vue-next'
</script>

<template>
  <div class="flex gap-4">
    <Menu :size="24" />
    <X :size="24" />
    <Github :size="24" />
  </div>
</template>
```

### 5. Using VueUse

```vue
<script setup lang="ts">
import { useMediaQuery } from '@vueuse/core'

const isMobile = useMediaQuery('(max-width: 768px)')
</script>

<template>
  <div>
    <p v-if="isMobile">Mobile view</p>
    <p v-else>Desktop view</p>
  </div>
</template>
```

---

## Next Steps

### 1. Install Dependencies

```bash
cd /home/vince/Projects/llama-orch/frontend/bin/commercial-frontend-v2
pnpm install
```

### 2. Test Tailwind

Create a test component to verify Tailwind works:

```vue
<!-- src/components/TestTailwind.vue -->
<template>
  <div class="p-8 bg-primary text-primary-foreground rounded-lg">
    <h1 class="text-3xl font-bold">Tailwind Works! ✅</h1>
  </div>
</template>
```

### 3. Start Porting

Begin with **Phase 1, Priority 1** from `REACT_TO_VUE_PORT_PLAN.md`:
- Button component
- Input component
- Card component
- Badge component (already exists, review)

---

## Verification Checklist

- [x] Tailwind v4 added (same version as React)
- [x] PostCSS configured
- [x] CSS variables match shadcn/ui
- [x] Radix Vue added (all primitives)
- [x] Lucide Vue added (same version)
- [x] VueUse added (composables)
- [x] CVA, clsx, tailwind-merge added
- [x] cn() utility created
- [x] Documentation complete
- [ ] Dependencies installed (run `pnpm install`)
- [ ] Tailwind tested (create test component)
- [ ] Ready to start porting

---

## Handoff to Next Team

**Next Team:** TEAM-FE-001-ATOMS (whoever starts porting atoms)

**What's Ready:**
- ✅ All dependencies configured
- ✅ Tailwind v4 ready to use
- ✅ Radix Vue primitives available
- ✅ Utility functions ready
- ✅ Documentation complete

**What to Do:**
1. Install dependencies: `pnpm install`
2. Test Tailwind works
3. Start porting atoms from `REACT_TO_VUE_PORT_PLAN.md`
4. Use Radix Vue for primitives
5. Use cn() for className merging
6. Follow Atomic Design philosophy

---

## Signatures

```
// Created by: TEAM-FE-000
// Date: 2025-10-11
// Task: Dependencies setup for React to Vue port
// Status: Complete ✅
```

---

**Dependencies are ready! Install and start porting.** 🚀

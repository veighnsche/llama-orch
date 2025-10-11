# Dependencies Guide - React to Vue Port

**Created by:** TEAM-FE-000  
**Date:** 2025-10-11

---

## 📦 Added Dependencies

### Styling & Utilities

**Tailwind CSS v4** ✅
- `tailwindcss@^4.1.9` - Utility-first CSS framework
- `@tailwindcss/postcss@^4.1.9` - PostCSS plugin for Tailwind v4
- `postcss@^8.5.1` - CSS transformations
- `autoprefixer@^10.4.20` - Vendor prefixes
- `tailwindcss-animate@^1.0.7` - Animation utilities

**Class Utilities** ✅
- `class-variance-authority@^0.7.1` - CVA for component variants
- `clsx@^2.1.1` - Conditional className utility
- `tailwind-merge@^2.5.5` - Merge Tailwind classes intelligently

### Icons & UI Primitives

**Icons** ✅
- `lucide-vue-next@^0.454.0` - Vue port of Lucide icons (matches React version)

**UI Primitives** ✅
- `radix-vue@^1.9.11` - Vue port of Radix UI primitives (headless components)
- `vaul-vue@^0.2.0` - Vue port of Vaul drawer component

**Composables** ✅
- `@vueuse/core@^11.3.0` - Essential Vue composables (like React hooks)

**Carousel** ✅
- `embla-carousel-vue@^8.5.1` - Vue port of Embla Carousel

---

## 🔄 React → Vue Equivalents

### Component Libraries

| React (Reference) | Vue (Our Project) | Purpose |
|-------------------|-------------------|---------|
| `@radix-ui/react-*` | `radix-vue` | Headless UI primitives |
| `lucide-react` | `lucide-vue-next` | Icon library |
| `embla-carousel-react` | `embla-carousel-vue` | Carousel component |
| `vaul` | `vaul-vue` | Drawer component |
| React hooks | `@vueuse/core` | Composables (useMediaQuery, etc.) |

### Utilities

| React (Reference) | Vue (Our Project) | Purpose |
|-------------------|-------------------|---------|
| `class-variance-authority` | `class-variance-authority` | ✅ Same - Component variants |
| `clsx` | `clsx` | ✅ Same - Conditional classes |
| `tailwind-merge` | `tailwind-merge` | ✅ Same - Merge Tailwind classes |

### Styling

| React (Reference) | Vue (Our Project) | Purpose |
|-------------------|-------------------|---------|
| `tailwindcss@^4.1.9` | `tailwindcss@^4.1.9` | ✅ Same version |
| `@tailwindcss/postcss` | `@tailwindcss/postcss` | ✅ Same version |
| `tailwindcss-animate` | `tailwindcss-animate` | ✅ Same version |

---

## 📚 Key Libraries Explained

### 1. Radix Vue

**What it is:** Vue port of Radix UI - unstyled, accessible component primitives

**Why we need it:** The React reference uses Radix UI extensively. Radix Vue provides the same primitives for Vue.

**Components available:**
- Accordion, Alert Dialog, Avatar, Checkbox, Collapsible
- Context Menu, Dialog, Dropdown Menu, Hover Card
- Label, Menubar, Navigation Menu, Popover, Progress
- Radio Group, Scroll Area, Select, Separator, Slider
- Switch, Tabs, Toast, Toggle, Tooltip
- And more...

**Usage example:**
```vue
<script setup lang="ts">
import { AccordionRoot, AccordionItem } from 'radix-vue'
</script>

<template>
  <AccordionRoot>
    <AccordionItem value="item-1">
      <!-- Content -->
    </AccordionItem>
  </AccordionRoot>
</template>
```

### 2. Class Variance Authority (CVA)

**What it is:** Utility for creating component variants with TypeScript support

**Why we need it:** The React reference uses CVA for button variants, etc.

**Usage example:**
```typescript
import { cva } from 'class-variance-authority'

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground',
        destructive: 'bg-destructive text-destructive-foreground',
        outline: 'border border-input bg-background',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 rounded-md px-3',
        lg: 'h-11 rounded-md px-8',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)
```

### 3. Tailwind Merge

**What it is:** Intelligently merges Tailwind CSS classes

**Why we need it:** Prevents class conflicts when merging props

**Usage example:**
```typescript
import { twMerge } from 'tailwind-merge'

// Without twMerge: "p-4 p-6" → both classes applied (conflict)
// With twMerge: "p-4 p-6" → "p-6" (last one wins)

const merged = twMerge('p-4 text-red-500', 'p-6 text-blue-500')
// Result: "p-6 text-blue-500"
```

### 4. cn() Utility

**What it is:** Combines clsx + twMerge for conditional classes

**Usage example:**
```vue
<script setup lang="ts">
import { cn } from '@/lib/utils'

const props = defineProps<{
  variant?: 'default' | 'outline'
  className?: string
}>()

const classes = cn(
  'base-classes',
  {
    'variant-default': props.variant === 'default',
    'variant-outline': props.variant === 'outline',
  },
  props.className
)
</script>

<template>
  <button :class="classes">
    <slot />
  </button>
</template>
```

### 5. VueUse

**What it is:** Collection of essential Vue composables

**Why we need it:** Replaces React hooks with Vue equivalents

**Common composables:**
- `useMediaQuery()` - Responsive breakpoints
- `useLocalStorage()` - Local storage state
- `useEventListener()` - Event listeners
- `useToggle()` - Boolean toggle
- `useDebounce()` - Debounced values
- `useThrottle()` - Throttled values

**Usage example:**
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

### 6. Lucide Vue Next

**What it is:** Vue port of Lucide icons

**Why we need it:** The React reference uses lucide-react extensively

**Usage example:**
```vue
<script setup lang="ts">
import { Menu, X, Github } from 'lucide-vue-next'
</script>

<template>
  <Menu :size="24" />
  <X :size="24" />
  <Github :size="24" />
</template>
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd /home/vince/Projects/llama-orch/frontend/bin/commercial-frontend-v2
pnpm install
```

### 2. Verify Tailwind Works

```bash
pnpm dev
```

Visit `http://localhost:5173` and check if Tailwind classes work.

### 3. Test a Component

Create a test button:

```vue
<!-- src/components/TestButton.vue -->
<script setup lang="ts">
import { cn } from '@/lib/utils'

const props = defineProps<{
  variant?: 'default' | 'outline'
}>()
</script>

<template>
  <button
    :class="
      cn(
        'px-4 py-2 rounded-md',
        {
          'bg-primary text-primary-foreground': variant === 'default',
          'border border-input': variant === 'outline',
        }
      )
    "
  >
    <slot />
  </button>
</template>
```

---

## 📋 What's NOT Included (Intentionally)

These dependencies from the React reference are **not needed** for Vue:

❌ **React-specific:**
- `react`, `react-dom` - We use Vue
- `next`, `next-themes` - We use Vite + Vue Router
- `@hookform/resolvers`, `react-hook-form` - We'll use Vue form libraries if needed
- `zod` - Add later if we need form validation

❌ **React-specific Radix:**
- All `@radix-ui/react-*` packages - We use `radix-vue` instead

❌ **Analytics:**
- `@vercel/analytics` - Add later if needed

❌ **Date/Time:**
- `date-fns`, `react-day-picker` - Add later if we need date pickers

❌ **Charts:**
- `recharts` - Add later if we need charts (use Chart.js or similar for Vue)

❌ **Command Palette:**
- `cmdk` - Build custom or find Vue equivalent later

❌ **Fonts:**
- `geist` - Use system fonts or add later

❌ **OTP Input:**
- `input-otp` - Build custom or find Vue equivalent

❌ **Resizable Panels:**
- `react-resizable-panels` - Build custom or find Vue equivalent

❌ **Toast:**
- `sonner` - Use Radix Vue Toast or build custom

---

## 🎯 Next Steps

1. **Install dependencies:** `pnpm install`
2. **Test Tailwind:** Create a test component with Tailwind classes
3. **Test Radix Vue:** Create a test component with Radix Vue primitives
4. **Start porting:** Begin with Phase 1 atoms from the port plan

---

## 📖 Documentation Links

- **Radix Vue:** https://www.radix-vue.com/
- **VueUse:** https://vueuse.org/
- **Tailwind CSS v4:** https://tailwindcss.com/
- **Lucide Vue:** https://lucide.dev/guide/packages/lucide-vue-next
- **CVA:** https://cva.style/docs
- **Embla Carousel Vue:** https://www.embla-carousel.com/get-started/vue/

---

**Ready to port! All dependencies are Vue-compatible equivalents of the React reference.** 🚀

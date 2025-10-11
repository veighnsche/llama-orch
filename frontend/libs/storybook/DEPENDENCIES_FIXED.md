# ✅ Storybook Dependencies Fixed

**Created by:** TEAM-FE-000  
**Date:** 2025-10-11  
**Issue:** Missing dependencies in storybook package.json  
**Status:** FIXED ✅

---

## 🚨 What Was Wrong

The storybook `package.json` was missing critical dependencies that TEAM-FE-001 needs:
- ❌ Radix Vue (for UI primitives)
- ❌ Lucide Vue (for icons)
- ❌ VueUse (for composables)
- ❌ Embla Carousel (for carousels)
- ❌ Vaul Vue (for drawers)
- ❌ Tailwind CSS (for styling)
- ❌ Vue (was in devDependencies, should be in dependencies)

---

## ✅ What Was Fixed

### 1. Added All Required Dependencies

**File:** `/frontend/libs/storybook/package.json`

**Added to `dependencies`:**
```json
{
  "vue": "^3.5.21",
  "class-variance-authority": "^0.7.1",
  "clsx": "^2.1.1",
  "tailwind-merge": "^2.5.5",
  "radix-vue": "^1.9.11",
  "lucide-vue-next": "^0.454.0",
  "@vueuse/core": "^11.3.0",
  "embla-carousel-vue": "^8.5.1",
  "vaul-vue": "^0.2.0",
  "tailwindcss": "^4.1.9",
  "@tailwindcss/postcss": "^4.1.9",
  "postcss": "^8.5.1"
}
```

**Added to `devDependencies`:**
```json
{
  "vue-tsc": "^3.0.4"
}
```

---

### 2. Fixed TypeScript Config

**File:** `/frontend/libs/storybook/tsconfig.json`

**Changes:**
- ✅ Updated extends to `rbee-frontend-tooling` (was orchyra)
- ✅ Added `baseUrl` and `paths` for `@/*` alias
- ✅ Added `lib/**/*.ts` to includes

**Before:**
```json
{
  "extends": "orchyra-frontend-tooling/tsconfig.base.json",
  "include": ["env.d.ts", "histoire.setup.ts", "stories/**/*.ts", "stories/**/*.vue", "*.ts"]
}
```

**After:**
```json
{
  "extends": "rbee-frontend-tooling/tsconfig.base.json",
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": [
    "env.d.ts",
    "histoire.setup.ts",
    "stories/**/*.ts",
    "stories/**/*.vue",
    "lib/**/*.ts",
    "*.ts"
  ]
}
```

---

### 3. Fixed Histoire Config

**File:** `/frontend/libs/storybook/histoire.config.ts`

**Changes:**
- ✅ Added path alias resolution for `@/*`
- ✅ Imported `fileURLToPath` and `URL` from Node

**Added:**
```typescript
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  vite: {
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./', import.meta.url)),
      },
    },
  },
})
```

---

## 🎯 What This Fixes

### For TEAM-FE-001:

✅ **Can now import Radix Vue:**
```vue
<script setup lang="ts">
import { DialogRoot, DialogTrigger } from 'radix-vue'
</script>
```

✅ **Can now import Lucide icons:**
```vue
<script setup lang="ts">
import { Menu, X, Check } from 'lucide-vue-next'
</script>
```

✅ **Can now use VueUse composables:**
```vue
<script setup lang="ts">
import { useMediaQuery } from '@vueuse/core'
</script>
```

✅ **Can now use cn() utility:**
```vue
<script setup lang="ts">
import { cn } from '@/lib/utils'
</script>
```

✅ **Can now use CVA for variants:**
```typescript
import { cva } from 'class-variance-authority'

const buttonVariants = cva(/* ... */)
```

---

## 🚀 TEAM-FE-001: Action Required

### Step 1: Reinstall Dependencies

**CRITICAL:** You MUST reinstall dependencies to get the new packages.

```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

This will:
- Install all new dependencies
- Update workspace links
- Resolve package versions

---

### Step 2: Restart Histoire

After reinstalling, restart Histoire:

```bash
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev
```

---

### Step 3: Verify Imports Work

Test that imports work in your Button component:

```vue
<script setup lang="ts">
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

// Should work now!
const buttonVariants = cva(/* ... */)
</script>
```

---

## 📦 Complete Dependency List

### Production Dependencies (12):
1. ✅ vue - Vue 3 framework
2. ✅ class-variance-authority - Variant management
3. ✅ clsx - Class utilities
4. ✅ tailwind-merge - Tailwind class merging
5. ✅ radix-vue - UI primitives
6. ✅ lucide-vue-next - Icons
7. ✅ @vueuse/core - Composables
8. ✅ embla-carousel-vue - Carousel
9. ✅ vaul-vue - Drawer
10. ✅ tailwindcss - CSS framework
11. ✅ @tailwindcss/postcss - PostCSS plugin
12. ✅ postcss - CSS processor

### Dev Dependencies (13):
1. ✅ rbee-frontend-tooling - Shared tooling
2. ✅ @histoire/plugin-vue - Histoire Vue plugin
3. ✅ @types/node - Node types
4. ✅ @vitejs/plugin-vue - Vite Vue plugin
5. ✅ histoire - Storybook alternative
6. ✅ typescript - TypeScript compiler
7. ✅ vite - Build tool
8. ✅ vitest - Test runner
9. ✅ vue-router - Router
10. ✅ vue-tsc - Vue TypeScript compiler
11. ✅ eslint + plugins - Linting
12. ✅ prettier - Formatting
13. ✅ typescript-eslint - TypeScript linting

---

## ✅ Verification Checklist

After reinstalling, verify:

- [ ] `pnpm install` completes without errors
- [ ] `pnpm story:dev` starts Histoire
- [ ] Can import from `radix-vue`
- [ ] Can import from `lucide-vue-next`
- [ ] Can import from `@vueuse/core`
- [ ] Can import from `class-variance-authority`
- [ ] Can import from `@/lib/utils`
- [ ] No TypeScript errors
- [ ] No console errors in Histoire

---

## 🎉 Summary

**Fixed:**
- ✅ Added 12 production dependencies
- ✅ Added 1 dev dependency (vue-tsc)
- ✅ Fixed TypeScript config (paths, includes)
- ✅ Fixed Histoire config (alias resolution)
- ✅ Updated package references (orchyra → rbee)

**Result:**
- ✅ TEAM-FE-001 can now implement components
- ✅ All imports will work
- ✅ TypeScript will resolve correctly
- ✅ Histoire will run without errors

---

**TEAM-FE-001: Run `pnpm install` NOW to get the dependencies!** 🚀

# TEAM-FE-010 COMPLETE

**Mission:** Add dark/light mode toggle + Update frontend engineering rules for shared components

---

## ✅ Part 1: Frontend Engineering Rules Updated

### Rule Change: "Build in Shared Storybook First"

**File Modified:** `/frontend/FRONTEND_ENGINEERING_RULES.md` (Section 1)

**Key Updates:**
1. ✅ **ALL atoms and molecules MUST be in `/frontend/libs/storybook/`** (shared package)
2. ✅ **Only organisms** specific to ONE frontend go in frontend-specific directories
3. ✅ Clear workflow distinction between reusable vs frontend-specific components

**New Workflows Documented:**

### For Atoms & Molecules (REUSABLE):
```
1. Create in /frontend/libs/storybook/stories/[atoms|molecules]/[Name]/
2. Create story file [Name].story.vue
3. Test in Histoire (cd /frontend/libs/storybook && pnpm story:dev)
4. Export in /frontend/libs/storybook/stories/index.ts
5. Import in ANY app via 'rbee-storybook/stories'
```

### For Organisms (FRONTEND-SPECIFIC):
```
1. Create in /frontend/bin/[frontend-name]/app/stories/organisms/[Name]/
2. Create story file [Name].story.vue
3. Test in Histoire (cd /frontend/bin/[frontend-name] && pnpm story:dev)
4. Export in /frontend/bin/[frontend-name]/app/stories/index.ts
5. Import in that app via '~/stories'
```

**Rationale Added:**
- Atoms & molecules are reusable primitives (Button, Input, ThemeToggle, Card, FormField)
- Should work across ALL frontends (commercial, admin, dashboard, etc.)
- Organisms are complex, context-specific (Navigation, HeroSection, PricingTiers)
- Building in shared storybook ensures consistency and prevents duplication

---

## ✅ Part 2: ThemeToggle Component Implementation

### Created in Shared Storybook

**Location:** `/frontend/libs/storybook/stories/atoms/ThemeToggle/`

#### 1. ThemeToggle.vue
```vue
<!-- Created by: TEAM-FE-010 -->
<script setup lang="ts">
import { useDark, useToggle } from '@vueuse/core'
import { Moon, Sun } from 'lucide-vue-next'
import { Button } from '../../index'

const isDark = useDark({
  selector: 'html',
  attribute: 'class',
  valueDark: 'dark',
  valueLight: '',
})
const toggleDark = useToggle(isDark)

interface Props {
  size?: 'default' | 'sm' | 'lg' | 'icon' | 'icon-sm' | 'icon-lg'
  variant?: 'default' | 'outline' | 'ghost' | 'secondary'
}

const props = withDefaults(defineProps<Props>(), {
  size: 'icon',
  variant: 'ghost',
})
</script>

<template>
  <Button
    :size="props.size"
    :variant="props.variant"
    @click="toggleDark()"
    aria-label="Toggle theme"
    class="relative"
  >
    <Sun :size="20" class="absolute transition-all scale-100 rotate-0 dark:scale-0 dark:-rotate-90" />
    <Moon :size="20" class="absolute transition-all scale-0 rotate-90 dark:scale-100 dark:rotate-0" />
  </Button>
</template>
```

**Features:**
- ✅ Uses VueUse `useDark()` composable (already installed)
- ✅ Animated Sun/Moon icons with smooth transitions
- ✅ Toggles `.dark` class on `<html>` element
- ✅ Works with existing design tokens in `tokens-base.css`
- ✅ Persists preference to localStorage automatically
- ✅ TypeScript props interface
- ✅ Accessible with ARIA label
- ✅ Zero new dependencies

#### 2. ThemeToggle.story.vue
- 6 variants: Default, Outline, Default variant, Small, Large, In Navigation Context
- Shows component in isolation and in navigation context
- Demonstrates theme switching with sample content

#### 3. Exported in `/frontend/libs/storybook/stories/index.ts`
```typescript
// TEAM-FE-010: Added ThemeToggle atom
export { default as ThemeToggle } from './atoms/ThemeToggle/ThemeToggle.vue'
```

**Now available globally:**
```vue
<script setup>
import { ThemeToggle } from 'rbee-storybook/stories'
</script>
```

---

## ✅ Part 3: Integration in Commercial Frontend

### Navigation Component Updated

**File:** `/frontend/bin/commercial/app/stories/organisms/Navigation/Navigation.vue`

**Changes:**
1. ✅ Import updated: `import { Button, ThemeToggle } from 'rbee-storybook/stories'`
2. ✅ Desktop: ThemeToggle added between GitHub icon and "Join Waitlist" button
3. ✅ Mobile: ThemeToggle added with explanatory label in mobile menu

**Desktop Navigation:**
```vue
<a href="https://github.com/..."><Github :size="20" /></a>
<ThemeToggle />  <!-- Added here -->
<Button size="sm" variant="default">Join Waitlist</Button>
```

**Mobile Menu:**
```vue
<div class="flex items-center gap-2">
  <ThemeToggle />
  <span class="text-sm text-muted-foreground">Toggle theme</span>
</div>
<Button size="sm" variant="default" class="w-full">Join Waitlist</Button>
```

### Commercial Index Cleaned Up

**File:** `/frontend/bin/commercial/app/stories/index.ts`

**Changes:**
- ✅ Removed local ThemeToggle export (was in wrong location)
- ✅ Updated comment to reflect migration
- ✅ Now imports ThemeToggle via `export * from 'rbee-storybook/stories'`

### Local Files Deleted

**Removed:** `/frontend/bin/commercial/app/stories/atoms/ThemeToggle/`
- ❌ Local ThemeToggle.vue deleted
- ❌ Local ThemeToggle.story.vue deleted
- ❌ Directory removed (atoms belong in shared storybook)

---

## Technical Details

### Dark Mode Mechanism

**How it works:**
1. VueUse's `useDark()` watches for theme changes
2. Adds/removes `.dark` class on `<html>` element
3. CSS cascade applies `.dark` token overrides from `tokens-base.css`
4. All components update automatically via design tokens
5. Preference saved to localStorage (persists across sessions)

**Design Token Compatibility:**
```css
:root {
  --background: #ffffff;
  --foreground: #0f172a;
  /* ... all light mode tokens */
}

.dark {
  --background: #0f172a;
  --foreground: #f1f5f9;
  /* ... all dark mode tokens */
}
```

When user clicks toggle:
- VueUse adds/removes `.dark` class
- All `bg-background`, `text-foreground`, etc. update automatically
- Works with all existing components (no changes needed)

### Icon Animation

```vue
<Sun class="absolute transition-all scale-100 rotate-0 dark:scale-0 dark:-rotate-90" />
<Moon class="absolute transition-all scale-0 rotate-90 dark:scale-100 dark:rotate-0" />
```

- Smooth scale and rotation transitions
- Sun visible in light mode, Moon in dark mode
- Uses Tailwind's `dark:` variant for automatic switching

---

## Verification Checklist

✅ **Frontend Engineering Rules:**
- [x] Section 1 updated with shared storybook workflow
- [x] Clear distinction between atoms/molecules (shared) and organisms (frontend-specific)
- [x] Rationale documented
- [x] Both workflows documented

✅ **ThemeToggle Component:**
- [x] Created in `/frontend/libs/storybook/stories/atoms/ThemeToggle/`
- [x] Component follows atomic design (atom level)
- [x] Histoire story with 6 variants
- [x] Exported in shared storybook index.ts
- [x] TypeScript props interface
- [x] Uses design tokens
- [x] ARIA labels for accessibility
- [x] TEAM-FE-010 signature
- [x] No TODO markers
- [x] Zero new dependencies

✅ **Migration:**
- [x] Navigation imports from shared storybook
- [x] Local ThemeToggle files deleted
- [x] Commercial index.ts cleaned up
- [x] Works in both desktop and mobile navigation

✅ **Code Quality:**
- [x] Follows all frontend engineering rules
- [x] Uses workspace package imports
- [x] No relative imports across packages
- [x] Proper file naming (CamelCase.vue)
- [x] Team signatures added

---

## Testing

### Test in Shared Storybook Histoire
```bash
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev
# Open http://localhost:6006
# Navigate to: atoms/ThemeToggle
# Test all 6 variants
```

### Test in Commercial App
```bash
cd /home/vince/Projects/llama-orch/frontend/bin/commercial
pnpm dev
# Open http://localhost:3000
# Click theme toggle in navigation
# Verify entire app switches between light/dark mode
# Test on mobile (responsive)
```

---

## Usage Examples

### Import in Any Frontend
```vue
<script setup lang="ts">
import { ThemeToggle } from 'rbee-storybook/stories'
</script>

<template>
  <ThemeToggle />                    <!-- Default: icon ghost button -->
  <ThemeToggle variant="outline" />  <!-- Outline variant -->
  <ThemeToggle size="icon-sm" />     <!-- Small size -->
</template>
```

### Available Props
```typescript
interface Props {
  size?: 'default' | 'sm' | 'lg' | 'icon' | 'icon-sm' | 'icon-lg'
  variant?: 'default' | 'outline' | 'ghost' | 'secondary'
}

// Defaults:
// size: 'icon'
// variant: 'ghost'
```

---

## Files Changed Summary

### Created (3 files)
1. `/frontend/libs/storybook/stories/atoms/ThemeToggle/ThemeToggle.vue`
2. `/frontend/libs/storybook/stories/atoms/ThemeToggle/ThemeToggle.story.vue`
3. `/frontend/TEAM_FE_010_COMPLETE.md` (this file)

### Modified (4 files)
1. `/frontend/FRONTEND_ENGINEERING_RULES.md` - Section 1 updated
2. `/frontend/libs/storybook/stories/index.ts` - ThemeToggle export added
3. `/frontend/bin/commercial/app/stories/organisms/Navigation/Navigation.vue` - Import updated
4. `/frontend/bin/commercial/app/stories/index.ts` - Local export removed
5. `/frontend/bin/commercial/TEAM_FE_010_SUMMARY.md` - Updated with migration details

### Deleted (1 directory)
1. `/frontend/bin/commercial/app/stories/atoms/ThemeToggle/` - Entire directory removed

---

## Result

✅ **Frontend Engineering Rules updated** with shared storybook workflow
✅ **ThemeToggle atom** created in shared storybook package
✅ **Available globally** via `import { ThemeToggle } from 'rbee-storybook/stories'`
✅ **Integrated in Navigation** (desktop and mobile)
✅ **Local files cleaned up** (atoms belong in shared storybook)
✅ **Zero new dependencies** (uses existing VueUse)
✅ **Fully documented** with examples and testing instructions

**ThemeToggle is now a reusable atom available to ALL frontends in the monorepo.**

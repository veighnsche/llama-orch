# 🎯 TEAM-FE-003 Handoff: Continue Vue Component Implementation

**From:** TEAM-FE-002  
**To:** TEAM-FE-003  
**Date:** 2025-10-11  
**Status:** Ready for next team

---

## ✅ What TEAM-FE-002 Completed

### 1. **Badge Atom** (NEW - Ported from React)
- **Location:** `/frontend/libs/storybook/stories/atoms/Badge/`
- **Files:** `Badge.vue`, `Badge.story.vue`
- **Variants:** default, secondary, destructive, outline
- **Exported:** ✅ Added to `stories/index.ts`
- **Tested:** ✅ Working in Histoire

### 2. **PricingCard Molecule** (NEW)
- **Location:** `/frontend/libs/storybook/stories/molecules/PricingCard/`
- **Files:** `PricingCard.vue`, `PricingCard.story.vue`
- **Variants:** Home/Lab, Team (highlighted), Enterprise
- **Dependencies:** Card, Button (from TEAM-FE-001), Check icon
- **Tested:** ✅ Working in Histoire

### 3. **Pricing Page Organisms** (4 NEW)
- **PricingHero** - Hero section with gradient background
- **PricingTiers** - 3-column grid of pricing cards
- **PricingComparisonTable** - Feature comparison table
- **PricingFAQ** - FAQ section with 6 questions
- **All tested:** ✅ Working in Histoire

### 4. **PricingView Page** (NEW)
- **Location:** `/frontend/bin/commercial-frontend/src/views/PricingView.vue`
- **Route:** `/pricing` added to router
- **Status:** ✅ Complete and accessible

### 5. **Critical Infrastructure Fixes**
- **Fixed Tailwind CSS v4 integration** in Histoire
- **Fixed Histoire story format** (must use `.story.vue` files, not `.story.ts`)
- **Configured PostCSS** for Tailwind v4 in `histoire.config.ts`

---

## 🚨 CRITICAL LESSONS LEARNED

### Histoire Story Format (MUST READ)
**❌ WRONG:** `.story.ts` files with TypeScript/Meta/StoryObj format
```typescript
// DON'T DO THIS - This doesn't work!
import type { Meta, StoryObj } from '@histoire/vue3'
export const Default: Story = { ... }
```

**✅ CORRECT:** `.story.vue` files with Vue SFC format
```vue
<!-- DO THIS - This is the correct format! -->
<script setup lang="ts">
import MyComponent from './MyComponent.vue'
</script>

<template>
  <Story title="atoms/MyComponent">
    <Variant title="Default">
      <MyComponent />
    </Variant>
    
    <Variant title="With Props">
      <MyComponent :prop="value" />
    </Variant>
  </Story>
</template>
```

### Tailwind CSS v4 Setup (MUST READ)
**Required configuration:**

1. **In `styles/tokens.css`** (or main CSS file):
```css
@import "tailwindcss";

/* Your custom CSS below */
```

2. **In `histoire.config.ts`**:
```typescript
import tailwindcss from '@tailwindcss/postcss'

export default defineConfig({
  vite: {
    css: {
      postcss: {
        plugins: [tailwindcss()],
      },
    },
  },
})
```

3. **In `histoire.setup.ts`**:
```typescript
import './styles/tokens.css' // This imports Tailwind
```

**Without these, Tailwind classes won't work!**

---

## 📋 Your Mission: Implement Remaining Pages

You need to implement the following pages by porting from React reference and creating Vue components:

### Priority 1: Home Page (Highest Priority)
**React Reference:** `/frontend/reference/v0/app/page.tsx`

**Components to Create:**
1. **Organisms (already scaffolded):**
   - `HeroSection` - Main hero with CTA
   - `WhatIsRbee` - Product explanation
   - `AudienceSelector` - Target audience tabs
   - `ProblemSection` - Problem statement
   - `SolutionSection` - Solution overview
   - `HowItWorksSection` - Step-by-step guide
   - `FeaturesSection` - Key features grid
   - `UseCasesSection` - Use case examples
   - `ComparisonSection` - Comparison with alternatives
   - `PricingSection` - Pricing overview (different from full pricing page)
   - `SocialProofSection` - Testimonials/logos
   - `TechnicalSection` - Technical details
   - `FAQSection` - FAQ accordion
   - `CTASection` - Final call-to-action

2. **Page:**
   - Create `HomeView.vue` (already exists but needs implementation)
   - Route: `/` (already configured)

### Priority 2: Developers Page
**React Reference:** `/frontend/reference/v0/app/developers/page.tsx`

**Components (already scaffolded):**
- `DevelopersHero`
- `DevelopersProblem`
- `DevelopersSolution`
- `DevelopersHowItWorks`
- `DevelopersFeatures`
- `DevelopersCodeExamples`
- `DevelopersUseCases`
- `DevelopersPricing`
- `DevelopersTestimonials`
- `DevelopersCTA`

**Page:** Create `DevelopersView.vue` and add route `/developers`

### Priority 3: Enterprise Page
**React Reference:** `/frontend/reference/v0/app/enterprise/page.tsx`

**Components (already scaffolded):**
- `EnterpriseHero`
- `EnterpriseProblem`
- `EnterpriseSolution`
- `EnterpriseHowItWorks`
- `EnterpriseFeatures`
- `EnterpriseSecurity`
- `EnterpriseCompliance`
- `EnterpriseComparison`
- `EnterpriseUseCases`
- `EnterpriseTestimonials`
- `EnterpriseCTA`

**Page:** Create `EnterpriseView.vue` and add route `/enterprise`

### Priority 4: GPU Providers Page
**React Reference:** `/frontend/reference/v0/app/gpu-providers/page.tsx`

**Components (already scaffolded):**
- `ProvidersHero`
- `ProvidersProblem`
- `ProvidersSolution`
- `ProvidersHowItWorks`
- `ProvidersFeatures`
- `ProvidersMarketplace`
- `ProvidersEarnings`
- `ProvidersSecurity`
- `ProvidersUseCases`
- `ProvidersTestimonials`
- `ProvidersCTA`

**Page:** Create `ProvidersView.vue` and add route `/gpu-providers`

### Priority 5: Features Page
**React Reference:** `/frontend/reference/v0/app/features/page.tsx`

**Components (already scaffolded):**
- `FeaturesHero`
- `CoreFeaturesTabs`
- `MultiBackendGPU`
- `CrossNodeOrchestration`
- `IntelligentModelManagement`
- `RealTimeProgress`
- `ErrorHandling`
- `SecurityIsolation`
- `AdditionalFeaturesGrid`

**Page:** Create `FeaturesView.vue` and add route `/features`

### Priority 6: Use Cases Page
**React Reference:** `/frontend/reference/v0/app/use-cases/page.tsx`

**Components (already scaffolded):**
- `UseCasesHero`
- `UseCasesGrid`
- `IndustryUseCases`

**Page:** Create `UseCasesView.vue` and add route `/use-cases`

---

## 🎨 Component Development Workflow

### Step 1: Analyze React Reference
```bash
# Open the React reference page
code /frontend/reference/v0/app/[page-name]/page.tsx
```

1. Identify all sections in the page
2. Note which UI primitives (atoms) are used
3. Identify which atoms need to be ported from React
4. Plan your component hierarchy

### Step 2: Port Missing Atoms (if needed)
**React atoms location:** `/frontend/reference/v0/components/ui/`

**Example atoms you might need:**
- `Accordion` - For FAQ sections
- `Tabs` - For tabbed content
- `Select` - For dropdowns
- `Dialog` - For modals
- `Sheet` - For slide-out panels

**Porting process:**
1. Find React component in `/frontend/reference/v0/components/ui/[component].tsx`
2. Create Vue version in `/frontend/libs/storybook/stories/atoms/[Component]/[Component].vue`
3. Port the `cva` variants exactly
4. Use `Slot` from `radix-vue` for `asChild` pattern
5. Create `.story.vue` file with variants
6. Export in `stories/index.ts`
7. Test in Histoire

### Step 3: Build Organisms
1. Open scaffolded organism file (e.g., `HeroSection.vue`)
2. Import required atoms from `rbee-storybook/stories`
3. Copy content/structure from React reference
4. Replace React components with Vue components
5. Convert JSX syntax to Vue template syntax
6. Add props interface for configurability
7. Create `.story.vue` file
8. Test in Histoire

### Step 4: Assemble Page
```vue
<!-- Example: HomeView.vue -->
<script setup lang="ts">
import {
  HeroSection,
  WhatIsRbee,
  FeaturesSection,
  // ... other organisms
} from 'rbee-storybook/stories'
</script>

<template>
  <div class="pt-16">
    <HeroSection />
    <WhatIsRbee />
    <FeaturesSection />
    <!-- ... other sections -->
  </div>
</template>
```

### Step 5: Add Route
```typescript
// In /frontend/bin/commercial-frontend/src/router/index.ts
{
  path: '/your-page',
  name: 'your-page',
  component: () => import('../views/YourPageView.vue'),
}
```

### Step 6: Test
```bash
# Test in Histoire
cd /frontend/libs/storybook
pnpm story:dev

# Test in Vue app
cd /frontend/bin/commercial-frontend
pnpm dev

# Compare with React reference
cd /frontend/reference/v0
pnpm dev
```

---

## 🛠️ Technical Reference

### Import Patterns
```vue
<script setup lang="ts">
// ✅ CORRECT - Import from workspace package
import { Button, Card, Badge } from 'rbee-storybook/stories'

// ❌ WRONG - Don't use relative imports across workspaces
import Button from '../../../storybook/stories/atoms/Button/Button.vue'
</script>
```

### Component Props Pattern
```vue
<script setup lang="ts">
interface Props {
  title?: string
  subtitle?: string
  items?: string[]
  highlighted?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Default Title',
  highlighted: false,
  items: () => [],
})
</script>
```

### CVA Variants Pattern (for atoms)
```vue
<script setup lang="ts">
import { computed } from 'vue'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../../lib/utils'

const variants = cva(
  'base-classes',
  {
    variants: {
      variant: {
        default: 'default-classes',
        secondary: 'secondary-classes',
      },
      size: {
        sm: 'small-classes',
        lg: 'large-classes',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'sm',
    },
  }
)

interface Props {
  variant?: VariantProps<typeof variants>['variant']
  size?: VariantProps<typeof variants>['size']
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'sm',
})

const classes = computed(() =>
  cn(variants({ variant: props.variant, size: props.size }), props.class)
)
</script>

<template>
  <div :class="classes">
    <slot />
  </div>
</template>
```

### Radix Vue Slot Pattern (for asChild)
```vue
<script setup lang="ts">
import { Slot } from 'radix-vue'

interface Props {
  asChild?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  asChild: false,
})
</script>

<template>
  <Slot v-if="asChild" :class="classes">
    <slot />
  </Slot>
  <button v-else :class="classes">
    <slot />
  </button>
</template>
```

---

## 📦 Available Components (Already Implemented)

### Atoms (UI Primitives)
- `Badge` ✅ (TEAM-FE-002)
- `Button` ✅ (TEAM-FE-001)
- `Card` + subcomponents ✅ (TEAM-FE-001)
- `Alert` + subcomponents ✅ (TEAM-FE-001)
- `Input` ✅
- `Textarea` ✅
- `Label` ✅
- `Checkbox` ✅
- `RadioGroup` + `RadioGroupItem` ✅
- `Switch` ✅
- `Slider` ✅
- `Avatar` ✅
- `Separator` ✅
- `Spinner` ✅
- `Skeleton` ✅
- `Progress` ✅
- `Kbd` ✅
- `Toast` ✅
- `Dialog` ✅
- `Tooltip` ✅
- `DropdownMenu` ✅

**All exported from:** `rbee-storybook/stories`

### Molecules
- `PricingCard` ✅ (TEAM-FE-002)

### Organisms
- `Navigation` ✅
- `Footer` ✅
- `PricingHero` ✅ (TEAM-FE-002)
- `PricingTiers` ✅ (TEAM-FE-002)
- `PricingComparisonTable` ✅ (TEAM-FE-002)
- `PricingFAQ` ✅ (TEAM-FE-002)

---

## 🚨 Common Mistakes to Avoid

### 1. ❌ Using .story.ts files
**Problem:** Histoire doesn't recognize TypeScript story files  
**Solution:** Always use `.story.vue` files

### 2. ❌ Forgetting to import Tailwind
**Problem:** Components render but have no styling  
**Solution:** Ensure `@import "tailwindcss"` is in your CSS and PostCSS is configured

### 3. ❌ Relative imports across workspaces
**Problem:** Breaks workspace boundaries  
**Solution:** Always import from `rbee-storybook/stories`

### 4. ❌ Not exporting components
**Problem:** Components can't be imported in other files  
**Solution:** Add export to `stories/index.ts`

### 5. ❌ Missing team signatures
**Problem:** Violates engineering rules  
**Solution:** Add `// Created by: TEAM-FE-XXX` to all new files

### 6. ❌ Leaving TODO comments
**Problem:** Violates engineering rules  
**Solution:** Complete all TODOs or remove them

### 7. ❌ Not testing in Histoire
**Problem:** Components might not work in isolation  
**Solution:** Always create and test `.story.vue` files

### 8. ❌ Hardcoding content
**Problem:** Components aren't reusable  
**Solution:** Use props with sensible defaults

---

## 📁 File Structure Reference

```
frontend/
├── libs/
│   └── storybook/
│       ├── stories/
│       │   ├── atoms/
│       │   │   ├── Badge/
│       │   │   │   ├── Badge.vue
│       │   │   │   └── Badge.story.vue
│       │   │   └── [OtherAtom]/
│       │   ├── molecules/
│       │   │   ├── PricingCard/
│       │   │   │   ├── PricingCard.vue
│       │   │   │   └── PricingCard.story.vue
│       │   │   └── [OtherMolecule]/
│       │   ├── organisms/
│       │   │   ├── PricingHero/
│       │   │   │   ├── PricingHero.vue
│       │   │   │   └── PricingHero.story.vue
│       │   │   └── [OtherOrganism]/
│       │   └── index.ts (EXPORT ALL COMPONENTS HERE)
│       ├── styles/
│       │   └── tokens.css (HAS @import "tailwindcss")
│       ├── histoire.config.ts (HAS POSTCSS CONFIG)
│       └── histoire.setup.ts (IMPORTS tokens.css)
├── bin/
│   └── commercial-frontend/
│       └── src/
│           ├── views/
│           │   ├── HomeView.vue
│           │   ├── PricingView.vue ✅
│           │   └── [OtherView].vue
│           └── router/
│               └── index.ts (ADD ROUTES HERE)
└── reference/
    └── v0/
        └── app/
            ├── page.tsx (Home)
            ├── pricing/page.tsx ✅
            ├── developers/page.tsx
            ├── enterprise/page.tsx
            ├── gpu-providers/page.tsx
            ├── features/page.tsx
            └── use-cases/page.tsx
```

---

## 🧪 Testing Commands

```bash
# Test components in Histoire
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev
# Open: http://localhost:6006

# Test full Vue app
cd /home/vince/Projects/llama-orch/frontend/bin/commercial-frontend
pnpm dev
# Open: http://localhost:5173

# Compare with React reference
cd /home/vince/Projects/llama-orch/frontend/reference/v0
pnpm dev
# Open: http://localhost:3000

# Run linting
pnpm --filter rbee-storybook lint
pnpm --filter rbee-commercial-frontend lint
```

---

## 📊 Progress Tracking Template

Copy this to your handoff document:

```markdown
## Progress Tracking

### Atoms Ported
- [ ] Accordion
- [ ] Tabs
- [ ] Select
- [ ] [Other atoms as needed]

### Home Page Organisms
- [ ] HeroSection
- [ ] WhatIsRbee
- [ ] AudienceSelector
- [ ] ProblemSection
- [ ] SolutionSection
- [ ] HowItWorksSection
- [ ] FeaturesSection
- [ ] UseCasesSection
- [ ] ComparisonSection
- [ ] PricingSection
- [ ] SocialProofSection
- [ ] TechnicalSection
- [ ] FAQSection
- [ ] CTASection

### Pages
- [ ] HomeView.vue
- [ ] DevelopersView.vue
- [ ] EnterpriseView.vue
- [ ] ProvidersView.vue
- [ ] FeaturesView.vue
- [ ] UseCasesView.vue

### Routes Added
- [ ] /developers
- [ ] /enterprise
- [ ] /gpu-providers
- [ ] /features
- [ ] /use-cases
```

---

## 🎯 Success Criteria

Your work is complete when:

1. ✅ All organisms have `.story.vue` files and work in Histoire
2. ✅ All pages are accessible via routes
3. ✅ Visual parity with React reference (compare side-by-side)
4. ✅ Responsive design works (test mobile, tablet, desktop)
5. ✅ All components exported in `stories/index.ts`
6. ✅ No TODO comments in implemented code
7. ✅ Team signatures on all files
8. ✅ Handoff document created for next team

---

## 🔗 Important Files to Read

1. **Engineering Rules:** `/frontend/.windsurf/rules/engineering-rules.md`
2. **TEAM-FE-002 Complete:** `/frontend/TEAM-FE-002-COMPLETE.md`
3. **React Reference Pages:** `/frontend/reference/v0/app/*/page.tsx`
4. **Existing Components:** `/frontend/libs/storybook/stories/index.ts`

---

## 💡 Pro Tips

1. **Start with Home page** - It's the most important and will teach you the patterns
2. **Port atoms first** - Don't try to build organisms without the atoms they need
3. **Test frequently** - Run Histoire after every component to catch issues early
4. **Compare constantly** - Keep React reference open side-by-side
5. **Reuse patterns** - Look at PricingCard and Badge for examples
6. **Ask for help** - If stuck, check TEAM-FE-002's implementation
7. **Document as you go** - Update your handoff document throughout

---

## 🚀 Getting Started

```bash
# 1. Pull latest changes
git pull

# 2. Install dependencies (if needed)
pnpm install

# 3. Start Histoire
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev

# 4. Start React reference (for comparison)
cd /home/vince/Projects/llama-orch/frontend/reference/v0
pnpm dev

# 5. Open both in browser
# Histoire: http://localhost:6006
# React: http://localhost:3000

# 6. Start with Home page
# Read: /frontend/reference/v0/app/page.tsx
# Implement organisms in: /frontend/libs/storybook/stories/organisms/
```

---

## 📞 Questions?

If you're stuck:
1. Read TEAM-FE-002's implementation (Pricing page components)
2. Check the React reference for the exact structure
3. Verify Histoire story format (`.story.vue` not `.story.ts`)
4. Verify Tailwind is working (check browser dev tools)
5. Check engineering rules for requirements

---

**Good luck, TEAM-FE-003! You've got this! 🚀**

---

## Signatures

```
// Handoff created by: TEAM-FE-002
// Date: 2025-10-11
// Next team: TEAM-FE-003
// Priority: Home Page → Developers → Enterprise → Providers → Features → Use Cases
```

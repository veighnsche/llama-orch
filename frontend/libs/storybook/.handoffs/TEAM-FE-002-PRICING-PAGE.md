# TEAM-FE-002: Pricing Page Implementation

**Team:** TEAM-FE-002  
**Date:** 2025-10-11  
**From:** TEAM-FE-000 (Project Manager)  
**Status:** READY TO START 🚀

---

## 🚨 CRITICAL: THIS IS A PORT FROM REACT

**YOU ARE PORTING THE PRICING PAGE FROM REACT TO VUE.**

**React Reference:** `/frontend/reference/v0/app/pricing/page.tsx`  
**Run:** `pnpm --filter frontend/reference/v0 dev`  
**URL:** http://localhost:3000/pricing

**ALWAYS compare your work side-by-side with the React reference.**

---

## 🎯 Your Mission

**Build the complete Pricing page.**

**Success = Pricing page deployed, tested, and matching React reference.**

---

## 📊 What TEAM-FE-001 Built

TEAM-FE-001 built 10 components. You can reuse:

### ✅ Available Components:
- **Button** - Use for "Download Now", "Start Trial", "Contact Sales"
- **Card** + subcomponents - Use for pricing tiers
- **Input** - (not needed for Pricing page)
- **Label** - (not needed for Pricing page)
- **Alert** - (not needed for Pricing page)
- **Textarea** - (not needed for Pricing page)
- **Checkbox** - (not needed for Pricing page)
- **Switch** - (not needed for Pricing page)
- **RadioGroup** - (not needed for Pricing page)
- **Slider** - (not needed for Pricing page)

**Note:** Most of TEAM-FE-001's components aren't needed for Pricing. That's okay.

---

## 🚨 CRITICAL: Understanding Your Task

**The React reference has the pricing page in ONE file** (`/frontend/reference/v0/app/pricing/page.tsx`)

**YOU need to:**
1. **Analyze** the page and decide how to split it into components
2. **Identify** which UI primitives (atoms) you need (e.g., Badge)
3. **Look at** the React reference for those UI primitives (`/frontend/reference/v0/components/ui/`)
4. **Port** those UI primitives to Vue
5. **Compose** them into molecules (e.g., PricingCard)
6. **Compose** molecules into organisms (e.g., PricingTiers)
7. **Assemble** organisms into the page

**You are the architect.** You decide how to componentize the page.

---

## 📖 Two Types of Components

### Type 1: UI Primitives (Atoms) - PORT FROM REACT

**These exist in React reference at `/components/ui/`**

**Examples:**
- Badge (`/components/ui/badge.tsx`)
- Button (`/components/ui/button.tsx`) - Already done ✅
- Card (`/components/ui/card.tsx`) - Already done ✅

**Your job:**
1. Find the React component in `/components/ui/`
2. Read the code
3. Port to Vue (same variants, same props, same styling)
4. Test in Histoire

---

### Type 2: Page Components (Molecules/Organisms) - CREATE NEW

**These DON'T exist as separate components in React reference**

**Examples:**
- PricingCard (molecule) - You create this by composing Card + Button + Badge
- PricingTiers (organism) - You create this by composing 3 PricingCards
- PricingHero (organism) - You create this by extracting the hero section

**Your job:**
1. Look at the pricing page in React reference
2. Identify logical sections (hero, tiers, table)
3. Design reusable components for those sections
4. Compose them from atoms
5. Test in Histoire

---

## 🔍 How to Find What You Need

### For UI Primitives (Atoms):

```bash
# List all React UI components
ls /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/

# Output:
# badge.tsx
# button.tsx
# card.tsx
# input.tsx
# ... etc

# Read a specific component
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/badge.tsx
```

### For Page Structure:

```bash
# Read the pricing page
cat /home/vince/Projects/llama-orch/frontend/reference/v0/app/pricing/page.tsx

# Or view it in browser
pnpm --filter frontend/reference/v0 dev
# Open: http://localhost:3000/pricing
```

---

## 📋 Your Assignments (Priority Order)

### Step 1: Analyze Pricing Page (30 min)

```bash
# Start React reference
cd /home/vince/Projects/llama-orch
pnpm --filter frontend/reference/v0 dev

# Open: http://localhost:3000/pricing
# Take screenshots
# Read the source code
cat /home/vince/Projects/llama-orch/frontend/reference/v0/app/pricing/page.tsx
```

**Identify sections:**
- [ ] Hero section (title, subtitle, gradient background)
- [ ] 3 pricing tiers (Free, Team, Enterprise)
- [ ] Feature comparison table
- [ ] Email capture section (at bottom)

**Identify UI primitives needed:**
- [ ] Badge (for "Most Popular" tag) - Look at `/components/ui/badge.tsx`
- [ ] Button (already built by TEAM-FE-001) ✅
- [ ] Card (already built by TEAM-FE-001) ✅
- [ ] Check icon (from lucide-vue-next)
- [ ] X icon (from lucide-vue-next)

**Decide component structure:**
- [ ] How to split into molecules?
- [ ] How to split into organisms?
- [ ] What props does each component need?

---

### Step 2: Build Badge Atom (1 hour)

**Why:** Need for "Most Popular" tag on Team tier

**How to find it:**
```bash
# Look at the React UI components folder
ls /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/

# Read the Badge component
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/badge.tsx
```

**Source:** `/frontend/reference/v0/components/ui/badge.tsx`

**Your job:**
1. Read the React Badge component
2. Understand its variants and props
3. Port it to Vue using CVA
4. Create story file
5. Test in Histoire

**Requirements:**
- Variants: default, secondary, destructive, outline
- Sizes: default, sm, lg
- Use CVA for variants
- Use Tailwind classes

**Deliverables:**
- `/frontend/libs/storybook/stories/atoms/Badge/Badge.vue`
- `/frontend/libs/storybook/stories/atoms/Badge/Badge.story.ts`
- Export in `stories/index.ts`
- Test in Histoire

---

### Step 3: Build PricingCard Molecule (2 hours)

**Why:** Each pricing tier is a PricingCard

**How to design it:**
1. Look at the pricing page in React reference
2. See the 3 pricing tier cards
3. Notice they have: title, price, description, features list, button
4. Design a reusable component that can render all 3 variants

**Composition:**
- Card (from TEAM-FE-001) ✅
- CardHeader, CardTitle, CardDescription, CardContent, CardFooter ✅
- Button (from TEAM-FE-001) ✅
- Badge (you just built) ✅
- Check icons (from lucide-vue-next)

**No React reference for this!** You're creating a NEW molecule by composing atoms.

**Props:**
```typescript
interface PricingCardProps {
  title: string
  price: string
  priceSubtext: string
  description: string
  features: string[]
  buttonText: string
  buttonVariant?: 'default' | 'outline'
  highlighted?: boolean
  badge?: string
}
```

**Deliverables:**
- `/frontend/libs/storybook/stories/molecules/PricingCard/PricingCard.vue`
- `/frontend/libs/storybook/stories/molecules/PricingCard/PricingCard.story.ts`
- Export in `stories/index.ts`
- Test in Histoire with all 3 tier variations

---

### Step 4: Build PricingHero Organism (1 hour)

**Why:** Hero section at top of page

**Content:**
```
Title: "Start Free. Scale When Ready."
Subtitle: "All tiers include the full rbee orchestrator..."
```

**Requirements:**
- Gradient background (from-slate-950 to-slate-900)
- Centered text
- Responsive typography
- Port exact copy from React reference

**Deliverables:**
- `/frontend/libs/storybook/stories/organisms/PricingHero/PricingHero.vue`
- `/frontend/libs/storybook/stories/organisms/PricingHero/PricingHero.story.ts`
- Export in `stories/index.ts`
- Test in Histoire

---

### Step 5: Build PricingTiers Organism (1 hour)

**Why:** Grid of 3 pricing cards

**Composition:**
- 3 PricingCard components
- Grid layout (md:grid-cols-3)
- Middle card highlighted

**Data:**
```typescript
const tiers = [
  {
    title: 'Home/Lab',
    price: '$0',
    priceSubtext: 'forever',
    description: 'For solo developers, hobbyists, homelab enthusiasts',
    features: [
      'Unlimited GPUs',
      'OpenAI-compatible API',
      'Multi-modal support',
      'Community support',
      'GPL open source',
      'CLI access',
      'Rhai scheduler'
    ],
    buttonText: 'Download Now',
    buttonVariant: 'outline'
  },
  {
    title: 'Team',
    price: '€99',
    priceSubtext: '/month',
    description: 'For small teams, startups',
    features: [
      'Everything in Home/Lab',
      'Web UI management',
      'Team collaboration',
      'Priority support',
      'Rhai script templates',
      'Usage analytics',
      'Email support'
    ],
    buttonText: 'Start 30-Day Trial',
    buttonVariant: 'default',
    highlighted: true,
    badge: 'Most Popular'
  },
  {
    title: 'Enterprise',
    price: 'Custom',
    priceSubtext: 'Contact sales',
    description: 'For large teams, enterprises',
    features: [
      'Everything in Team',
      'Dedicated instances',
      'Custom SLAs',
      'White-label option',
      'Enterprise support',
      'On-premises deployment',
      'Professional services'
    ],
    buttonText: 'Contact Sales',
    buttonVariant: 'outline'
  }
]
```

**Deliverables:**
- `/frontend/libs/storybook/stories/organisms/PricingTiers/PricingTiers.vue`
- `/frontend/libs/storybook/stories/organisms/PricingTiers/PricingTiers.story.ts`
- Export in `stories/index.ts`
- Test in Histoire

---

### Step 6: Build FeatureComparisonTable Organism (2 hours)

**Why:** Detailed comparison table below pricing tiers

**Requirements:**
- Responsive table
- 4 columns: Feature, Home/Lab, Team, Enterprise
- Check/X icons for features
- Middle column highlighted (Team)
- Port all features from React reference

**Features to include:**
- Number of GPUs
- OpenAI-compatible API
- Multi-GPU orchestration
- Rhai scheduler
- CLI access
- Web UI
- Team collaboration
- Priority support
- Usage analytics
- Email support
- Dedicated instances
- Custom SLAs
- White-label option
- Enterprise support
- On-premises deployment
- Professional services

**Deliverables:**
- `/frontend/libs/storybook/stories/organisms/FeatureComparisonTable/FeatureComparisonTable.vue`
- `/frontend/libs/storybook/stories/organisms/FeatureComparisonTable/FeatureComparisonTable.story.ts`
- Export in `stories/index.ts`
- Test in Histoire

---

### Step 7: Assemble Pricing Page (1 hour)

**Create:** `/frontend/bin/commercial-frontend/src/views/PricingView.vue`

**Structure:**
```vue
<script setup lang="ts">
import {
  PricingHero,
  PricingTiers,
  FeatureComparisonTable
} from 'rbee-storybook/stories'
</script>

<template>
  <div class="pt-16">
    <PricingHero />
    <PricingTiers />
    <FeatureComparisonTable />
  </div>
</template>
```

**Add route:** `/frontend/bin/commercial-frontend/src/router/index.ts`

```typescript
{
  path: '/pricing',
  name: 'pricing',
  component: () => import('../views/PricingView.vue')
}
```

---

### Step 8: Test & Compare (1 hour)

```bash
# Terminal 1: React reference
pnpm --filter frontend/reference/v0 dev
# Open: http://localhost:3000/pricing

# Terminal 2: Vue app
pnpm --filter rbee-commercial-frontend dev
# Open: http://localhost:5173/pricing

# Compare side-by-side
# Take screenshots
# Fix any differences
```

---

## ✅ Success Criteria

Your work is COMPLETE when ALL of these are checked:

### Components:
- [ ] Badge atom implemented and tested
- [ ] PricingCard molecule implemented and tested
- [ ] PricingHero organism implemented and tested
- [ ] PricingTiers organism implemented and tested
- [ ] FeatureComparisonTable organism implemented and tested

### Page:
- [ ] PricingView.vue created
- [ ] Route added to router
- [ ] Page accessible at /pricing
- [ ] All sections render correctly
- [ ] No console errors

### Quality:
- [ ] Matches React reference visually
- [ ] Responsive on mobile (375px)
- [ ] Responsive on tablet (768px)
- [ ] Responsive on desktop (1024px+)
- [ ] All text matches React reference
- [ ] All colors match React reference
- [ ] All spacing matches React reference
- [ ] Screenshots taken (React vs Vue)

### Documentation:
- [ ] Team signatures added to all files
- [ ] All components exported in index.ts
- [ ] All stories tested in Histoire
- [ ] Handoff document created for TEAM-FE-003

---

## 🚀 Getting Started

### Step 1: Set Up Environment

```bash
cd /home/vince/Projects/llama-orch

# Make sure dependencies are installed
pnpm install

# Start React reference (for comparison)
pnpm --filter frontend/reference/v0 dev
# Opens: http://localhost:3000/pricing

# Start Histoire (for development)
pnpm --filter rbee-storybook story:dev
# Opens: http://localhost:6006

# Start Vue app (for testing)
pnpm --filter rbee-commercial-frontend dev
# Opens: http://localhost:5173
```

---

### Step 2: Read React Reference

```bash
# View the pricing page source
cat /home/vince/Projects/llama-orch/frontend/reference/v0/app/pricing/page.tsx

# View the Badge component
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/badge.tsx
```

---

### Step 3: Start Building

Follow the priority order:
1. Badge atom (1 hour)
2. PricingCard molecule (2 hours)
3. PricingHero organism (1 hour)
4. PricingTiers organism (1 hour)
5. FeatureComparisonTable organism (2 hours)
6. Assemble page (1 hour)
7. Test & compare (1 hour)

**Total estimated time: 9 hours (1-2 days)**

---

## 📚 Required Reading

**BEFORE YOU START, READ THESE:**

1. **Frontend Engineering Rules** (MANDATORY)
   - `/frontend/FRONTEND_ENGINEERING_RULES.md`
   - Section 0: Dependencies
   - Section 1: Component Development Rules
   - Section 2: Component Implementation Rules

2. **Page-First Strategy**
   - `/frontend/PAGE_FIRST_STRATEGY.md`
   - Understand why we're building page-by-page

3. **React to Vue Port Plan**
   - `/frontend/bin/commercial-frontend/REACT_TO_VUE_PORT_PLAN.md`
   - See the overall strategy

---

## 🎨 Component Template

### Badge.vue Template:

```vue
<!-- Created by: TEAM-FE-002 -->
<script setup lang="ts">
import { computed } from 'vue'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-primary text-primary-foreground hover:bg-primary/80',
        secondary: 'border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80',
        destructive: 'border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80',
        outline: 'text-foreground',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
)

interface Props {
  variant?: VariantProps<typeof badgeVariants>['variant']
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
})

const classes = computed(() =>
  cn(badgeVariants({ variant: props.variant }), props.class)
)
</script>

<template>
  <div :class="classes">
    <slot />
  </div>
</template>
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ MISTAKE 1: Not Comparing with React Reference

**Wrong:** Build based on memory or guessing.

**Correct:** Open React reference side-by-side, compare constantly.

---

### ❌ MISTAKE 2: Hardcoding Content

**Wrong:**
```vue
<h1>Start Free. Scale When Ready.</h1>
```

**Correct:**
```vue
<script setup lang="ts">
interface Props {
  title: string
  subtitle: string
}
</script>

<template>
  <h1>{{ title }}</h1>
  <p>{{ subtitle }}</p>
</template>
```

---

### ❌ MISTAKE 3: Not Testing in Histoire

**Wrong:** Build component, skip Histoire, use directly in page.

**Correct:** Build component, test in Histoire, THEN use in page.

---

### ❌ MISTAKE 4: Building All at Once

**Wrong:** Build all components, then assemble page.

**Correct:** Build one component, test it, move to next.

---

## 📊 Progress Tracking

Update this as you complete each step:

- [ ] **Badge atom** - 0% complete
  - [ ] Component implemented
  - [ ] Story created
  - [ ] Tested in Histoire
  - [ ] Exported in index.ts

- [ ] **PricingCard molecule** - 0% complete
  - [ ] Component implemented
  - [ ] Story created with 3 variants
  - [ ] Tested in Histoire
  - [ ] Exported in index.ts

- [ ] **PricingHero organism** - 0% complete
  - [ ] Component implemented
  - [ ] Story created
  - [ ] Tested in Histoire
  - [ ] Exported in index.ts

- [ ] **PricingTiers organism** - 0% complete
  - [ ] Component implemented
  - [ ] Story created
  - [ ] Tested in Histoire
  - [ ] Exported in index.ts

- [ ] **FeatureComparisonTable organism** - 0% complete
  - [ ] Component implemented
  - [ ] Story created
  - [ ] Tested in Histoire
  - [ ] Exported in index.ts

- [ ] **PricingView page** - 0% complete
  - [ ] Page created
  - [ ] Route added
  - [ ] All organisms imported
  - [ ] Tested in browser

- [ ] **Testing & Comparison** - 0% complete
  - [ ] Side-by-side comparison done
  - [ ] Screenshots taken
  - [ ] Visual parity confirmed
  - [ ] Mobile responsive verified

**Overall Progress:** 0/7 steps (0%)

---

## 🎯 Definition of Done

The Pricing page is DONE when:

1. ✅ All components built and tested in Histoire
2. ✅ Page assembled and accessible at /pricing
3. ✅ Matches React reference visually (side-by-side comparison)
4. ✅ Responsive on mobile, tablet, desktop
5. ✅ No console errors or warnings
6. ✅ All text matches React reference exactly
7. ✅ All colors and spacing match React reference
8. ✅ Screenshots taken (React vs Vue)
9. ✅ Team signatures added to all files
10. ✅ Handoff document created for TEAM-FE-003

**If ANY item is unchecked, the page is NOT done.**

---

## 📸 Screenshots Required

Take screenshots of:

1. **React reference** - Full pricing page
2. **Vue implementation** - Full pricing page
3. **Side-by-side comparison** - React vs Vue
4. **Mobile view** - 375px width
5. **Tablet view** - 768px width
6. **Desktop view** - 1024px+ width

Store in: `/frontend/bin/commercial-frontend/.handoffs/TEAM-FE-002-screenshots/`

---

## 🚨 Critical Reminders

1. **Build in storybook FIRST** - Not in the application
2. **Test in Histoire** - Before using in page
3. **Compare with React** - Constantly, not just at the end
4. **One component at a time** - Don't build everything at once
5. **Port exact content** - Don't change copy or design
6. **Read the rules** - `/frontend/FRONTEND_ENGINEERING_RULES.md`

---

## 📞 Need Help?

If you get stuck:

1. **Read the React reference** - The answer is usually there
2. **Check TEAM-FE-001's components** - See how they did it
3. **Look at the dependencies guide** - Shows how to use CVA, Radix Vue, etc.
4. **Check Histoire** - Make sure components render correctly

---

## 🎉 Next Team

When you're done, TEAM-FE-003 will build the Home page.

But that's NOT your concern. Focus on YOUR page: Pricing.

---

## ✅ Final Checklist

Before submitting your handoff:

- [ ] Badge atom complete
- [ ] PricingCard molecule complete
- [ ] PricingHero organism complete
- [ ] PricingTiers organism complete
- [ ] FeatureComparisonTable organism complete
- [ ] PricingView page complete
- [ ] Route added to router
- [ ] Page accessible at /pricing
- [ ] Matches React reference visually
- [ ] Responsive on all screen sizes
- [ ] Screenshots taken (React vs Vue)
- [ ] No console errors
- [ ] Team signatures added
- [ ] This checklist complete

**If you can't check ALL boxes, keep working.**

---

**From:** TEAM-FE-000 (Project Manager)  
**To:** TEAM-FE-002  
**Status:** READY TO START  
**Priority:** CRITICAL - First complete page

**Good luck! You're building the first complete page of the Vue app.** 🚀

---

## Signatures

```
// Handoff created by: TEAM-FE-000
// Date: 2025-10-11
// For: TEAM-FE-002
// Page: Pricing
// Estimated effort: 1-2 days
// Status: Ready to start
```

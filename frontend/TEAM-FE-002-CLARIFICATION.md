# ✅ TEAM-FE-002 Instructions Clarified

**Created by:** TEAM-FE-000  
**Date:** 2025-10-11  
**Issue:** Need to clarify how to find and build components  
**Status:** Complete ✅

---

## 🚨 The Key Point

**The React reference has the pricing page in ONE file, not split into components.**

**TEAM-FE-002 needs to:**
1. **Analyze** the page and decide how to componentize it
2. **Port** UI primitives (atoms) from `/components/ui/` when needed
3. **Create** new molecules/organisms by composing atoms
4. **Assemble** everything into the page

---

## 📖 Two Types of Components

### Type 1: UI Primitives (Atoms) - PORT FROM REACT

**Location:** `/frontend/reference/v0/components/ui/`

**These already exist in React as separate components:**
- badge.tsx
- button.tsx (already ported by TEAM-FE-001 ✅)
- card.tsx (already ported by TEAM-FE-001 ✅)
- input.tsx
- label.tsx
- ... etc

**Workflow:**
```bash
# 1. List available UI components
ls /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/

# 2. Read the one you need
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/badge.tsx

# 3. Port to Vue
# Create: /frontend/libs/storybook/stories/atoms/Badge/Badge.vue

# 4. Test in Histoire
pnpm --filter rbee-storybook story:dev
```

---

### Type 2: Page Components (Molecules/Organisms) - CREATE NEW

**Location:** `/frontend/reference/v0/app/pricing/page.tsx` (all in one file)

**These DON'T exist as separate components in React:**
- PricingCard (molecule)
- PricingTiers (organism)
- PricingHero (organism)
- FeatureComparisonTable (organism)

**Workflow:**
```bash
# 1. Read the pricing page
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
# - FeatureComparisonTable: Comparison table

# 5. Create components by composing atoms
# PricingCard = Card + Button + Badge + Check icons

# 6. Test in Histoire
pnpm --filter rbee-storybook story:dev
```

---

## 🎯 What TEAM-FE-002 Does

### Step 1: Analyze (30 min)

**Read the pricing page:**
```bash
cat /home/vince/Projects/llama-orch/frontend/reference/v0/app/pricing/page.tsx
```

**Identify:**
- [ ] What sections exist? (hero, tiers, table)
- [ ] What UI primitives are needed? (Badge, Button, Card, icons)
- [ ] How to split into reusable components?

---

### Step 2: Port UI Primitives (1 hour)

**Need Badge atom:**

```bash
# 1. Read React Badge
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/badge.tsx

# 2. Port to Vue
# Create: /frontend/libs/storybook/stories/atoms/Badge/Badge.vue
# - Same variants (default, secondary, destructive, outline)
# - Same props
# - Same Tailwind classes
# - Use CVA for variants

# 3. Create story
# Create: /frontend/libs/storybook/stories/atoms/Badge/Badge.story.ts

# 4. Test in Histoire
pnpm --filter rbee-storybook story:dev
```

---

### Step 3: Create Molecules (2 hours)

**Create PricingCard molecule:**

**No React reference for this!** You design it yourself.

**How:**
1. Look at the 3 pricing tiers in the pricing page
2. Notice they all have: title, price, description, features, button
3. Design a reusable PricingCard component
4. Compose from atoms: Card + Button + Badge + Check icons

```vue
<!-- PricingCard.vue -->
<script setup lang="ts">
import { Card, CardHeader, CardTitle, CardContent, CardFooter, Button, Badge } from 'rbee-storybook/stories'
import { Check } from 'lucide-vue-next'

interface Props {
  title: string
  price: string
  features: string[]
  buttonText: string
  highlighted?: boolean
  badge?: string
}
</script>

<template>
  <Card :class="highlighted ? 'border-amber-500' : ''">
    <Badge v-if="badge">{{ badge }}</Badge>
    <CardHeader>
      <CardTitle>{{ title }}</CardTitle>
    </CardHeader>
    <CardContent>
      <div class="text-4xl font-bold">{{ price }}</div>
      <ul>
        <li v-for="feature in features" :key="feature">
          <Check /> {{ feature }}
        </li>
      </ul>
    </CardContent>
    <CardFooter>
      <Button>{{ buttonText }}</Button>
    </CardFooter>
  </Card>
</template>
```

---

### Step 4: Create Organisms (2 hours)

**Create PricingTiers organism:**

**No React reference for this!** You design it yourself.

**How:**
1. Look at the pricing tiers section in the pricing page
2. Notice it's a grid of 3 cards
3. Design a PricingTiers component
4. Compose from molecules: 3 PricingCards

```vue
<!-- PricingTiers.vue -->
<script setup lang="ts">
import { PricingCard } from 'rbee-storybook/stories'

const tiers = [
  {
    title: 'Home/Lab',
    price: '$0',
    features: ['Unlimited GPUs', 'OpenAI-compatible API', ...],
    buttonText: 'Download Now'
  },
  {
    title: 'Team',
    price: '€99',
    features: ['Everything in Home/Lab', 'Web UI', ...],
    buttonText: 'Start Trial',
    highlighted: true,
    badge: 'Most Popular'
  },
  {
    title: 'Enterprise',
    price: 'Custom',
    features: ['Everything in Team', 'Dedicated instances', ...],
    buttonText: 'Contact Sales'
  }
]
</script>

<template>
  <section class="py-24 bg-white">
    <div class="container mx-auto px-4">
      <div class="grid md:grid-cols-3 gap-8">
        <PricingCard
          v-for="tier in tiers"
          :key="tier.title"
          v-bind="tier"
        />
      </div>
    </div>
  </section>
</template>
```

---

### Step 5: Assemble Page (1 hour)

**Create PricingView.vue:**

```vue
<!-- PricingView.vue -->
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

---

## ✅ What Was Clarified in TEAM-FE-002 Kickoff

### Added Section: "Understanding Your Task"

**Explains:**
- React reference has pricing page in ONE file
- YOU need to split it into components
- YOU are the architect

---

### Added Section: "Two Types of Components"

**Type 1: UI Primitives (Atoms) - PORT FROM REACT**
- These exist in `/components/ui/`
- Port them to Vue

**Type 2: Page Components (Molecules/Organisms) - CREATE NEW**
- These DON'T exist as separate components
- You create them by composing atoms

---

### Added Section: "How to Find What You Need"

**For UI Primitives:**
```bash
ls /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/badge.tsx
```

**For Page Structure:**
```bash
cat /home/vince/Projects/llama-orch/frontend/reference/v0/app/pricing/page.tsx
# Or view in browser
```

---

### Updated Step 2: Build Badge Atom

**Added "How to find it" section:**
```bash
ls /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/
cat /home/vince/Projects/llama-orch/frontend/reference/v0/components/ui/badge.tsx
```

**Added "Your job" section:**
1. Read the React Badge component
2. Understand its variants and props
3. Port it to Vue using CVA
4. Create story file
5. Test in Histoire

---

### Updated Step 3: Build PricingCard Molecule

**Added "How to design it" section:**
1. Look at the pricing page in React reference
2. See the 3 pricing tier cards
3. Notice they have: title, price, description, features list, button
4. Design a reusable component that can render all 3 variants

**Added note:**
"No React reference for this! You're creating a NEW molecule by composing atoms."

---

## 📊 Summary

### Before Clarification:
- ❌ Not clear where to find UI components
- ❌ Not clear that pricing page is ONE file
- ❌ Not clear they need to split it themselves
- ❌ Not clear difference between porting vs creating

### After Clarification:
- ✅ Clear: UI primitives are in `/components/ui/`
- ✅ Clear: Pricing page is ONE file, needs splitting
- ✅ Clear: TEAM-FE-002 decides how to componentize
- ✅ Clear: Two types of components (port vs create)
- ✅ Clear: How to find what they need (ls, cat commands)
- ✅ Clear: Workflow for each type

---

## 🎯 Key Takeaways for TEAM-FE-002

1. **UI Primitives (atoms):** Port from `/components/ui/` when needed
2. **Page Components (molecules/organisms):** Create new by composing atoms
3. **You are the architect:** You decide how to split the page
4. **Use ls and cat:** To find and read React components
5. **Test everything in Histoire:** Before using in page

---

**TEAM-FE-002 now has clear instructions on how to find and build components!** 🚀

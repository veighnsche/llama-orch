# Pricing & Use Cases Page Components

**Created by:** TEAM-FE-000  
**Date:** 2025-10-11  
**Issue:** Pricing and Use Cases pages have inline content, not split into components  
**Solution:** Create organism components for all sections

---

## 🎯 Problem

The React reference has two pages with inline content:
- `/app/pricing/page.tsx` - All content inline (389 lines)
- `/app/use-cases/page.tsx` - All content inline (269 lines)

These need to be split into reusable organism components in the storybook.

---

## ✅ Solution: Create Organism Components

### Pricing Page Components (4 organisms)

#### 1. PricingHero
**Location:** `frontend/libs/storybook/stories/organisms/PricingHero/`

**Content:**
- Hero section with headline
- "Start Free. Scale When Ready."
- Subheadline about honest pricing

**Props:**
```typescript
interface Props {
  title: string
  subtitle: string
  description: string
}
```

#### 2. PricingTiers
**Location:** `frontend/libs/storybook/stories/organisms/PricingTiers/`

**Content:**
- 3 pricing cards (Home/Lab, Team, Enterprise)
- Each card with:
  - Title
  - Price
  - Feature list with checkmarks
  - CTA button
  - Target audience

**Props:**
```typescript
interface PricingTier {
  name: string
  price: string
  priceSubtext: string
  features: string[]
  ctaText: string
  ctaVariant: 'primary' | 'outline'
  targetAudience: string
  highlighted?: boolean
  badge?: string
}

interface Props {
  tiers: PricingTier[]
}
```

#### 3. PricingComparisonTable
**Location:** `frontend/libs/storybook/stories/organisms/PricingComparisonTable/`

**Content:**
- Detailed feature comparison table
- 3 columns (Home/Lab, Team, Enterprise)
- Feature rows with checkmarks/crosses
- Highlighted Team column

**Props:**
```typescript
interface ComparisonFeature {
  name: string
  homeLab: boolean | string
  team: boolean | string
  enterprise: boolean | string
}

interface Props {
  title: string
  features: ComparisonFeature[]
}
```

#### 4. PricingFAQ
**Location:** `frontend/libs/storybook/stories/organisms/PricingFAQ/`

**Content:**
- FAQ section with 6 questions
- Each question in a card
- Title + answer

**Props:**
```typescript
interface FAQItem {
  question: string
  answer: string
}

interface Props {
  title: string
  faqs: FAQItem[]
}
```

---

### Use Cases Page Components (3 organisms)

#### 1. UseCasesHero
**Location:** `frontend/libs/storybook/stories/organisms/UseCasesHero/`

**Content:**
- Hero section
- "Built for Those Who Value Independence"
- Description

**Props:**
```typescript
interface Props {
  title: string
  subtitle: string
  description: string
}
```

#### 2. UseCasesGrid
**Location:** `frontend/libs/storybook/stories/organisms/UseCasesGrid/`

**Content:**
- 8 use case cards in grid
- Each card with:
  - Icon
  - Title
  - Scenario
  - Solution
  - Benefit

**Props:**
```typescript
interface UseCase {
  icon: string // Lucide icon name
  iconColor: string
  title: string
  scenario: string
  solution: string
  benefit: string
}

interface Props {
  title: string
  useCases: UseCase[]
}
```

#### 3. IndustryUseCases
**Location:** `frontend/libs/storybook/stories/organisms/IndustryUseCases/`

**Content:**
- Industry-specific solutions
- 6 industry cards
- Each card with title + description

**Props:**
```typescript
interface Industry {
  name: string
  description: string
}

interface Props {
  title: string
  subtitle: string
  industries: Industry[]
}
```

---

## 📋 Updated Scaffolding

I need to create these 7 additional organism components:

### Pricing Page (4 components)
- [ ] PricingHero
- [ ] PricingTiers
- [ ] PricingComparisonTable
- [ ] PricingFAQ

### Use Cases Page (3 components)
- [ ] UseCasesHero
- [ ] UseCasesGrid
- [ ] IndustryUseCases

---

## 🔄 Updated Page Assembly

### PricingView.vue
```vue
<script setup lang="ts">
import {
  Navigation,
  PricingHero,
  PricingTiers,
  PricingComparisonTable,
  PricingFAQ,
  EmailCapture,
  Footer,
} from 'orchyra-storybook/stories'
</script>

<template>
  <div>
    <Navigation />
    <main class="pt-16">
      <PricingHero />
      <PricingTiers />
      <PricingComparisonTable />
      <PricingFAQ />
      <EmailCapture />
    </main>
    <Footer />
  </div>
</template>
```

### UseCasesView.vue
```vue
<script setup lang="ts">
import {
  Navigation,
  UseCasesHero,
  UseCasesGrid,
  IndustryUseCases,
  EmailCapture,
  Footer,
} from 'orchyra-storybook/stories'
</script>

<template>
  <div>
    <Navigation />
    <main class="pt-16">
      <UseCasesHero />
      <UseCasesGrid />
      <IndustryUseCases />
      <EmailCapture />
    </main>
    <Footer />
  </div>
</template>
```

---

## 📊 Updated Component Count

**Before:** 58 organisms  
**After:** 65 organisms (+7)

**Total components:** 128 (49 atoms + 14 molecules + 65 organisms)

---

## ✅ Action Items

1. **Create scaffolding** for 7 new organisms
2. **Update index.ts** exports
3. **Update DEVELOPER_CHECKLIST.md** with new components
4. **Update REACT_TO_VUE_PORT_PLAN.md** with new components

---

**Created by:** TEAM-FE-000  
**Status:** Documented - Ready to scaffold

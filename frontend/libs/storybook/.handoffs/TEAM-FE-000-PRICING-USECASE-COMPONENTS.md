# TEAM-FE-000: Pricing & Use Cases Components Added

**Team:** TEAM-FE-000 (Project Manager)  
**Date:** 2025-10-11  
**Task:** Add missing components for Pricing and Use Cases pages  
**Status:** Complete ✅

---

## 🎯 Problem Identified

The React reference has two pages with **inline content** (not split into components):

1. **`/app/pricing/page.tsx`** - 389 lines, all inline
2. **`/app/use-cases/page.tsx`** - 269 lines, all inline

These need to be split into reusable organism components for the Vue port.

---

## ✅ Solution Implemented

Created **7 new organism components** to properly componentize these pages.

### Pricing Page (4 components)

1. **PricingHero** - Hero section with headline and description
2. **PricingTiers** - 3 pricing cards (Home/Lab, Team, Enterprise)
3. **PricingComparisonTable** - Detailed feature comparison table
4. **PricingFAQ** - FAQ section with 6 questions

### Use Cases Page (3 components)

1. **UseCasesHero** - Hero section
2. **UseCasesGrid** - 8 use case cards (Solo Developer, Small Team, etc.)
3. **IndustryUseCases** - 6 industry-specific solutions

---

## 📁 Files Created

### Component Files
```
frontend/libs/storybook/stories/organisms/
├── PricingHero/
│   ├── PricingHero.vue
│   └── PricingHero.story.ts
├── PricingTiers/
│   ├── PricingTiers.vue
│   └── PricingTiers.story.ts
├── PricingComparisonTable/
│   ├── PricingComparisonTable.vue
│   └── PricingComparisonTable.story.ts
├── PricingFAQ/
│   ├── PricingFAQ.vue
│   └── PricingFAQ.story.ts
├── UseCasesHero/
│   ├── UseCasesHero.vue
│   └── UseCasesHero.story.ts
├── UseCasesGrid/
│   ├── UseCasesGrid.vue
│   └── UseCasesGrid.story.ts
└── IndustryUseCases/
    ├── IndustryUseCases.vue
    └── IndustryUseCases.story.ts
```

### Documentation Files
```
frontend/bin/commercial-frontend-v2/
└── PRICING_USECASE_COMPONENTS.md  # Component specifications

frontend/libs/storybook/
├── SCAFFOLD_PRICING_USECASE.sh    # Scaffolding script
└── .handoffs/
    └── TEAM-FE-000-PRICING-USECASE-COMPONENTS.md  # This file
```

### Updated Files
```
frontend/libs/storybook/stories/index.ts  # Added 7 new exports
```

---

## 📊 Updated Component Count

**Before:** 121 components (49 atoms + 14 molecules + 58 organisms)  
**After:** 128 components (49 atoms + 14 molecules + 65 organisms)

**Added:** +7 organisms

---

## 🔄 Page Assembly

### PricingView.vue (Updated)
```vue
<script setup lang="ts">
import {
  Navigation,
  PricingHero,           // NEW
  PricingTiers,          // NEW
  PricingComparisonTable, // NEW
  PricingFAQ,            // NEW
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

### UseCasesView.vue (Updated)
```vue
<script setup lang="ts">
import {
  Navigation,
  UseCasesHero,      // NEW
  UseCasesGrid,      // NEW
  IndustryUseCases,  // NEW
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

## 📋 Component Specifications

### PricingHero
**Source:** Lines 8-23 of `/app/pricing/page.tsx`

**Props:**
```typescript
interface Props {
  title: string
  subtitle: string
  description: string
}
```

**Content:**
- Gradient background
- Large headline with colored accent
- Description text

---

### PricingTiers
**Source:** Lines 25-177 of `/app/pricing/page.tsx`

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

**Content:**
- 3 pricing cards in grid
- Each card with features list (checkmarks)
- CTA button
- "Most Popular" badge on Team tier

---

### PricingComparisonTable
**Source:** Lines 179-323 of `/app/pricing/page.tsx`

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

**Content:**
- Responsive table
- 3 columns (Home/Lab, Team, Enterprise)
- Feature rows with checkmarks/crosses
- Highlighted Team column

---

### PricingFAQ
**Source:** Lines 325-383 of `/app/pricing/page.tsx`

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

**Content:**
- 6 FAQ cards
- Each card with question + answer
- Light background cards

---

### UseCasesHero
**Source:** Lines 8-22 of `/app/use-cases/page.tsx`

**Props:**
```typescript
interface Props {
  title: string
  subtitle: string
  description: string
}
```

**Content:**
- Gradient background
- "Built for Those Who Value Independence"
- Description

---

### UseCasesGrid
**Source:** Lines 24-198 of `/app/use-cases/page.tsx`

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

**Content:**
- 8 use case cards in 2-column grid
- Each card with:
  - Colored icon
  - Title
  - Scenario (problem)
  - Solution
  - Benefit (checkmark)

**Use Cases:**
1. The Solo Developer
2. The Small Team
3. The Homelab Enthusiast
4. The Enterprise
5. The Freelance Developer
6. The Research Lab
7. The Open Source Maintainer
8. The GPU Provider

---

### IndustryUseCases
**Source:** Lines 201-263 of `/app/use-cases/page.tsx`

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

**Content:**
- 6 industry cards in 3-column grid
- Each card with title + description

**Industries:**
1. Financial Services
2. Healthcare
3. Legal
4. Government
5. Education
6. Manufacturing

---

## ✅ Verification Checklist

- [x] 7 new organism components created
- [x] All components have .vue files
- [x] All components have .story.ts files
- [x] All components have TODO comments
- [x] All components have TEAM-FE-000 signatures
- [x] index.ts updated with 7 new exports
- [x] Documentation created (PRICING_USECASE_COMPONENTS.md)
- [x] Handoff document created (this file)
- [x] Scaffolding script created

---

## 🎯 Next Steps for Developer Teams

### Pricing Page Components

1. **TEAM-FE-XXX: PricingHero**
   - Port from lines 8-23 of `/app/pricing/page.tsx`
   - Implement gradient background
   - Add headline with colored accent
   - Test in Histoire

2. **TEAM-FE-XXX: PricingTiers**
   - Port from lines 25-177 of `/app/pricing/page.tsx`
   - Create 3 pricing cards
   - Add feature lists with checkmarks
   - Add "Most Popular" badge
   - Test in Histoire

3. **TEAM-FE-XXX: PricingComparisonTable**
   - Port from lines 179-323 of `/app/pricing/page.tsx`
   - Create responsive table
   - Add checkmarks/crosses
   - Highlight Team column
   - Test in Histoire

4. **TEAM-FE-XXX: PricingFAQ**
   - Port from lines 325-383 of `/app/pricing/page.tsx`
   - Create 6 FAQ cards
   - Add question + answer
   - Test in Histoire

### Use Cases Page Components

5. **TEAM-FE-XXX: UseCasesHero**
   - Port from lines 8-22 of `/app/use-cases/page.tsx`
   - Implement gradient background
   - Add headline
   - Test in Histoire

6. **TEAM-FE-XXX: UseCasesGrid**
   - Port from lines 24-198 of `/app/use-cases/page.tsx`
   - Create 8 use case cards
   - Add icons (Lucide Vue)
   - Add scenario/solution/benefit
   - Test in Histoire

7. **TEAM-FE-XXX: IndustryUseCases**
   - Port from lines 201-263 of `/app/use-cases/page.tsx`
   - Create 6 industry cards
   - Add title + description
   - Test in Histoire

---

## 📚 References

- **React source:** `/frontend/reference/v0/app/pricing/page.tsx`
- **React source:** `/frontend/reference/v0/app/use-cases/page.tsx`
- **Component specs:** `/frontend/bin/commercial-frontend-v2/PRICING_USECASE_COMPONENTS.md`
- **Developer checklist:** `/frontend/libs/storybook/DEVELOPER_CHECKLIST.md`

---

## Signatures

```
// Created by: TEAM-FE-000
// Date: 2025-10-11
// Task: Add Pricing and Use Cases page components
// Status: Scaffolding complete ✅
```

---

**Total components now: 128 (was 121)**  
**Ready for implementation!** 🚀

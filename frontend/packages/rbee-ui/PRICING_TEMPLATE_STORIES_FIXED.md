# PricingTemplate Stories Fixed - Props Import Pattern

## Problem

Stories were duplicating props inline instead of importing from page props files, creating:
- **Maintenance nightmare**: Update props in 2 places
- **Drift risk**: Page and story can get out of sync
- **Code bloat**: 200+ lines of duplicated props per story

## Solution

Import props from page files - **NEVER DUPLICATE**.

## Before (WRONG ❌)

```typescript
// PricingTemplate.stories.tsx
import { Layers, Shield, Unlock, Zap } from 'lucide-react'
import { pricingHero } from '@rbee/ui/assets'

export const OnHomePage: Story = {
  args: {
    kickerBadges: [
      { icon: <Unlock className="h-3.5 w-3.5" />, label: 'Open source' },
      { icon: <Zap className="h-3.5 w-3.5" />, label: 'OpenAI-compatible' },
      // ... 200+ lines of duplicated props
    ],
    tiers: [
      { title: 'Home/Lab', price: '€0', features: [...] },
      { title: 'Team', price: '€99', features: [...] },
      { title: 'Enterprise', price: 'Custom', features: [...] },
    ],
    editorialImage: { src: pricingHero, alt: '...' },
    footer: { mainText: '...', subText: '...' },
  },
}

export const OnPricingPage: Story = {
  args: {
    // Another 150+ lines of duplicated props
  },
}

export const OnDevelopersPage: Story = {
  args: {
    // Another 150+ lines of duplicated props
  },
}
```

**Total: 500+ lines of duplicated props**

## After (CORRECT ✅)

```typescript
// PricingTemplate.stories.tsx
import type { Meta, StoryObj } from '@storybook/react'
import {
  pricingTemplateProps,
  developersPricingTemplateProps,
} from '@rbee/ui/pages'
import { pricingTemplateProps as homePricingTemplateProps } from '@rbee/ui/pages/HomePage'
import { PricingTemplate } from './PricingTemplate'

const meta = {
  title: 'Templates/PricingTemplate',
  component: PricingTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof PricingTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * PricingTemplate as used on the Home page
 * - Kicker badges (Open source, OpenAI-compatible, Multi-GPU, No feature gates)
 * - Editorial image below tiers
 * - Full featured variant
 */
export const OnHomePage: Story = {
  args: homePricingTemplateProps,
}

/**
 * PricingTemplate as used on the Pricing page
 * - No kicker badges
 * - No editorial image
 * - Pricing-focused footer text
 */
export const OnPricingPage: Story = {
  args: pricingTemplateProps,
}

/**
 * PricingTemplate as used on the Developers page
 * - No kicker badges
 * - No editorial image
 * - Developer-focused footer text
 */
export const OnDevelopersPage: Story = {
  args: developersPricingTemplateProps,
}
```

**Total: 50 lines, all props imported**

## Pattern to Follow

### 1. Export Props from Page Index

```typescript
// src/pages/PricingPage/index.ts
export {
  pricingTemplateContainerProps,
  pricingTemplateProps,
  // ... other props
} from "./PricingPageProps";
export { default as PricingPage } from "./PricingPage";
```

### 2. Import Props in Story

```typescript
// src/templates/XTemplate/XTemplate.stories.tsx
import type { Meta, StoryObj } from '@storybook/react'
import { xTemplateProps } from '@rbee/ui/pages'  // Import from barrel
import { XTemplate } from './XTemplate'

export const OnHomePage: Story = {
  args: xTemplateProps,  // Use imported props
}
```

### 3. Multiple Page Usage

```typescript
// If used on multiple pages, import all variants
import {
  pricingTemplateProps,           // From PricingPage
  developersPricingTemplateProps, // From DevelopersPage
} from '@rbee/ui/pages'
import { pricingTemplateProps as homePricingTemplateProps } from '@rbee/ui/pages/HomePage'

export const OnHomePage: Story = {
  args: homePricingTemplateProps,
}

export const OnPricingPage: Story = {
  args: pricingTemplateProps,
}

export const OnDevelopersPage: Story = {
  args: developersPricingTemplateProps,
}
```

## Reference: EmailCapture Pattern

See `src/templates/EmailCapture/EmailCapture.stories.tsx` for the gold standard:

```typescript
import { 
  emailCaptureProps,
  featuresEmailCaptureProps,
  useCasesEmailCaptureProps,
  pricingEmailCaptureProps,
  developersEmailCaptureProps,
  enterpriseEmailCaptureProps,
} from '@rbee/ui/pages'

export const OnHomePage: Story = {
  args: emailCaptureProps,
}

export const OnFeaturesPage: Story = {
  args: featuresEmailCaptureProps,
}

// ... etc
```

**Zero duplication. Perfect.**

## Benefits

✅ **Single source of truth** - Props defined once in page files  
✅ **No drift** - Story always matches page  
✅ **Easy updates** - Change props once, propagates everywhere  
✅ **Clean stories** - 10 lines instead of 200+  
✅ **Type safety** - Import ensures props match template interface  

## Rule

**NEVER write inline props in stories. ALWAYS import from page files.**

If you find yourself typing `args: { ... }` with more than one line, **STOP** and import instead.

---

**This is the pattern. Follow it.**

# FAQSection Component

**Created by:** TEAM-086

## Overview

`FAQSection` is a reusable, fully-featured FAQ organism with search, category filtering, accordion UI, and optional support card. It consolidates all FAQ functionality across the application.

## Features

- ✅ **Search & Filter**: Real-time search with category filters
- ✅ **Accordion UI**: Collapsible Q&A with smooth animations
- ✅ **Grouped Display**: FAQs organized by category with separators
- ✅ **Support Card**: Optional sticky sidebar with image, links, and CTA
- ✅ **JSON-LD Schema**: SEO-optimized FAQPage structured data
- ✅ **Fully Accessible**: ARIA labels, keyboard navigation, screen reader support
- ✅ **Customizable**: All content, styling, and behavior configurable via props

## Usage

### Basic (Homepage - Technical FAQs)

```tsx
import { FAQSection } from '@/components/organisms/FaqSection/FaqSection'

export default function HomePage() {
  return <FAQSection />
}
```

### Custom (Pricing Page - Pricing FAQs)

```tsx
import { FAQSection } from '@/components/organisms/FaqSection/FaqSection'
import { pricingFaqItems, pricingCategories } from '@/components/organisms/FaqSection/pricing-faqs'

export default function PricingPage() {
  return (
    <FAQSection
      title="Pricing FAQs"
      subtitle="Answers on licensing, upgrades, trials, and payments."
      badgeText="Pricing • Plans & Billing"
      categories={pricingCategories}
      faqItems={pricingFaqItems}
      showSupportCard={false}
      jsonLdEnabled={true}
    />
  )
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `title` | `string` | `'rbee FAQ'` | Section heading |
| `subtitle` | `string` | `'Quick answers...'` | Section description |
| `badgeText` | `string` | `'Support • Self-hosted AI'` | Optional badge above title |
| `categories` | `string[]` | `['Setup', 'Models', ...]` | Category filter chips |
| `faqItems` | `FAQItem[]` | `defaultFaqItems` | Array of FAQ items |
| `showSupportCard` | `boolean` | `true` | Show/hide right sidebar support card |
| `supportCardImage` | `string` | `'/images/faq-beehive.png'` | Support card image path |
| `supportCardImageAlt` | `string` | `'...'` | Image alt text |
| `supportCardTitle` | `string` | `'Still stuck?'` | Support card heading |
| `supportCardLinks` | `Array<{label, href}>` | `[...]` | List of support links |
| `supportCardCTA` | `{label, href}` | `{...}` | Primary CTA button |
| `jsonLdEnabled` | `boolean` | `true` | Enable/disable JSON-LD schema |

## FAQItem Interface

```tsx
interface FAQItem {
  value: string          // Unique ID for accordion
  question: string       // Question text
  answer: React.ReactNode // Answer (supports JSX for rich formatting)
  category: string       // Category for filtering/grouping
}
```

## Creating Custom FAQ Sets

1. Create a new file (e.g., `my-faqs.tsx`) in the `FaqSection` directory
2. Export categories and items:

```tsx
import { FAQItem } from './FaqSection'

export const myCategories = ['Category1', 'Category2']

export const myFaqItems: FAQItem[] = [
  {
    value: 'item-1',
    question: 'Your question?',
    answer: (
      <div className="prose prose-neutral dark:prose-invert max-w-none prose-p:my-2">
        <p>Your answer with <code>code</code> and formatting.</p>
      </div>
    ),
    category: 'Category1',
  },
  // ... more items
]
```

3. Use in your page:

```tsx
import { FAQSection } from '@/components/organisms/FaqSection/FaqSection'
import { myFaqItems, myCategories } from '@/components/organisms/FaqSection/my-faqs'

export default function MyPage() {
  return (
    <FAQSection
      title="My FAQs"
      categories={myCategories}
      faqItems={myFaqItems}
    />
  )
}
```

## Consolidation History

**Previous State:**
- `FaqSection.tsx` - Homepage technical FAQs (better design)
- `Pricing/pricing-faq.tsx` - Pricing-specific FAQs (duplicate component)

**Consolidation (TEAM-086):**
1. Made `FaqSection` accept props for full customization
2. Extracted pricing FAQs to `pricing-faqs.tsx` data file
3. Updated pricing page to use `FAQSection` with pricing props
4. Deleted duplicate `pricing-faq.tsx` component
5. Removed export from `organisms/index.ts`

**Result:** Single, reusable FAQ component with multiple content sets.

## Design Decisions

- **Better Design Choice**: Used `FaqSection` (homepage version) as base because:
  - Grouped display by category with visual separators
  - Support card with image and links
  - Expand/Collapse all functionality
  - Better visual hierarchy and spacing
  - More polished animations and interactions

- **Pricing Variant**: Configured with `showSupportCard={false}` for cleaner pricing page layout

## Files

- `FaqSection.tsx` - Main component
- `pricing-faqs.tsx` - Pricing-specific FAQ data
- `README.md` - This documentation

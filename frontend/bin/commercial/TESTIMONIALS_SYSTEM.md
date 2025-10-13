# Testimonials System

## Summary

Centralized testimonials system with single source of truth for quotes, stats, and reusable UI components following Atomic Design principles.

## Architecture

### Data Layer (`data/testimonials.ts`)

**Single source of truth** for all testimonial content:

```typescript
export const TESTIMONIALS: Testimonial[] = [
  // Provider testimonials (sector: 'provider')
  { id: 'marcus', name: 'Marcus T.', role: 'Gaming PC Owner', sector: 'provider', payout: '€160/mo', ... },
  { id: 'sarah', name: 'Sarah K.', role: 'Homelab Enthusiast', sector: 'provider', payout: '€420/mo', ... },
  { id: 'david', name: 'David L.', role: 'Former Miner', sector: 'provider', payout: '€780/mo', ... },
  
  // Enterprise testimonials (sectors: 'finance', 'healthcare', 'legal')
  { id: 'klaus', name: 'Dr. Klaus M.', role: 'CTO', org: 'European Bank', sector: 'finance', ... },
  { id: 'anna', name: 'Anna S.', role: 'DPO', org: 'Healthcare Provider', sector: 'healthcare', ... },
  { id: 'michael', name: 'Michael R.', role: 'Managing Partner', org: 'Law Firm', sector: 'legal', ... },
]

export const TESTIMONIAL_STATS: TestimonialStat[] = [
  { id: 'gdpr', value: '100%', label: 'GDPR Compliant' },
  { id: 'retention', value: '7 Years', label: 'Audit Retention' },
  { id: 'violations', value: 'Zero', label: 'Compliance Violations' },
  { id: 'support', value: '24/7', label: 'Enterprise Support' },
]
```

**Taxonomy**: Each testimonial includes `sector` for filtering:
- `provider` - GPU providers earning payouts
- `finance` - Financial institutions
- `healthcare` - Healthcare organizations
- `legal` - Law firms
- `government` - Government entities

### Component Hierarchy

```
Atom: RatingStars
├── Props: { rating, size: 'sm'|'md' }
├── Accessibility: aria-label with rating value
└── Styling: Lucide Star icons, fill-primary for active

Molecule: TestimonialCard
├── Props: { t: Testimonial, showVerified?, delayIndex? }
├── Structure:
│   ├── Sector icon chip (Building2, Heart, Scale, Cpu)
│   ├── Verified badge ("Verified payout" or "Verified customer")
│   ├── Author/Role/Org/Payout
│   ├── RatingStars (if rating present)
│   └── Blockquote with cite
├── Accessibility: aria-describedby linking author/role
└── Styling: rounded-2xl gradient card, hover:shadow-md

Molecule: StatTile
├── Props: { value, label, helpText? }
├── Structure: Large value + small label
├── Accessibility: role="group", aria-label
└── Styling: rounded-xl border card, hover effects

Organism: TestimonialsRail
├── Props: { sectorFilter?, limit?, layout: 'grid'|'carousel', showStats? }
├── Behavior:
│   ├── Filters TESTIMONIALS by sector
│   ├── Grid: 3-column grid on desktop
│   ├── Carousel: horizontal scroll on mobile, grid on desktop
│   └── Stats: optional TESTIMONIAL_STATS display
├── Accessibility:
│   ├── aria-labelledby for section
│   ├── Carousel: tabIndex=0, focus ring, sr-only instructions
│   └── Staggered animations with delay
└── Styling: animate-in fade-in-50, responsive layouts
```

## Component APIs

### RatingStars (Atom)

```tsx
<RatingStars rating={5} size="sm" />
```

**Props**:
- `rating: number` - 1-5 stars
- `size?: 'sm' | 'md'` - Icon size (default: 'md')
- `className?: string` - Additional classes

**Accessibility**: Includes `aria-label` with rating value and `sr-only` text.

### TestimonialCard (Molecule)

```tsx
// New API (recommended)
<TestimonialCard t={testimonial} showVerified delayIndex={0} />

// Legacy API (backward compatible)
<TestimonialCard
  name="Marcus T."
  role="Gaming PC Owner"
  quote="..."
  rating={5}
  verified
/>
```

**New Props**:
- `t: Testimonial` - Testimonial data from `TESTIMONIALS`
- `showVerified?: boolean` - Show verified badge (default: true)
- `delayIndex?: number` - Animation delay multiplier (default: 0)
- `className?: string` - Additional classes

**Features**:
- Sector-specific icons (Building2, Heart, Scale, Cpu)
- Verified badges ("Verified payout" for providers, "Verified customer" for enterprise)
- Automatic payout display for provider testimonials
- Organization display for enterprise testimonials
- RatingStars integration
- Proper blockquote/cite semantics

### StatTile (Molecule)

```tsx
<StatTile value="100%" label="GDPR Compliant" />
```

**Props**:
- `value: string` - Stat value (e.g., "100%", "7 Years")
- `label: string` - Stat label/explanation
- `helpText?: string` - Optional screen reader text
- `className?: string` - Additional classes

### TestimonialsRail (Organism)

```tsx
// Provider testimonials (carousel)
<TestimonialsRail
  sectorFilter="provider"
  layout="carousel"
  headingId="providers-h2"
/>

// Enterprise testimonials (grid + stats)
<TestimonialsRail
  sectorFilter={['finance', 'healthcare', 'legal']}
  layout="grid"
  showStats
  headingId="enterprise-testimonials-h2"
/>
```

**Props**:
- `sectorFilter?: Sector | Sector[]` - Filter by sector(s)
- `limit?: number` - Max testimonials to show
- `layout?: 'grid' | 'carousel'` - Layout mode (default: 'grid')
- `showStats?: boolean` - Show TESTIMONIAL_STATS (default: false)
- `className?: string` - Additional classes
- `headingId?: string` - ID for aria-labelledby (default: 'testimonials-h2')

**Layouts**:
- **Grid**: 3-column grid on desktop, stacked on mobile
- **Carousel**: Horizontal scroll on mobile (snap-x), 3-column grid on desktop

**Accessibility**:
- Section has `aria-labelledby` linking to heading
- Carousel: `tabIndex={0}`, focus ring, `sr-only` instructions for keyboard navigation
- Staggered animations: 60ms delay per card

## Usage Examples

### Enterprise Page

```tsx
import { TestimonialsRail } from '@/components/organisms/TestimonialsRail/TestimonialsRail'

export function EnterpriseTestimonials() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center animate-in fade-in-50 slide-in-from-bottom-2">
          <h2 id="enterprise-testimonials-h2" className="mb-4 text-4xl font-bold text-foreground">
            Trusted by Regulated Industries
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            Organizations in highly regulated industries trust rbee for compliance-first AI infrastructure.
          </p>
        </div>

        <TestimonialsRail
          sectorFilter={['finance', 'healthcare', 'legal']}
          layout="grid"
          showStats
          headingId="enterprise-testimonials-h2"
        />
      </div>
    </section>
  )
}
```

### Providers Page

```tsx
import { TestimonialsRail } from '@/components/organisms/TestimonialsRail/TestimonialsRail'

export function ProvidersTestimonials() {
  return (
    <section className="px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <header className="mb-14 text-center">
          <h2 id="providers-h2" className="mb-3 text-4xl font-bold text-foreground">
            What Real Providers Are Earning
          </h2>
          <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
            GPU owners on the rbee marketplace turn idle time into steady payouts.
          </p>
        </header>

        <TestimonialsRail
          sectorFilter="provider"
          layout="carousel"
          headingId="providers-h2"
        />
      </div>
    </section>
  )
}
```

## Copy Guidelines

### Testimonial Quotes

**Crisp, number-forward, compliant-focused**:
- Keep quotes < 22 words when possible
- Lead with metrics/outcomes
- Use em dashes (—) for emphasis
- Include specific compliance terms (PCI-DSS, HIPAA, GDPR, SOC2)

**Provider testimonials** emphasize:
- Payout amounts (€160/mo, €420/mo, €780/mo)
- Setup speed ("under 10 minutes")
- Comparison to alternatives ("better than mining")

**Enterprise testimonials** emphasize:
- Compliance requirements (PCI-DSS, HIPAA, GDPR)
- Audit results ("zero findings")
- Data sovereignty ("EU-only", "on-premises")
- Specific use cases ("patient management", "attorney-client privilege")

### Verified Badges

- **Providers**: "Verified payout" (text-[11px] bg-secondary rounded-full)
- **Enterprise**: "Verified customer" (same styling)

## Styling Tokens

All semantic tokens from Tailwind config:
- `bg-card`, `bg-background`, `bg-gradient-to-b from-card to-background`
- `border-border`, `text-foreground`, `text-muted-foreground`, `text-primary`
- `bg-primary/10` (sector icon chips)
- `bg-secondary` (verified badges)
- `text-chart-3` (success/checkmarks)
- `hover:shadow-md transition-shadow` (card hover)

## Animations

Using only `tw-animate-css` utilities:
- Header: `animate-in fade-in-50 slide-in-from-bottom-2`
- Cards: `animate-in fade-in-50` with staggered delays (60ms per card)
- Stats: `animate-in fade-in-50` with 200ms delay

## Accessibility Features

1. **Semantic HTML**:
   - `<section aria-labelledby="...">` for sections
   - `<blockquote>` with `<cite>` for quotes
   - Proper heading hierarchy

2. **ARIA Labels**:
   - RatingStars: `aria-label="5 out of 5 stars"` + `sr-only` text
   - TestimonialCard: `aria-describedby` linking author/role
   - StatTile: `role="group"` with `aria-label`
   - Carousel: `aria-label` with navigation instructions

3. **Keyboard Navigation**:
   - Carousel: `tabIndex={0}` with visible focus ring
   - `sr-only` instructions: "Use arrow keys or swipe to navigate"

4. **Screen Readers**:
   - Sector icons: `aria-hidden="true"`
   - Verified badges: visible text (no icon-only)
   - All interactive elements have accessible names

## Benefits

### Maintainability
- **Single source of truth**: Update quotes/stats in one place
- **Type safety**: TypeScript interfaces prevent errors
- **Consistent copy**: No duplicate/divergent testimonials

### Reusability
- **Atomic Design**: Small, composable components
- **Context-aware**: Filter by sector, choose layout
- **Backward compatible**: Legacy API still works

### Performance
- **No CLS**: Fixed dimensions, no layout shift
- **Optimized animations**: CSS-only, no JS overhead
- **Responsive**: Mobile-first with progressive enhancement

### Accessibility
- **WCAG compliant**: Semantic HTML, ARIA labels, keyboard navigation
- **Screen reader friendly**: Proper landmarks, descriptions
- **Focus management**: Visible focus rings, logical tab order

## Files Modified/Created

### Data
1. ✅ `data/testimonials.ts` - Single source of truth

### Components
2. ✅ `components/atoms/RatingStars/RatingStars.tsx` - Created
3. ✅ `components/molecules/TestimonialCard/TestimonialCard.tsx` - Updated (new API + legacy support)
4. ✅ `components/molecules/StatTile/StatTile.tsx` - Already existed (perfect as-is)
5. ✅ `components/organisms/TestimonialsRail/TestimonialsRail.tsx` - Created

### Pages
6. ✅ `components/organisms/Enterprise/enterprise-testimonials.tsx` - Refactored
7. ✅ `components/organisms/Providers/providers-testimonials.tsx` - Refactored

### Exports
8. ✅ `components/atoms/index.ts` - Added RatingStars
9. ✅ `components/organisms/index.ts` - Added TestimonialsRail

### Documentation
10. ✅ `TESTIMONIALS_SYSTEM.md` - This file

## QA Checklist

- ✅ Mobile: testimonials render as snap carousel
- ✅ Desktop: 3-column grid layout
- ✅ Stats consistent across pages (from TESTIMONIAL_STATS)
- ✅ Quotes use blockquote/cite semantics
- ✅ Sector icons match testimonial type
- ✅ Verified badges display correctly
- ✅ Ratings use RatingStars atom
- ✅ No CLS from images (none used by default)
- ✅ Only semantic tokens used
- ✅ Only tw-animate-css utilities used
- ✅ Keyboard navigation works in carousel
- ✅ Screen readers announce all content correctly
- ✅ Focus rings visible on interactive elements
- ✅ Backward compatibility maintained (legacy API works)

## Future Enhancements

### Potential Additions
- Avatar images (optional, with Next.js Image)
- More sectors (government, education, etc.)
- Testimonial carousel auto-play (optional)
- Video testimonials (optional)
- Company logos (optional)

### Migration Path
All existing pages using legacy TestimonialCard API will continue to work. Gradually migrate to new API:

```tsx
// Old (still works)
<TestimonialCard name="..." role="..." quote="..." />

// New (recommended)
<TestimonialCard t={TESTIMONIALS.find(t => t.id === 'marcus')} />
```

## Outcome

A centralized, maintainable testimonials system that:
- Eliminates duplicate code and copy
- Ensures consistency across pages
- Provides reusable, accessible components
- Supports multiple layouts and contexts
- Maintains backward compatibility
- Follows Atomic Design principles
- Uses only semantic tokens and tw-animate-css

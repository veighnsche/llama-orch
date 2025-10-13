# Use Cases Section – Porting Guide

## Overview

The `UseCasesSection` organism provides a **prop-driven, story-driven layout** for displaying mini case studies with social proof. It replaces hardcoded use-case sections across audience pages.

## Architecture

```
UseCasesSection (organism)
├── Header (kicker + title + subtitle)
├── CaseCard[] (molecule grid)
│   ├── Icon plate + Avatar
│   ├── Quote block
│   └── Facts list (3 rows)
└── Micro-CTA rail (optional)
```

## API

```tsx
type Case = {
  icon: React.ReactNode          // Lucide icon component
  title: string                  // e.g., "Gaming PC Owners"
  subtitle?: string              // e.g., "Most common provider type"
  quote: string                  // First-person testimonial
  facts: { label: string; value: string }[]  // Exactly 3 recommended
  image?: { src: string; alt: string }       // Avatar or context image
  highlight?: string             // Optional badge text
}

type UseCasesSectionProps = {
  kicker?: string                // e.g., "Real Providers, Real Earnings"
  title: string                  // e.g., "Who's Earning with rbee?"
  subtitle?: string              // Supporting copy
  cases: Case[]                  // 4 cases recommended (2x2 grid)
  ctas?: {
    primary?: { label: string; href: string }
    secondary?: { label: string; href: string }
  }
  className?: string
}
```

## Porting Existing Sections

### Step 1: Identify Target Components

Search for components with:
- Hardcoded persona/audience cards
- Testimonial-style quotes
- Stats/facts in key-value format
- Similar visual hierarchy

**Common patterns:**
- `components/organisms/Audiences/`
- `components/organisms/WhoUses/`
- `components/organisms/Personas/`

### Step 2: Map Data to `Case[]`

For each existing card, extract:

```tsx
{
  icon: <YourIcon />,           // Keep existing Lucide icon
  title: 'Card Title',          // From h3/heading
  subtitle: 'Tagline',          // From subheading (optional)
  quote: 'First-person quote',  // From paragraph/blockquote
  facts: [                      // From stats list
    { label: 'Setup:', value: '3-6 GPUs' },
    { label: 'Availability:', value: '20-24 h/day' },
    { label: 'Monthly:', value: '€300-600' },  // Auto-highlighted
  ],
  image: {                      // Add avatar if desired
    src: '/images/path.jpg',
    alt: 'Descriptive prompt for context',
  },
}
```

### Step 3: Replace Component

**Before:**
```tsx
export function AudienceUseCases() {
  return (
    <section>
      <h2>Our Audiences</h2>
      <div className="grid">
        {/* Hardcoded cards */}
      </div>
    </section>
  )
}
```

**After:**
```tsx
import { UseCasesSection } from '@/components/organisms/Providers/providers-use-cases'

export function AudienceUseCases() {
  const cases: Case[] = [
    // ... mapped data
  ]

  return (
    <UseCasesSection
      kicker="Real Stories"
      title="Our Audiences"
      subtitle="Supporting copy here"
      cases={cases}
      ctas={{
        primary: { label: 'Get Started', href: '/signup' },
        secondary: { label: 'Learn More', href: '#details' },
      }}
    />
  )
}
```

### Step 4: Verify Visual Consistency

- **Typography:** Matches design tokens (tracking-tight, font-extrabold)
- **Spacing:** py-20 lg:py-28, gap-6
- **Motion:** Staggered delays (75/150/200/300ms)
- **Responsive:** Avatar hidden on mobile, cards stack

## Image Guidelines

### Avatar Images

Place in `/public/images/{audience}/usecases/{persona-slug}.jpg`

**Prompts (if generating):**
- **Gaming PC Owner**: "portrait of a friendly young adult PC gamer, casual hoodie, sitting at modern gaming desk with dual monitors, RGB-lit tower visible with tempered glass side panel showing colorful GPU fans, mechanical keyboard with backlight, soft dark navy background with subtle ambient lighting, cinematic rim lighting from monitor glow, warm trustworthy expression, professional headshot composition, 50mm lens, shallow depth of field"
- **Homelab Enthusiast**: "portrait of a tech enthusiast in casual flannel shirt, standing beside professional home server rack with 19-inch rails, 4U chassis with multiple GPUs visible through ventilated front panel, blue LED status lights, perfectly cable-managed with velcro straps and labeled ethernet cables, cozy home office setting with wooden desk and plants, warm moody lighting from desk lamp, soft bokeh background, confident knowledgeable expression, documentary photography style, natural colors, 50mm lens"
- **Former Crypto Miner**: "portrait of a resourceful entrepreneur in tech t-shirt, standing next to repurposed open-air mining frame with aluminum rails, 8 GPUs mounted horizontally with industrial-grade risers, clean cable management with zip ties, converted into professional workstation setup, industrial metal shelving unit, modern minimalist workspace with concrete wall texture, cool LED strip lighting casting blue-white glow, organized and efficient aesthetic, determined entrepreneurial expression, editorial photography style, high contrast, 35mm lens"
- **Workstation Owner**: "portrait of a creative professional 3D artist in casual button-up shirt, sitting at ergonomic desk with 34-inch ultrawide curved monitor displaying 3D modeling software, powerful workstation tower visible under desk with mesh front panel and subtle white LED accents, graphics tablet and stylus on desk, minimalist creative studio with white walls and floating shelves, soft diffused studio lighting from large window with sheer curtains, calm focused expression, professional lifestyle photography, clean modern aesthetic, natural light, 50mm lens, airy atmosphere"

**Specs:**
- Size: 48x48px (displayed), provide 96x96px (2x)
- Format: JPG/WebP
- Style: Contextual rigs > faces (if brand disallows portraits)

## Accessibility

- Icons: `aria-hidden` (decorative)
- Avatars: Descriptive `alt` (context, not identity)
- Quotes: Plain text (no blockquote styling required)
- Numbers: `tabular-nums` (prevent layout shift)
- Motion: Respects `prefers-reduced-motion`

## Motion Details

- **Header:** `animate-in fade-in slide-in-from-bottom-2`
- **Cards:** Staggered `delay-75/150/200/300`
- **Hover:** `hover:translate-y-0.5` on cards, `hover:scale-[1.02]` on icon plate
- **Library:** tw-animate-css (built-in Tailwind utilities)

## Facts List Highlighting

The component auto-highlights earnings values:

```tsx
const isEarnings = fact.label.toLowerCase().includes('monthly')
```

If your label contains "monthly", the value gets `font-semibold text-primary`.

**Recommended labels:**
- "Monthly:" (auto-highlighted)
- "Setup:" / "Typical GPU:"
- "Availability:"

## CTA Rail

Optional bottom rail with primary + secondary buttons:

```tsx
ctas={{
  primary: { label: 'Start Earning', href: '/signup' },
  secondary: { label: 'Estimate My Payout', href: '#calculator' },
}}
```

Omit `ctas` prop to hide the rail entirely.

## Example: Porting "Developers" Audience

**Old component:**
```tsx
// components/organisms/Developers/DeveloperUseCases.tsx
export function DeveloperUseCases() {
  return (
    <section className="py-24">
      <h2>Who Uses Our API?</h2>
      <div className="grid md:grid-cols-3">
        <div>
          <Code className="text-primary" />
          <h3>Indie Hackers</h3>
          <p>"I built a chatbot in a weekend..."</p>
          <ul>
            <li>Requests: 10k/mo</li>
            <li>Cost: €50/mo</li>
          </ul>
        </div>
        {/* ... more cards */}
      </div>
    </section>
  )
}
```

**New component:**
```tsx
import { UseCasesSection, type Case } from '@/components/organisms/Providers/providers-use-cases'
import { Code, Rocket, Building } from 'lucide-react'

export function DeveloperUseCases() {
  const cases: Case[] = [
    {
      icon: <Code />,
      title: 'Indie Hackers',
      quote: 'I built a chatbot in a weekend using the API. Cost is predictable.',
      facts: [
        { label: 'Requests:', value: '10k/mo' },
        { label: 'Latency:', value: '<200ms p95' },
        { label: 'Monthly:', value: '€50' },
      ],
    },
    // ... more cases
  ]

  return (
    <UseCasesSection
      kicker="Developer Stories"
      title="Who Uses Our API?"
      subtitle="From indie hackers to enterprises, our API scales with you."
      cases={cases}
      ctas={{
        primary: { label: 'Get API Key', href: '/api/signup' },
        secondary: { label: 'View Docs', href: '/docs' },
      }}
    />
  )
}
```

## Checklist

- [ ] Extracted all hardcoded cards to `Case[]` data
- [ ] Mapped icon, title, subtitle, quote, facts
- [ ] Added avatars (optional, hidden on mobile)
- [ ] Verified "Monthly:" label for auto-highlighting
- [ ] Tested responsive layout (mobile stack, desktop 2x2)
- [ ] Confirmed motion (staggered delays, hover states)
- [ ] Accessibility: icons aria-hidden, avatars descriptive alt
- [ ] CTAs point to correct routes
- [ ] Removed old component file

## Benefits

✅ **Consistency:** Same visual language across all audience pages  
✅ **Maintainability:** Update layout once, propagate everywhere  
✅ **Flexibility:** Swap copy/assets per audience without layout changes  
✅ **Performance:** Shared component = smaller bundle  
✅ **Accessibility:** Baked-in a11y patterns  

---

**Questions?** Check `providers-use-cases.tsx` for implementation details.

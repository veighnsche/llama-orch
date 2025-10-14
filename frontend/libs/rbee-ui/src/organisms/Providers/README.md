# ProvidersUseCases / UseCasesSection

## Quick Start

```tsx
import { UseCasesSection } from '@/components/organisms/Providers/providers-use-cases'
import { Gamepad2 } from 'lucide-react'

<UseCasesSection
  kicker="Real Providers, Real Earnings"
  title="Who's Earning with rbee?"
  subtitle="From gamers to homelab builders, anyone with a spare GPU can turn idle time into income."
  cases={[
    {
      icon: <Gamepad2 />,
      title: 'Gaming PC Owners',
      subtitle: 'Most common provider type',
      quote: 'I game ~3-4 hours/day. The rest, my 4090 was idle. Now it earns ~€150/mo while I\'m at work or asleep.',
      facts: [
        { label: 'Typical GPU:', value: 'RTX 4080-4090' },
        { label: 'Availability:', value: '16-20 h/day' },
        { label: 'Monthly:', value: '€120-180' },
      ],
      image: {
        src: '/images/providers/usecases/gaming-pc-owner.jpg',
        alt: 'portrait of a friendly PC gamer at a desk with RGB-lit tower',
      },
    },
    // ... 3 more cases
  ]}
  ctas={{
    primary: { label: 'Start Earning', href: '/signup' },
    secondary: { label: 'Estimate My Payout', href: '#earnings-calculator' },
  }}
/>
```

## Features

- **Story-driven layout:** Mini case studies with quotes + facts
- **Social proof:** Avatar images (optional, hidden on mobile)
- **Auto-highlighting:** "Monthly:" facts get primary color
- **Motion:** Staggered fade-in with hover states (tw-animate-css)
- **Prop-driven:** Reusable across all audience pages
- **Accessible:** Icons aria-hidden, tabular-nums, descriptive alts

## Layout

```
┌─────────────────────────────────────────────────┐
│  Kicker (text-sm text-primary/80)              │
│  Title (text-4xl font-extrabold tracking-tight)│
│  Subtitle (text-lg lg:text-xl)                 │
│                                                 │
│  ┌──────────────┐  ┌──────────────┐           │
│  │ [Icon] Title │  │ [Icon] Title │           │
│  │ Quote        │  │ Quote        │           │
│  │ • Fact 1     │  │ • Fact 1     │           │
│  │ • Fact 2     │  │ • Fact 2     │           │
│  │ • Monthly: € │  │ • Monthly: € │ (primary) │
│  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐           │
│  │ [Icon] Title │  │ [Icon] Title │           │
│  │ ...          │  │ ...          │           │
│  └──────────────┘  └──────────────┘           │
│                                                 │
│  Ready to join them?                           │
│  [Primary CTA] [Secondary CTA]                 │
└─────────────────────────────────────────────────┘
```

## Responsive

- **Mobile:** Cards stack (1 col), avatars hidden
- **Desktop:** 2x2 grid, avatars visible (48px circle)

## Motion

- Header: `animate-in fade-in slide-in-from-bottom-2`
- Cards: Staggered `delay-75/150/200/300`
- Hover: `translate-y-0.5` on cards, `scale-[1.02]` on icon plate

## Porting Guide

See [USE_CASES_PORTING_GUIDE.md](./USE_CASES_PORTING_GUIDE.md) for detailed migration instructions.

## Files

- `providers-use-cases.tsx` – Component implementation
- `USE_CASES_PORTING_GUIDE.md` – Migration guide for other sections
- `README.md` – This file

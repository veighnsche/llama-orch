# SolutionSection Migration Guide

**For teams porting solution/how-it-works sections to the unified component**

---

## Quick Start

Replace your custom solution component with `SolutionSection`:

```tsx
import { SolutionSection } from '@/components/organisms/SolutionSection/SolutionSection'
import { Icon1, Icon2, Icon3, Icon4 } from 'lucide-react'

export function YourSolution() {
  return (
    <SolutionSection
      kicker="How It Works"
      title="Your Benefit-Led Headline"
      subtitle="Your value proposition in one sentence."
      features={[
        { icon: <Icon1 className="h-8 w-8" aria-hidden="true" />, title: 'Feature 1', body: 'Short description.' },
        { icon: <Icon2 className="h-8 w-8" aria-hidden="true" />, title: 'Feature 2', body: 'Short description.' },
        { icon: <Icon3 className="h-8 w-8" aria-hidden="true" />, title: 'Feature 3', body: 'Short description.' },
        { icon: <Icon4 className="h-8 w-8" aria-hidden="true" />, title: 'Feature 4', body: 'Short description.' },
      ]}
      steps={[
        { title: 'Step 1', body: 'Action-oriented description.' },
        { title: 'Step 2', body: 'Action-oriented description.' },
        { title: 'Step 3', body: 'Action-oriented description.' },
        { title: 'Step 4', body: 'Action-oriented description.' },
      ]}
      ctaPrimary={{ label: 'Primary Action', href: '/signup' }}
      ctaSecondary={{ label: 'Secondary Action', href: '/docs' }}
    />
  )
}
```

---

## Migration Checklist

### 1. Map Your Content

| Old Component | New Prop | Notes |
|---------------|----------|-------|
| Section heading | `title` | Make it benefit-led |
| Subheading | `subtitle` | One-sentence value prop |
| Small label | `kicker` | Optional, e.g., "How It Works" |
| Feature cards | `features[]` | Icon + title + body |
| Steps/process | `steps[]` | Title + body (action verbs) |
| Earnings/metrics | `earnings?` | Optional, for providers/enterprise |
| Primary CTA | `ctaPrimary?` | Label + href + optional ariaLabel |
| Secondary CTA | `ctaSecondary?` | Label + href + optional ariaLabel |

### 2. Update Icons

**Old:**
```tsx
<Icon className="h-6 w-6 text-primary" />
```

**New:**
```tsx
<Icon className="h-8 w-8" aria-hidden="true" />
```

- Icons are now `h-8 w-8` (14 → 8 in Tailwind scale)
- Color is applied by the container (`text-primary` from icon plate)
- Always add `aria-hidden="true"`

### 3. Tighten Copy

**Features:**
- Title: 2-4 words, title case
- Body: 1 sentence, 8-12 words max

**Steps:**
- Title: Action verb + object (e.g., "Install rbee", "Configure GPUs")
- Body: 1 sentence, imperative mood

**Examples:**

❌ **Too verbose:**
```tsx
{
  title: 'You can earn passive income',
  body: 'With rbee, you can earn between €50 and €200 per month per GPU, even while you are sleeping, gaming, or working on other projects.'
}
```

✅ **Crisp:**
```tsx
{
  title: 'Passive Income',
  body: 'Earn €50–200/mo per GPU—even while you game or sleep.'
}
```

### 4. Add Earnings Card (Optional)

**For providers (revenue focus):**
```tsx
earnings={{
  rows: [
    { model: 'RTX 4090', meta: '24GB VRAM • 450W', value: '€180/mo', note: 'at 80% utilization' },
    { model: 'RTX 4080', meta: '16GB VRAM • 320W', value: '€140/mo', note: 'at 80% utilization' },
    { model: 'RTX 3080', meta: '10GB VRAM • 320W', value: '€90/mo', note: 'at 80% utilization' },
  ],
  disclaimer: 'Actuals vary with demand, pricing, and availability. These are conservative estimates.',
}}
```

**For enterprise (compliance focus):**
```tsx
earnings={{
  title: 'Compliance Metrics',
  rows: [
    { model: 'Data Sovereignty', meta: 'GDPR Art. 44', value: '100%', note: 'EU-only' },
    { model: 'Audit Retention', meta: 'GDPR Art. 30', value: '7 years', note: 'immutable' },
    { model: 'Security Layers', meta: 'Defense-in-depth', value: '5 layers', note: 'zero-trust' },
  ],
  disclaimer: 'rbee is designed to meet GDPR, NIS2, and EU AI Act requirements. Consult your legal team for certification.',
}}
```

**For developers (no earnings):**
```tsx
// Omit earnings prop entirely
```

### 5. Add CTAs

**Primary CTA:**
```tsx
ctaPrimary={{
  label: 'Start Earning',
  href: '/signup',
  ariaLabel: 'Start earning with rbee', // Optional, auto-generated if omitted
}}
```

**Secondary CTA:**
```tsx
ctaSecondary={{
  label: 'Estimate My Payout',
  href: '#earnings-calculator',
}}
```

**Both buttons are optional.** Omit if your section doesn't need CTAs.

---

## Common Patterns

### Pattern 1: Provider/Revenue Focus
- **Features**: Passive income, control, security, ease
- **Steps**: Install → Configure → Join → Earn
- **Earnings**: GPU models with monthly revenue
- **CTAs**: "Start Earning" + "Estimate My Payout"

### Pattern 2: Developer/Control Focus
- **Features**: Zero cost, privacy, stability, hardware utilization
- **Steps**: Install → Add hardware → Download models → Build
- **Earnings**: None
- **CTAs**: "Get Started" + "View Documentation"

### Pattern 3: Enterprise/Compliance Focus
- **Features**: Data sovereignty, audit retention, event types, zero dependencies
- **Steps**: Deploy → Configure policies → Enable logging → Run
- **Earnings**: Compliance metrics (sovereignty, retention, layers)
- **CTAs**: "Request Demo" + "View Compliance Docs"

---

## Styling Customization

### Section Background

**Default:**
```tsx
bg-[radial-gradient(60rem_40rem_at_10%_-20%,theme(colors.primary/10),transparent)]
```

**Custom:**
```tsx
<SolutionSection
  className="bg-gradient-to-b from-background to-muted"
  // ... other props
/>
```

### Section ID (for anchor links)

```tsx
<SolutionSection
  id="how-it-works"
  // ... other props
/>
```

Now users can link to `#how-it-works`.

---

## Accessibility Requirements

### Icons
- Always add `aria-hidden="true"` to decorative icons
- Icons are purely visual; meaning comes from text

### Buttons
- Provide custom `ariaLabel` if button text is ambiguous
- Example: `ariaLabel: 'Start earning with rbee'` for "Start Earning" button

### Headings
- `title` renders as `<h2>`
- Feature/step titles render as `<h3>` (implicit, via font styles)
- Maintain heading hierarchy on your page

### Motion
- All animations include `motion-reduce:animate-none`
- Users with `prefers-reduced-motion` see instant rendering

---

## Testing Your Migration

### Visual Regression
1. Compare before/after screenshots
2. Check mobile (features stack to 2-up then 1-up)
3. Verify timeline line doesn't overlap on narrow screens

### Accessibility
1. Run axe DevTools
2. Tab through all buttons (should focus in order)
3. Test with screen reader (VoiceOver/NVDA)

### Content
1. Read aloud: Does it sound conversational?
2. Scan test: Can you understand in 5 seconds?
3. Mobile test: Does copy fit without wrapping awkwardly?

---

## Troubleshooting

### "My icons are too small"
Use `h-8 w-8` (not `h-6 w-6`). The icon plate is `h-14 w-14`, so icons should be ~57% of that.

### "My timeline steps are too long"
Keep step bodies to 1 sentence, 12 words max. Use imperative mood ("Run one command" not "You run one command").

### "I need more than 4 features"
The grid is `lg:grid-cols-4`. If you have 5+ features, consider:
- Combining related features
- Moving some to a separate "Features" section
- Using a different layout (not SolutionSection)

### "I need a different grid layout"
Currently not supported. SolutionSection is optimized for 4 features. For custom layouts, create a new component or extend SolutionSection with a `gridClassName` prop (future enhancement).

### "My earnings card doesn't fit"
Limit to 3 rows max. If you need more, consider:
- Showing top 3 with "View full pricing" link
- Using a separate pricing table section
- Abbreviating model names

---

## Examples

See `SOLUTION_SECTION_UNIFICATION.md` for complete examples of:
- `ProvidersSolution` (with earnings)
- `DevelopersSolution` (no earnings)
- `EnterpriseSolution` (with compliance metrics)

---

## Support

Questions? Check:
1. `SolutionSection.tsx` - TypeScript types and JSDoc
2. `SOLUTION_SECTION_UNIFICATION.md` - Full API reference
3. Existing wrappers (`providers-solution.tsx`, etc.) - Real-world examples

---

**Happy migrating!** 🚀

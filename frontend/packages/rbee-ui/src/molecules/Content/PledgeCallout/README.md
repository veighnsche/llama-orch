# PledgeCallout Component

## Purpose

High-contrast callout emphasizing rbee's privacy and security guarantees.

## Usage

```tsx
import { PledgeCallout } from '@/components/molecules'

<PledgeCallout />
```

## Structure

- Shield icon in colored circle (`bg-accent/20`, `text-chart-2`)
- Two-line pledge text
- Link to security details page

## Content

**Headline**: Your models. Your rules.

**Body**: rbee enforces zero-trust auth, immutable audit trails, and strict bind policies—so your code stays yours.

**Link**: Security details → `/security`

## Styling

- Container: `rounded-2xl border bg-card p-6 md:p-7 shadow-sm`
- Layout: `flex gap-4 items-start`
- Icon wrapper: `h-9 w-9 rounded-full bg-accent/20`
- Icon: `size-5 text-chart-2`
- Text: `text-sm md:text-base font-semibold` (headline), `text-sm text-muted-foreground leading-6` (body)

## Accessibility

- `aria-hidden="true"` on decorative icon
- Semantic link with hover underline
- Proper color contrast in light/dark modes

# Enterprise Hero — Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│ ENTERPRISE COMPLIANCE HERO — IMPLEMENTATION REFERENCE           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ NEW MOLECULES                                                   │
│   • StatTile      → components/molecules/StatTile/             │
│   • ComplianceChip → components/molecules/ComplianceChip/      │
│                                                                 │
│ ATOMS USED                                                      │
│   • Button        → @/components/atoms/Button/Button           │
│   • Badge         → @/components/atoms/Badge/Badge             │
│   • Icons         → lucide-react (Shield, Lock, FileCheck)     │
│                                                                 │
│ KEY FEATURES                                                    │
│   ✓ Semantic HTML (section, h1, ul/li, time)                   │
│   ✓ ARIA labels (labelledby, describedby, live regions)        │
│   ✓ Motion (tw-animate-css: fade-in, slide-in)                 │
│   ✓ Sticky audit panel (lg:sticky lg:top-24)                   │
│   ✓ Radial gradient background                                 │
│   ✓ Decorative illustration (Next.js Image)                    │
│                                                                 │
│ ACCESSIBILITY                                                   │
│   ✓ WCAG 2.1 AA compliant                                      │
│   ✓ Keyboard navigable                                         │
│   ✓ Screen reader optimized                                    │
│   ✓ Contrast ratios ≥4.5:1                                     │
│   ✓ Focus indicators visible                                   │
│                                                                 │
│ RESPONSIVE BREAKPOINTS                                          │
│   • Mobile:  <768px  → Single column, no sticky               │
│   • Tablet:  768-1023px → Single column, image visible        │
│   • Desktop: ≥1024px → Two columns, sticky panel              │
│                                                                 │
│ MOTION TIMING                                                   │
│   • Header:      0ms delay, 500ms duration                     │
│   • Tiles:       120ms delay                                   │
│   • Audit panel: 150ms delay                                   │
│                                                                 │
│ COMPLIANCE PROOF                                                │
│   • 100% GDPR Compliant                                        │
│   • 7 Years Audit Retention                                    │
│   • Zero US Cloud Dependencies                                 │
│   • GDPR / SOC2 / ISO 27001 chips                              │
│                                                                 │
│ AUDIT CONSOLE                                                   │
│   • 4 sample events (auth, data, task, compliance)             │
│   • Filter strip (All, Auth, Data, Exports)                    │
│   • ISO 8601 timestamps (<time> elements)                      │
│   • Floating badges (EU Only, 32 Types)                        │
│                                                                 │
│ CTAs                                                            │
│   • Primary:   "Schedule Demo" (solid, primary)                │
│   • Secondary: "View Compliance Details" (outline, #compliance)│
│                                                                 │
│ PENDING TASKS                                                   │
│   ⚠ Create decorative image: /illustrations/audit-ledger.webp │
│   ⚠ Wire "Schedule Demo" to booking flow                       │
│   ⚠ Run full QA checklist (see TESTING_CHECKLIST.md)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## File Locations

```
frontend/bin/commercial/
├── components/
│   ├── atoms/
│   │   ├── Button/Button.tsx
│   │   └── Badge/Badge.tsx
│   ├── molecules/
│   │   ├── StatTile/StatTile.tsx          ← NEW
│   │   ├── ComplianceChip/ComplianceChip.tsx ← NEW
│   │   └── index.ts                        ← UPDATED
│   └── organisms/
│       └── Enterprise/
│           ├── enterprise-hero.tsx         ← REDESIGNED
│           ├── ENTERPRISE_HERO_REDESIGN.md ← DOCS
│           ├── TESTING_CHECKLIST.md        ← QA
│           └── QUICK_REFERENCE.md          ← THIS FILE
└── public/
    └── illustrations/
        └── README-audit-ledger.md          ← ASSET SPEC
```

## Import Pattern

```tsx
import { Button } from '@/components/atoms/Button/Button'
import { Badge } from '@/components/atoms/Badge/Badge'
import { StatTile } from '@/components/molecules/StatTile/StatTile'
import { ComplianceChip } from '@/components/molecules/ComplianceChip/ComplianceChip'
import { Shield, Lock, FileCheck, Filter } from 'lucide-react'
import Link from 'next/link'
import Image from 'next/image'
```

## Usage Example

```tsx
<StatTile
  value="100%"
  label="GDPR Compliant"
  helpText="Full compliance with EU General Data Protection Regulation"
/>

<ComplianceChip
  icon={<FileCheck className="h-3 w-3" />}
  ariaLabel="GDPR Compliant certification"
>
  GDPR Compliant
</ComplianceChip>
```

## Design Tokens

```css
/* Colors */
--primary: #f59e0b        /* Amber */
--foreground: #0f172a     /* Dark slate */
--card: #1e293b           /* Card background (dark mode) */
--border: #334155         /* Border (dark mode) */
--muted-foreground: #94a3b8 /* Muted text (dark mode) */
--chart-3: #10b981        /* Success green */

/* Spacing */
--radius: 0.5rem          /* Base border radius */
```

## Animation Classes

```tsx
// Section header
className="animate-in fade-in-50 slide-in-from-bottom-2 duration-500"

// Tiles (with delay)
className="animate-in fade-in-50 [animation-delay:120ms]"

// Audit panel (with delay)
className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:150ms]"
```

## Sticky Panel

```tsx
// Right column with sticky behavior
<div className="relative w-full max-w-lg lg:sticky lg:top-24">
  {/* Audit console content */}
</div>
```

## Background Gradient

```tsx
// Section with radial gradient
<section className="bg-gradient-to-b from-background via-card to-background bg-[radial-gradient(60rem_40rem_at_20%_-10%,theme(colors.primary/8),transparent)]">
```

## Time Elements

```tsx
<time dateTime="2025-10-11T14:23:15Z" className="text-muted-foreground/70">
  2025-10-11 14:23:15 UTC
</time>
```

## ARIA Pattern

```tsx
// Section
<section aria-labelledby="enterprise-hero-h1" role="region">
  <h1 id="enterprise-hero-h1">...</h1>
  
  // CTAs with describedby
  <Button aria-describedby="compliance-proof-bar">...</Button>
  
  // Proof bar
  <div id="compliance-proof-bar" aria-live="polite">...</div>
  
  // Events list
  <ul role="list" aria-label="Recent audit events">
    <li aria-label="auth.success by admin@company.eu at ...">...</li>
  </ul>
</section>
```

## Testing Commands

```bash
# Development server
pnpm --filter @rbee/commercial dev

# Build (type checking)
pnpm --filter @rbee/commercial build

# Linting
pnpm --filter @rbee/commercial lint

# Format check
pnpm --filter @rbee/commercial format:check
```

## Browser DevTools Checks

1. **Accessibility tab**: No violations
2. **Console**: No errors or warnings
3. **Network**: Image loads correctly (or 404 handled gracefully)
4. **Performance**: No layout shifts (CLS = 0)
5. **Lighthouse**: Accessibility score = 100

---

**Version:** 1.0  
**Last Updated:** 2025-10-13  
**Status:** ✅ Implementation Complete, Pending QA

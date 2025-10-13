# CompliancePillar API Reference

**Component:** `CompliancePillar.tsx`  
**Type:** Molecule (Reusable)  
**Version:** 1.0

---

## Overview

`CompliancePillar` is a reusable molecule for displaying compliance standards (GDPR, SOC2, ISO 27001) with requirements checklist and optional callout content for endpoints, criteria, or controls.

---

## Props API

```typescript
interface CompliancePillarProps {
  /** Icon element (e.g., Globe, Shield, Lock) */
  icon: ReactNode
  
  /** Standard name (e.g., "GDPR", "SOC2", "ISO 27001") */
  title: string
  
  /** Standard type (e.g., "EU Regulation", "US Standard") */
  subtitle: string
  
  /** List of compliance requirements */
  checklist: string[]
  
  /** Optional callout content (endpoints, criteria, controls) */
  callout?: ReactNode
  
  /** Additional CSS classes */
  className?: string
}
```

---

## Usage Examples

### Basic Usage

```tsx
import { CompliancePillar } from '@/components/molecules/CompliancePillar/CompliancePillar'
import { Shield } from 'lucide-react'

<CompliancePillar
  icon={<Shield className="h-6 w-6" />}
  title="SOC2"
  subtitle="US Standard"
  checklist={[
    'Auditor query API',
    '32 audit event types',
    '7-year retention (Type II)',
    'Tamper-evident hash chains',
  ]}
/>
```

### With Callout

```tsx
<CompliancePillar
  icon={<Globe className="h-6 w-6" />}
  title="GDPR"
  subtitle="EU Regulation"
  checklist={[
    '7-year audit retention (Art. 30)',
    'Data access records (Art. 15)',
    'Erasure tracking (Art. 17)',
  ]}
  callout={
    <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
      <div className="mb-2 font-semibold text-chart-3">Compliance Endpoints</div>
      <div className="space-y-1 font-mono text-xs text-foreground/85">
        <div>GET /v2/compliance/data-access</div>
        <div>POST /v2/compliance/data-export</div>
      </div>
    </div>
  }
/>
```

### Full Example (Enterprise)

```tsx
<CompliancePillar
  icon={<Lock className="h-6 w-6" aria-hidden="true" />}
  title="ISO 27001"
  subtitle="International Standard"
  checklist={[
    'Incident records (A.16)',
    '3-year minimum retention',
    'Access logging (A.9)',
    'Crypto controls (A.10)',
    'Ops security (A.12)',
    'Security policies (A.5)',
  ]}
  callout={
    <div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
      <div className="mb-2 font-semibold text-chart-3">ISMS Controls</div>
      <div className="space-y-1 text-xs text-foreground/85">
        <div>✓ 114 controls implemented</div>
        <div>✓ Risk assessment framework</div>
        <div>✓ Continuous monitoring</div>
      </div>
    </div>
  }
/>
```

---

## Structure

```tsx
<div className="rounded-2xl border border-border bg-card/60 p-8 hover:shadow-lg">
  {/* Header */}
  <div className="flex items-center gap-3">
    <div className="rounded-xl bg-primary/10 p-3">{icon}</div>
    <div>
      <h3 id="{title-id}">{title}</h3>
      <p>{subtitle}</p>
    </div>
  </div>
  
  {/* Checklist */}
  <ul className="space-y-3" role="list">
    {checklist.map(item => (
      <li aria-label="{title} requirement: {item}">
        <Check className="text-chart-3" />
        <span>{item}</span>
      </li>
    ))}
  </ul>
  
  {/* Callout (optional) */}
  {callout}
</div>
```

---

## Styling Guidelines

### Card

- **Wrapper:** `rounded-2xl border border-border bg-card/60 p-8`
- **Hover:** `hover:shadow-lg transition-shadow`
- **Height:** `h-full` (equal height in grid)

### Header

- **Icon container:** `rounded-xl bg-primary/10 p-3 text-primary`
- **Icon size:** `h-6 w-6` (Lucide icons)
- **Title:** `text-2xl font-bold text-foreground`
- **Subtitle:** `text-sm text-muted-foreground`

### Checklist

- **List:** `<ul className="space-y-3" role="list">`
- **Item:** `flex items-start gap-2`
- **Icon:** `Check` from Lucide, `h-4 w-4 text-chart-3`
- **Text:** `text-sm leading-relaxed text-foreground/85`

### Callout (Recommended Pattern)

```tsx
<div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
  <div className="mb-2 font-semibold text-chart-3">{title}</div>
  <div className="space-y-1 text-xs text-foreground/85">
    {/* Content */}
  </div>
</div>
```

---

## Accessibility

### ARIA

- Card: `aria-labelledby="{title-id}"`
- Title: `id="{title-id}"` (auto-generated from title)
- List: `role="list"`
- Items: `role="listitem"` + `aria-label="{title} requirement: {item}"`

### Icons

- All icons: `aria-hidden="true"`
- Check icons: Decorative (context from text)

### Contrast

- Checklist text: `text-foreground/85` (≥4.5:1)
- Subtitle: `text-muted-foreground` (≥4.5:1)
- Callout text: `text-foreground/85` (≥4.5:1)

---

## Content Guidelines

### Checklist Items

- **Length:** 3-8 words per item
- **Pattern:** Action-noun or noun-descriptor
- **Examples:**
  - ✅ "7-year audit retention (Art. 30)"
  - ✅ "Auditor query API"
  - ✅ "Tamper-evident hash chains"
  - ❌ "We provide comprehensive audit retention capabilities for 7 years"

### Article References

- **Format:** `(Art. XX)` or `(A.XX)`
- **Examples:**
  - GDPR: `(Art. 30)`, `(Art. 15)`, `(Art. 17)`
  - ISO 27001: `(A.16)`, `(A.9)`, `(A.10)`

### Callout Content

**Three patterns:**

1. **Endpoints** (GDPR):
   - Use `font-mono text-xs`
   - Format: `GET /v2/compliance/endpoint`

2. **Criteria** (SOC2):
   - Use checkmarks: `✓`
   - Format: `✓ Category (Code)`

3. **Controls** (ISO 27001):
   - Use checkmarks: `✓`
   - Format: `✓ Control description`

---

## Design Tokens

```css
/* Card */
--border: #334155
--card: #1e293b

/* Header */
--primary: #f59e0b
--foreground: #0f172a

/* Checklist */
--chart-3: #10b981  /* Check icons */
--foreground: #0f172a

/* Callout */
--chart-3: #10b981  /* Border/background/title */
```

---

## Responsive Behavior

### Mobile (<768px)

- Card stacks vertically
- Full width
- Padding: `p-8` maintained

### Tablet (768-1023px)

- 2-column grid (if in grid)
- Equal height cards

### Desktop (≥1024px)

- 3-column grid (typical usage)
- Equal height cards
- Hover shadow effect

---

## Best Practices

### Icon Selection

- **GDPR:** Globe (international regulation)
- **SOC2:** Shield (security standard)
- **ISO 27001:** Lock (security controls)

### Checklist Length

- **Optimal:** 5-7 items
- **Minimum:** 3 items
- **Maximum:** 8 items (more gets cluttered)

### Callout Usage

- Use for concrete evidence (endpoints, criteria, controls)
- Keep concise (3-5 items)
- Use consistent styling across pillars

### Accessibility

- Always include `aria-hidden="true"` on icons
- Keep checklist items descriptive (screen reader friendly)
- Use semantic HTML (`<ul>`, `<li>`)

---

## Related Components

- **EnterpriseCompliance** — Uses three CompliancePillar instances
- **Button** — Used in CTA band below pillars
- **ComplianceChip** — Alternative for smaller compliance indicators

---

## Migration from Inline Implementation

**Before (inline):**
```tsx
<div className="rounded-lg border border-border bg-card p-8">
  <div className="mb-6 flex items-center gap-3">
    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
      <Globe className="h-6 w-6 text-primary" />
    </div>
    <div>
      <h3>GDPR</h3>
      <p>EU Regulation</p>
    </div>
  </div>
  <div className="space-y-3">
    {items.map(item => (
      <div className="flex items-start gap-2">
        <FileCheck className="h-4 w-4 text-chart-3" />
        <span>{item}</span>
      </div>
    ))}
  </div>
</div>
```

**After (CompliancePillar):**
```tsx
<CompliancePillar
  icon={<Globe className="h-6 w-6" />}
  title="GDPR"
  subtitle="EU Regulation"
  checklist={items}
/>
```

**Benefits:**
- Consistent styling across all pillars
- Better accessibility (semantic HTML, ARIA)
- Easier to maintain (single source of truth)
- Hover effects included
- Improved contrast (text-foreground/85)

---

**Last Updated:** 2025-10-13  
**Version:** 1.0  
**Status:** ✅ Production Ready

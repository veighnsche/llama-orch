# Comparison Matrix Structure

## Component Hierarchy

```
EnterpriseComparison (Organism)
├── Header Block
│   ├── Eyebrow: "Feature Matrix"
│   ├── H2: "Why Enterprises Choose rbee"
│   ├── Subcopy
│   └── Legal Caption
│
├── Legend (Atom)
│   ├── ✓ Included
│   ├── ✗ Not available
│   └── Partial (with explanation)
│
├── Desktop View (md+)
│   └── MatrixTable (Molecule)
│       ├── Table Header
│       │   ├── Feature (col)
│       │   ├── rbee (Self-Hosted) [ACCENT]
│       │   ├── OpenAI / Anthropic
│       │   ├── Azure OpenAI
│       │   └── Notes (optional)
│       │
│       └── Table Body
│           └── 12 Feature Rows
│               ├── Feature Name (th scope="row")
│               ├── rbee Value [bg-primary/5]
│               ├── OpenAI Value
│               ├── Azure Value
│               └── Note (optional)
│
├── Mobile View (<md)
│   ├── Provider Switcher (Segmented Control)
│   │   ├── rbee (Self-Hosted)
│   │   ├── OpenAI / Anthropic
│   │   └── Azure OpenAI
│   │
│   └── MatrixCard (Molecule)
│       ├── Provider Header [ACCENT for rbee]
│       └── Feature List (12 items)
│           └── Feature + Status Badge
│
└── Footnote
    └── "* Comparison based on publicly available information..."
```

## Data Flow

```
comparison-data.ts
├── PROVIDERS: Provider[]
│   └── { key, label, accent? }
│
└── FEATURES: Row[]
    └── { feature, values: Record<key, boolean|'Partial'|string>, note? }

↓ Props ↓

MatrixTable (Desktop)
├── columns={PROVIDERS}
└── rows={FEATURES}

MatrixCard (Mobile)
├── provider={PROVIDERS[selectedIndex]}
└── rows={FEATURES}
```

## Responsive Behavior

### Desktop (md+)
- Full table visible
- rbee column highlighted with bg-primary/5
- Row hover effects
- Zebra striping for readability
- Optional Notes column (right-aligned)

### Mobile (<md)
- Table hidden
- Segmented control to switch providers
- Single card view (no horizontal scroll)
- Touch-friendly buttons
- Compact feature list with status badges

## Accessibility Features

1. **Semantic HTML**
   - `<table>` with `<caption>`
   - `<th scope="col">` for column headers
   - `<th scope="row">` for feature names
   - Proper heading hierarchy

2. **ARIA Labels**
   - Status icons: aria-label="Included|Not available|Partial"
   - Provider switcher: aria-pressed
   - Skip link for screen readers

3. **Keyboard Navigation**
   - Tab through table cells
   - Focus visible on all interactive elements
   - Skip link accessible via keyboard

4. **Visual Indicators**
   - Color + icon for status (not color alone)
   - Sufficient contrast ratios
   - Hover states for interactive elements

## Animation Sequence

```
0ms    → Header: fade-in-50 slide-in-from-bottom-2
100ms  → Legend: fade-in-50
150ms  → Matrix: fade-in-50
```

## Status Rendering

### Boolean Values
- `true` → ✓ (Check icon, text-chart-3)
- `false` → ✗ (X icon, text-destructive)

### String Values
- `'Partial'` → Chip badge with tooltip
- Other strings → Plain text display

## Styling Tokens

### Colors
- Primary accent: `bg-primary/5`, `text-primary`
- Success: `text-chart-3`
- Destructive: `text-destructive`
- Muted: `text-muted-foreground`
- Borders: `border-border`, `border-border/80`, `border-border/60`

### Spacing
- Section padding: `px-6 py-24`
- Container: `max-w-7xl mx-auto`
- Cell padding: `p-3` (compact)
- Card padding: `p-5`

### Effects
- Row hover: `hover:bg-secondary/30 transition-colors`
- Zebra: `odd:bg-background even:bg-background/60`
- Rounded corners: `rounded-2xl` (cards), `rounded-full` (chips)

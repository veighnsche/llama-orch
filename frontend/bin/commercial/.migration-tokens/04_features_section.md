# Work Unit 04: Features Section

**Priority:** MEDIUM  
**Component:** `/components/features-section.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 6 | `bg-slate-50` | Section background (light) |
| 9 | `text-slate-900` | Section heading |
| 40 | `bg-white` | Tab content background |
| 40 | `border-slate-200` | Tab content border |
| 42 | `text-slate-900` | Tab heading |
| 43 | `text-slate-600` | Tab description |
| 49 | `bg-slate-900` | Code block background |
| 50 | `text-slate-400` | Code comment |
| 51 | `text-green-400` | Code command |
| 53 | `text-green-400` | Code command |
| 56 | `bg-green-50` | Success callout background |
| 56 | `border-green-200` | Success callout border |
| 57 | `text-green-900` | Success callout text |
| 63 | `bg-white` | Tab content background |
| 63 | `border-slate-200` | Tab content border |
| 65 | `text-slate-900` | Tab heading |
| 66 | `text-slate-600` | Tab description |
| 73 | `text-slate-600` | GPU label |
| 74 | `bg-slate-100` | Progress bar background |
| 75 | `bg-amber-500` | Progress bar fill |
| 76 | `text-white` | Progress bar text |
| 106 | `bg-blue-50` | Info callout background |
| 106 | `border-blue-200` | Info callout border |
| 107 | `text-blue-900` | Info callout text |
| 113 | `bg-white` | Tab content background |
| 113 | `border-slate-200` | Tab content border |
| 115 | `text-slate-900` | Tab heading |
| 116 | `text-slate-600` | Tab description |
| 122 | `bg-slate-900` | Code block background |
| 123 | `text-slate-400` | Code comment |
| 124 | `text-purple-400` | Code keyword |
| 125 | `text-slate-300` | Code text |
| 127 | `text-green-300` | Code string |
| 144 | `bg-amber-50` | Warning callout background |
| 144 | `border-amber-200` | Warning callout border |
| 145 | `text-amber-900` | Warning callout text |
| 151 | `bg-white` | Tab content background |
| 151 | `border-slate-200` | Tab content border |
| 153 | `text-slate-900` | Tab heading |
| 154 | `text-slate-600` | Tab description |
| 159 | `bg-slate-900` | Code block background |
| 160 | `text-slate-400` | Code event label |
| 161 | `text-slate-300` | Code JSON |
| 170 | `bg-slate-50` | Neutral callout background |
| 170 | `border-slate-200` | Neutral callout border |
| 171 | `text-slate-900` | Neutral callout text |

---

## Proposed Token Replacements

### Section Layout

```tsx
// Line 6: Section background
className="py-24 bg-secondary"

// Line 9: Section heading
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"
```

### Tab Content Containers

```tsx
// Lines 40, 63, 113, 151: Tab content
className="bg-card border border-border rounded-lg p-8 space-y-6"

// Lines 42, 65, 115, 153: Tab headings
className="text-2xl font-bold text-card-foreground mb-3"

// Lines 43, 66, 116, 154: Tab descriptions
className="text-muted-foreground leading-relaxed"
```

### Code Blocks

```tsx
// Lines 49, 122, 159: Code block background
className="bg-muted rounded-lg p-6 font-mono text-sm"

// Lines 50, 123, 160: Code comments
className="text-muted-foreground"

// Lines 51, 53: Code commands (keep green - semantic)
className="text-chart-3 mt-2"  // chart-3 is green

// Line 124: Code keyword (keep purple - semantic)
className="text-chart-4"  // chart-4 is purple

// Lines 125, 161: Code text
className="text-foreground"

// Line 127: Code string (keep green - semantic)
className="text-chart-3"

// Line 76: Progress bar text
className="text-primary-foreground"
```

### Progress Bars

```tsx
// Line 73: GPU label
className="text-muted-foreground text-sm"

// Line 74: Progress bar background
className="flex-1 h-8 bg-secondary rounded-full overflow-hidden"

// Line 75: Progress bar fill
className="h-full bg-primary flex items-center justify-end pr-2"
```

### Callout Boxes

```tsx
// Line 56: Success callout
className="bg-chart-3/10 border border-chart-3/20 rounded-lg p-4"
// Text:
className="text-foreground font-medium"

// Line 106: Info callout
className="bg-chart-2/10 border border-chart-2/20 rounded-lg p-4"
// Text:
className="text-foreground font-medium"

// Line 144: Warning callout
className="bg-primary/10 border border-primary/20 rounded-lg p-4"
// Text:
className="text-foreground font-medium"

// Line 170: Neutral callout
className="bg-muted border border-border rounded-lg p-4"
// Text:
className="text-foreground font-medium"
```

---

## New Tokens Required

**Optional:** Add semantic callout tokens for better control:

```css
/* Add to globals.css */
:root {
  --success: #10b981;           /* green-500 */
  --success-foreground: #ffffff;
  --info: #3b82f6;              /* blue-500 */
  --info-foreground: #ffffff;
  --warning: #f59e0b;           /* amber-500 */
  --warning-foreground: #ffffff;
}

.dark {
  --success: #22c55e;           /* green-400 */
  --success-foreground: #0f172a;
  --info: #60a5fa;              /* blue-400 */
  --info-foreground: #0f172a;
  --warning: #fbbf24;           /* amber-400 */
  --warning-foreground: #0f172a;
}

@theme inline {
  --color-success: var(--success);
  --color-success-foreground: var(--success-foreground);
  --color-info: var(--info);
  --color-info-foreground: var(--info-foreground);
  --color-warning: var(--warning);
  --color-warning-foreground: var(--warning-foreground);
}
```

Then use:
```tsx
// Success callout
className="bg-success/10 border border-success/20 rounded-lg p-4"
className="text-success-foreground font-medium"

// Info callout
className="bg-info/10 border border-info/20 rounded-lg p-4"
className="text-info-foreground font-medium"

// Warning callout
className="bg-warning/10 border border-warning/20 rounded-lg p-4"
className="text-warning-foreground font-medium"
```

---

## Implementation Notes

### Code Block Syntax Highlighting

The code blocks use semantic colors:
- **Green:** Commands, strings, success
- **Purple:** Keywords
- **Blue:** Metadata, info
- **Amber:** Warnings

These should map to `chart-*` tokens or new semantic tokens.

### Callout Box Patterns

Callout boxes follow a pattern:
- Background: `color/10` (10% opacity)
- Border: `color/20` (20% opacity)
- Text: Full color or foreground

This pattern works well with tokens and provides good contrast.

### Progress Bar Accessibility

Ensure progress bars have sufficient contrast in both themes. The fill should be clearly visible against the background.

---

## Verification Checklist

- [ ] Section background works in both themes
- [ ] Tab content cards are readable in both themes
- [ ] Code blocks have good contrast
- [ ] Syntax highlighting colors are semantic
- [ ] Progress bars are visible in both themes
- [ ] Callout boxes have good contrast
- [ ] Success (green) callouts are distinct
- [ ] Info (blue) callouts are distinct
- [ ] Warning (amber) callouts are distinct
- [ ] Neutral callouts are distinct
- [ ] No hardcoded `slate-*`, `amber-*`, `green-*`, `blue-*` classes remain

---

## Estimated Complexity

**Medium** - Many color instances, callout boxes need careful handling, code syntax highlighting must be preserved.

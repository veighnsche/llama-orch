# Work Unit 11: Technical & How It Works

**Priority:** LOW  
**Components:**
- `/components/technical-section.tsx`
- `/components/how-it-works-section.tsx`

---

## Technical Section

### Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 6 | `bg-white` | Section background |
| 9 | `text-slate-900` | Section heading |
| 17 | `text-slate-900` | Subsection heading |
| 20, 29, 38, 47, 56 | `bg-green-100` | List item indicator background |
| 21, 30, 39, 48, 57 | `bg-green-600` | List item indicator dot |
| 24, 33, 42, 51, 60 | `text-slate-900` | List item heading |
| 25, 34, 43, 52, 61 | `text-slate-600` | List item description |
| 69 | `text-slate-900` | Subsection heading |
| 71, 75, 79, 83, 87 | `bg-slate-50` | Tech stack card background |
| 71, 75, 79, 83, 87 | `border-slate-200` | Tech stack card border |
| 72, 76, 80, 84, 88 | `text-slate-900` | Tech stack name |
| 73, 77, 81, 85, 89 | `text-slate-600` | Tech stack description |
| 93 | `bg-amber-50` | Open source callout background |
| 93 | `border-amber-200` | Open source callout border |
| 95 | `text-amber-900` | Open source heading |
| 96 | `text-amber-700` | Open source subtext |
| 98 | `border-amber-300` | Button border |

### Proposed Replacements (Technical)

```tsx
// Line 6: Section background
className="py-24 bg-background"

// Lines 9, 17, 69: Headings
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"
className="text-2xl font-bold text-foreground"

// Lines 20, 29, 38, 47, 56: List indicator backgrounds
className="h-6 w-6 rounded-full bg-chart-3/10 flex items-center justify-center flex-shrink-0 mt-0.5"

// Lines 21, 30, 39, 48, 57: List indicator dots
className="h-2 w-2 rounded-full bg-chart-3"

// Lines 24, 33, 42, 51, 60: List item headings
className="font-medium text-card-foreground"

// Lines 25, 34, 43, 52, 61: List item descriptions
className="text-sm text-muted-foreground"

// Lines 71, 75, 79, 83, 87: Tech stack cards
className="bg-secondary border border-border rounded-lg p-4"

// Lines 72, 76, 80, 84, 88: Tech stack names
className="font-medium text-card-foreground"

// Lines 73, 77, 81, 85, 89: Tech stack descriptions
className="text-sm text-muted-foreground"

// Line 93: Open source callout
className="bg-primary/10 border border-primary/20 rounded-lg p-4 flex items-center justify-between"

// Line 95: Open source heading
className="font-bold text-foreground"

// Line 96: Open source subtext
className="text-sm text-muted-foreground"

// Line 98: Button
className="border-primary/30 bg-transparent"
```

---

## How It Works Section

### Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 3 | `bg-white` | Section background |
| 6 | `text-slate-900` | Section heading |
| 15, 33, 56, 74 | `bg-amber-500` | Step number background |
| 15, 33, 56, 74 | `text-white` | Step number text |
| 18, 36, 59, 77 | `text-slate-900` | Step heading |
| 19, 37, 60, 78 | `text-slate-600` | Step description |
| 23, 41, 64, 82 | `bg-slate-900` | Code block background |
| 24, 42, 65 | `text-green-400` | Code command (green) |
| 25, 43, 44, 47, 48, 66, 67 | `text-slate-400` | Code comment/secondary |
| 26, 45, 49 | `text-slate-300` | Code output |
| 83 | `text-purple-400` | Code keyword (purple) |
| 84, 88, 89, 91, 92, 93, 96, 97, 99 | `text-slate-300` | Code text |
| 85 | `text-purple-400` | Code keyword (purple) |
| 86, 94, 97 | `text-green-300` | Code string (green) |
| 91 | `text-blue-300` | Code function (blue) |

### Proposed Replacements (How It Works)

```tsx
// Line 3: Section background
className="py-24 bg-background"

// Line 6: Section heading
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"

// Lines 15, 33, 56, 74: Step numbers
className="inline-flex items-center justify-center h-12 w-12 rounded-full bg-primary text-primary-foreground font-bold text-xl"

// Lines 18, 36, 59, 77: Step headings
className="text-2xl font-bold text-foreground"

// Lines 19, 37, 60, 78: Step descriptions
className="text-muted-foreground leading-relaxed"

// Lines 23, 41, 64, 82: Code blocks
className="bg-muted rounded-lg p-6 font-mono text-sm"

// Lines 24, 42, 65: Code commands (keep green - semantic)
className="text-chart-3"

// Lines 25, 43, 44, 47, 48, 66, 67: Code comments
className="text-muted-foreground"

// Lines 26, 45, 49, 84, 88, 89, 92, 93, 96, 99: Code text/output
className="text-foreground"

// Lines 83, 85: Code keywords (keep purple - semantic)
className="text-chart-4"

// Lines 86, 94, 97: Code strings (keep green - semantic)
className="text-chart-3"

// Line 91: Code function (keep blue - semantic)
className="text-chart-2"
```

---

## New Tokens Required

**None** - All colors map to existing tokens. Code syntax highlighting uses chart colors.

---

## Implementation Notes

### Technical Section List Indicators

The green dots indicate completed/available features. This should use `chart-3` (green) to maintain the success semantic.

### Open Source Callout

The amber callout highlighting open source should use `primary` tokens to maintain brand consistency.

### Code Syntax Highlighting

The "How It Works" section uses semantic colors for code:
- **Green:** Commands, strings, success
- **Purple:** Keywords
- **Blue:** Functions
- **Muted:** Comments

These should map to chart tokens for consistency with other code blocks.

---

## Verification Checklist

- [ ] Technical section renders correctly in both themes
- [ ] List indicators (green dots) are visible
- [ ] Tech stack cards are readable
- [ ] Open source callout is prominent
- [ ] How It Works section renders correctly in both themes
- [ ] Step numbers are visible and prominent
- [ ] Code blocks have good contrast
- [ ] Syntax highlighting colors are semantic
- [ ] Commands (green) are distinct
- [ ] Keywords (purple) are distinct
- [ ] Functions (blue) are distinct
- [ ] Comments are de-emphasized
- [ ] No hardcoded `slate-*`, `amber-*`, `green-*`, `purple-*`, `blue-*` classes remain

---

## Estimated Complexity

**Low-Medium** - Straightforward replacements, code syntax highlighting must be preserved.

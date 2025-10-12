# Work Unit 09: FAQ Section

**Priority:** LOW  
**Component:** `/components/faq-section.tsx`

---

## Current Hardcoded Colors

| Line | Current Class | Semantic Purpose |
|------|---------------|------------------|
| 5 | `bg-slate-50` | Section background |
| 8 | `text-slate-900` | Section heading |
| 15 | `bg-white` | Accordion item background |
| 15 | `border-slate-200` | Accordion item border |
| 16 | `text-slate-900` | Accordion trigger text |
| 19 | `text-slate-600` | Accordion content text |
| 26, 36, 46, 57, 70, 80, 90 | `bg-white` | Accordion item background |
| 26, 36, 46, 57, 70, 80, 90 | `border-slate-200` | Accordion item border |
| 27, 37, 47, 58, 71, 81, 91 | `text-slate-900` | Accordion trigger text |
| 30, 40, 50, 61, 74, 84, 94 | `text-slate-600` | Accordion content text |
| 63 | `bg-slate-100` | Inline code background |

---

## Proposed Token Replacements

### Section Layout

```tsx
// Line 5: Section background
className="py-24 bg-secondary"

// Line 8: Section heading
className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance"
```

### Accordion Items

```tsx
// Lines 15, 26, 36, 46, 57, 70, 80, 90: Accordion item containers
className="bg-card border border-border rounded-lg px-6"

// Lines 16, 27, 37, 47, 58, 71, 81, 91: Accordion triggers
className="text-left font-semibold text-card-foreground hover:no-underline"

// Lines 19, 30, 40, 50, 61, 74, 84, 94: Accordion content
className="text-muted-foreground leading-relaxed"
```

### Inline Code

```tsx
// Line 63: Inline code element
className="bg-muted px-2 py-1 rounded text-sm font-mono"
```

---

## New Tokens Required

**None** - All colors map to existing tokens.

---

## Implementation Notes

### Accordion Component

The FAQ uses the shadcn/ui Accordion component. The token replacements should work seamlessly with the component's built-in styles.

### Inline Code Styling

The inline code element (line 63) uses `bg-slate-100` which should map to `bg-muted` for theme consistency.

### Content Structure

Each FAQ item has:
- **Trigger:** Bold, dark text (question)
- **Content:** Regular weight, muted text (answer)

This hierarchy should be preserved with tokens.

---

## Verification Checklist

- [ ] Section background works in both themes
- [ ] Accordion items are readable in both themes
- [ ] Accordion triggers (questions) are prominent
- [ ] Accordion content (answers) is readable
- [ ] Inline code has good contrast
- [ ] Accordion borders are visible
- [ ] Hover states work on triggers
- [ ] Expanded/collapsed states work
- [ ] No hardcoded `slate-*` classes remain

---

## Estimated Complexity

**Low** - Simple, repetitive structure. Straightforward token replacements.

---

## Design Considerations

### Readability

FAQ sections need excellent readability:
- Questions should be scannable (bold, prominent)
- Answers should be easy to read (good line height, contrast)
- Code examples should stand out

### Accordion Behavior

Ensure the accordion component's built-in styles work with tokens:
- Expanded state background
- Hover states
- Focus states
- Transition animations

The shadcn/ui Accordion component should handle these automatically with the token system.

# Dark Mode Quick Reference Card

## Color Tokens (Dark Mode)

```css
/* Canvas & Surfaces */
--background: #0b1220;     /* Canvas */
--foreground: #e5eaf1;     /* Body text */
--card: #141c2a;           /* Card surface */
--popover: #161f2e;        /* Overlay surface */

/* Brand */
--primary: #b45309;        /* Default */
--accent: #d97706;         /* Hover */
--ring: #b45309;           /* Focus */

/* Structure */
--border: #263347;         /* Borders */
--input: #2a3a51;          /* Input shells */
--muted: #0e1726;          /* Chips/rows */
--muted-foreground: #a9b4c5; /* Secondary text */
```

---

## Brand Progression

```tsx
// Button Primary
default:  #b45309
hover:    #d97706
active:   #92400e

// Link
default:  text-[color:var(--accent)]  // #d97706
hover:    text-white + decoration-amber-400

// Ghost Button
hover:    bg-white/[0.04]  // Neutral, not amber
active:   bg-white/[0.06]
```

---

## Input Styling

```tsx
// Background
bg-[#0f172a]

// Placeholder
placeholder:text-[#8b9bb0]

// Inset Shadow
[box-shadow:inset_0_1px_0_rgba(255,255,255,0.04),0_1px_2px_rgba(0,0,0,0.25)]
```

---

## Table Styling

```tsx
// Header
bg-[rgba(255,255,255,0.03)]
text-slate-200

// Row Hover (Neutral)
hover:bg-[rgba(255,255,255,0.025)]

// Selected Row
data-[state=selected]:bg-[rgba(255,255,255,0.04)]
+ focus ring (amber)

// Numeric Cells
text-slate-200 tabular-nums
```

---

## Shadow Tokens

```tsx
// Use CSS variables (auto-adapt to dark mode)
[box-shadow:var(--shadow-sm)]   // Cards
[box-shadow:var(--shadow-md)]   // Popovers
[box-shadow:var(--shadow-lg)]   // Dialogs
```

---

## Amber Restraint Rule

**At view-level, allow ONE amber emphasis domain at a time:**
- CTA **OR** link cluster **OR** data highlight
- **Never all three simultaneously**

---

## Accessibility

```
✅ White on #b45309 ≥ AA (14px)
✅ White on #d97706 ≥ AA (14px)
✅ #e5eaf1 on #0b1220 ≥ AAA
✅ #a9b4c5 on #0e1726 ≥ AA
✅ Focus ring #b45309 over #0b1220 > 3:1
```

---

## Common Patterns

### Card on Canvas
```tsx
<div className="bg-[#0b1220] p-8">
  <Card className="w-[350px]">
    {/* Content */}
  </Card>
</div>
```

### Form Input
```tsx
<input
  placeholder="Enter text..."
  className="placeholder:text-[#8b9bb0] bg-[#0f172a] border-input
    [box-shadow:inset_0_1px_0_rgba(255,255,255,0.04),0_1px_2px_rgba(0,0,0,0.25)]"
/>
```

### Table Row
```tsx
<TableRow
  data-state="selected"
  tabIndex={0}
>
  <TableCell className="text-slate-200 tabular-nums">
    {value}
  </TableCell>
</TableRow>
```

---

## Storybook

View dark mode showcases:
```bash
pnpm --filter @rbee/ui storybook
```

Navigate to:
- **Atoms/Card/Dark Mode**
- **Atoms/Table/Dark Mode**

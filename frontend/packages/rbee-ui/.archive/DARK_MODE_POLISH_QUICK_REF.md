# Dark Mode Polish Quick Reference

## Drift Fix

```tsx
// ❌ BEFORE (hard-coded, can drift)
bg-[#0f172a]

// ✅ AFTER (tracks canvas token)
bg-[color:var(--background)]
```

---

## Selection & Caret

```css
/* Selection (amber-700 wash) */
.dark ::selection {
    background: rgba(185, 83, 9, 0.32);
    color: #ffffff;
}

/* Caret (accent) */
.dark input,
.dark textarea {
    caret-color: #f59e0b;
}
```

---

## Overlay Scrim + Blur

```tsx
// Dialog overlay
bg-black/60 backdrop-blur-[2px]

// Motion
data-[state=open]:fade-in-50
data-[state=closed]:fade-out-50
```

---

## Link States

```tsx
// Default
text-[color:var(--accent)]  // #d97706

// Hover
text-white decoration-amber-300

// Visited
visited:text-[#b45309]

// Focus
focus-visible:ring-[color:var(--ring)]
focus-visible:ring-offset-[color:var(--background)]
```

---

## Inverse Focus (Opt-In)

```tsx
import { focusInverse } from '@rbee/ui/utils/focus-ring'

// Use when element bg ~ amber
<Badge className={focusInverse}>
  Accent Badge
</Badge>
```

---

## Disabled States

```tsx
// All form inputs & buttons
disabled:bg-[#1a2435]
disabled:text-[#6c7a90]
disabled:border-[#223047]
disabled:placeholder:text-[#6c7a90]
```

---

## Sticky Table Headers

```tsx
<TableHead sticky>
  Column Name
</TableHead>

// Applies:
// - sticky top-0 z-10
// - backdrop-blur-[2px]
// - bg-[rgba(20,28,42,0.85)] (dark)
```

---

## Table Row Focus

```tsx
<TableRow tabIndex={0}>
  {/* cells */}
</TableRow>

// Focus ring:
// - ring-2 ring-[color:var(--ring)]
// - ring-offset-2 ring-offset-[color:var(--background)]
```

---

## Scrollbar Styling

```css
/* Automatically applied in dark mode */
.dark ::-webkit-scrollbar {
    height: 10px;
    width: 10px;
}

.dark ::-webkit-scrollbar-thumb {
    background: #263347;
    border-radius: 9999px;
}

.dark ::-webkit-scrollbar-thumb:hover {
    background: #2f3f56;
}
```

---

## Inline Code

```tsx
// Automatically styled in dark mode
<p>Use <code>inline code</code> for commands.</p>

// Applies:
// - font-weight: 500
// - color: #e8eef7
// - background: rgba(255,255,255,0.05)
```

---

## Common Patterns

### Form with Disabled State
```tsx
<Input placeholder="Email" disabled />
// Shows: bg-[#1a2435], text-[#6c7a90]
```

### Link with Visited State
```tsx
<Button variant="link" asChild>
  <a href="/docs">Documentation</a>
</Button>
// After click: text-[#b45309]
```

### Dialog with Scrim
```tsx
<Dialog>
  <DialogTrigger>Open</DialogTrigger>
  <DialogContent>
    {/* Overlay: bg-black/60 backdrop-blur-[2px] */}
  </DialogContent>
</Dialog>
```

### Sticky Table
```tsx
<Table>
  <TableHeader>
    <TableRow>
      <TableHead sticky>ID</TableHead>
      <TableHead sticky>Name</TableHead>
    </TableRow>
  </TableHeader>
  {/* Scrollable body */}
</Table>
```

---

## Storybook

View polish showcases:
```bash
pnpm --filter @rbee/ui storybook
```

Navigate to:
- **Atoms/Forms/Dark Mode Polish**
- **Atoms/Button/Dark Mode Polish**
- **Atoms/Overlays/Dark Mode Polish**
- **Atoms/Table/Dark Mode Sticky**

---

## Verification

```bash
# Build
pnpm --filter @rbee/ui build

# Storybook
pnpm --filter @rbee/ui storybook
```

Test:
1. Select text → See amber-700 wash
2. Type in input → See accent caret
3. Click link → See visited state
4. Open dialog → See scrim + blur
5. Scroll table → See sticky header
6. Tab through rows → See focus ring

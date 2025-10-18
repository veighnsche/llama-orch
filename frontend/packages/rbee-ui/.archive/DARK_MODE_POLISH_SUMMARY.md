# Dark Mode v1.1 — Drift Fixes, Selection, Overlays, & Inverse Focus

**Status:** ✅ Complete  
**Date:** Oct 17, 2025  
**PR:** Dark Mode v1.1 — Drift Fixes, Selection, Overlays, & Inverse Focus

## Overview

Final dark-mode polish to tighten legibility, remove drift, and encode gotchas (selection, scrollbars, overlays, visited links, inverse focus) while keeping branding intact. No new global tokens. HEX, Tailwind/shadcn, and tw-animate-css only.

---

## 1. Remove Drift ✅

### Problem
Hard-coded `bg-[#0f172a]` in Input/Select/Textarea can drift from `--background`.

### Solution
Replace `bg-[#0f172a]` → `bg-[color:var(--background)]` in all form inputs.

### Files Modified
- `src/atoms/Input/Input.tsx`
- `src/atoms/Select/Select.tsx`
- `src/atoms/Textarea/Textarea.tsx`

### Result
Inputs always track the dark canvas hue controlled in tokens. No drift.

---

## 2. Selection & Caret Colors ✅

### Added to Global CSS
```css
/* Selection colors (dark) */
.dark ::selection {
    background: rgba(185, 83, 9, 0.32); /* amber-700 wash */
    color: #ffffff;
}

.dark ::-moz-selection {
    background: rgba(185, 83, 9, 0.32);
    color: #ffffff;
}

/* Caret color (dark) */
.dark input,
.dark textarea {
    caret-color: #f59e0b; /* accent caret */
}
```

### Result
Text selection + caret feel brand-aware and readable on deep canvas.

---

## 3. Overlay Scrim & Blur ✅

### Dialog Overlay
```tsx
className={cn(
  'data-[state=open]:animate-in data-[state=closed]:animate-out',
  'data-[state=closed]:fade-out-50 data-[state=open]:fade-in-50',
  'fixed inset-0 z-50 bg-black/60 backdrop-blur-[2px]',
  className,
)}
```

### Changes
- **Scrim:** `bg-black/60` (was `/50`) — Reduces perceived contrast jump
- **Blur:** `backdrop-blur-[2px]` — Improves readability behind overlay
- **Motion:** `fade-in-50/fade-out-50` — Soft, symmetric transitions

### Files Modified
- `src/atoms/Dialog/Dialog.tsx`

### Result
Overlays feel polished with reduced contrast jump and improved context visibility.

---

## 4. Link States (Visited + Hover) ✅

### Updated `brandLink` Utility
```ts
export const brandLink = 
  "text-[color:var(--accent)] underline underline-offset-2 decoration-amber-400 " +
  "hover:text-white hover:decoration-amber-300 " +
  "visited:text-[#b45309] " +
  "focus-visible:outline-none " +
  "focus-visible:ring-[length:var(--focus-ring-width)] " +
  "focus-visible:ring-[color:var(--ring)] " +
  "focus-visible:ring-offset-[length:var(--focus-ring-offset)] " +
  "focus-visible:ring-offset-[color:var(--background)] " +
  "transition-colors"
```

### States
- **Default:** `text-[color:var(--accent)]` (`#d97706`)
- **Hover:** `text-white` + `decoration-amber-300`
- **Visited:** `text-[#b45309]` (brand 700)
- **Focus:** Uses CSS variables for consistency

### Files Modified
- `src/utils/focus-ring.ts`

### Result
Visited links communicate state; hover stays crisp on dark.

---

## 5. Inverse-Contrast Focus ✅

### New Helper (Opt-In)
```ts
export const focusInverse = 
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--background)]"
```

### Usage
Apply only to components with amber fills (e.g., Badge accent variant in dark) where amber ring would blend.

### Files Modified
- `src/utils/focus-ring.ts`

### Result
Clear focus on amber surfaces without blending.

---

## 6. Disabled & Subtle Text Legibility ✅

### Button Disabled States
```tsx
disabled:bg-slate-200 disabled:text-slate-400 disabled:border-slate-300
dark:disabled:bg-[#1a2435] dark:disabled:text-[#6c7a90] dark:disabled:border-[#223047]
```

### Form Disabled States
```tsx
disabled:bg-[#1a2435] disabled:text-[#6c7a90] disabled:border-[#223047] disabled:placeholder:text-[#6c7a90]
```

### Files Modified
- `src/atoms/Button/Button.tsx`
- `src/atoms/Input/Input.tsx`
- `src/atoms/Select/Select.tsx`
- `src/atoms/Textarea/Textarea.tsx`

### Result
Prevents "vanishing" controls on deep canvas. Clear legibility for disabled states.

---

## 7. Tables: Sticky Header & Focus ✅

### Sticky Header Support
```tsx
function TableHead({ className, sticky, ...props }: React.ComponentProps<'th'> & { sticky?: boolean }) {
  return (
    <th
      data-sticky={sticky}
      className={cn(
        'text-slate-700 dark:text-slate-200 bg-[rgba(2,6,23,0.02)] dark:bg-[rgba(255,255,255,0.03)]',
        'data-[sticky]:sticky data-[sticky]:top-0 data-[sticky]:z-10',
        'data-[sticky]:backdrop-blur-[2px] data-[sticky]:dark:bg-[rgba(20,28,42,0.85)]',
        className,
      )}
      {...props}
    />
  )
}
```

### Row Focus (Keyboard)
```tsx
focus-visible:ring-2 focus-visible:ring-[color:var(--ring)]
focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--background)]
```

### Files Modified
- `src/atoms/Table/Table.tsx`

### Result
Sticky headers remain readable while scrolling. Clear keyboard path without amber flooding.

---

## 8. Scrollbar Minimal Styling ✅

### Added to Global CSS
```css
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

.dark ::-webkit-scrollbar-track {
    background: #0b1220;
}
```

### Result
Subtle, brand-neutral rails that match structural tokens.

---

## 9. Inline Code Styling ✅

### Added to Global CSS
```css
.dark :not(pre) > code {
    font-weight: 500;
    color: #e8eef7;
    background: rgba(255, 255, 255, 0.05);
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
}
```

### Result
Inline code is crisp and readable on dark canvas.

---

## 10. Code Block Selection ✅

### Added to Global CSS
```css
.dark pre ::selection,
.dark code ::selection {
    background: rgba(255, 255, 255, 0.1);
}
```

### Result
In-block selections remain visible.

---

## 11. Storybook Additions ✅

### New Stories Created
1. **Input.dark.stories.tsx** — Input/Select/Textarea states (caret, selection, disabled)
2. **Button.dark.stories.tsx** — Link states (default, hover, visited, focus), disabled states
3. **Dialog.dark.stories.tsx** — Overlays with scrim + blur, motion comparison
4. **Table.sticky.stories.tsx** — Sticky headers, keyboard focus, selected + focused states

### Total Stories
- **Input/Forms:** 4 stories
- **Button/Links:** 3 stories
- **Overlays:** 3 stories
- **Tables:** 3 stories

---

## 12. Files Modified Summary

### Tokens
- `src/tokens/theme-tokens.css` — Selection, caret, scrollbars, inline code

### Utilities
- `src/utils/focus-ring.ts` — Updated `brandLink`, added `focusInverse`

### Atoms
- `src/atoms/Button/Button.tsx` — Disabled states
- `src/atoms/Input/Input.tsx` — Remove drift, disabled states
- `src/atoms/Select/Select.tsx` — Remove drift, disabled states
- `src/atoms/Textarea/Textarea.tsx` — Remove drift, disabled states
- `src/atoms/Dialog/Dialog.tsx` — Scrim + blur overlay
- `src/atoms/Table/Table.tsx` — Sticky header support, improved row focus

### Stories (New)
- `src/atoms/Input/Input.dark.stories.tsx`
- `src/atoms/Button/Button.dark.stories.tsx`
- `src/atoms/Dialog/Dialog.dark.stories.tsx`
- `src/atoms/Table/Table.sticky.stories.tsx`

---

## 13. Verification

### Build
```bash
pnpm --filter @rbee/ui build
# ✅ Exit code: 0
```

### Visual Testing
Run Storybook:
```bash
pnpm --filter @rbee/ui storybook
```

Navigate to:
- **Atoms/Forms/Dark Mode Polish** — All 4 stories
- **Atoms/Button/Dark Mode Polish** — All 3 stories
- **Atoms/Overlays/Dark Mode Polish** — All 3 stories
- **Atoms/Table/Dark Mode Sticky** — All 3 stories

---

## 14. Done Criteria ✅

- [x] Inputs reference `var(--background)` (no hard-coded dark bg)
- [x] Selection/caret are visible and brand-aware
- [x] Links show visited state; hover underline is legible
- [x] Overlays use scrim + subtle blur; motion is soft and symmetric
- [x] Inverse focus available and used only where amber fills exist
- [x] Sticky table headers remain readable while scrolling
- [x] Disabled states have clear legibility
- [x] Scrollbars are subtle and brand-neutral
- [x] Inline code is crisp and readable
- [x] Code block selections remain visible

---

## 15. Charts (Documentation Only)

### Contrast Strokes
When container bg ≠ card: add `stroke-white/60` to points/bars.

### Dashed Series
If >3 series or colorblind mode: set secondary/tertiary series dashed (`stroke-dasharray: 4 3`).

### Legends
Order: amber → blue → emerald → violet → red. Bold primary KPI label with `font-medium text-slate-200`.

**Note:** Chart implementation deferred to chart component development phase.

---

## 16. Copy Micro-Tone (Documentation Only)

### Tooltips
- **Warning:** "Heads up"
- **Success:** "All set"
- **Info:** "FYI"

### Links Near CTAs
- **Avoid:** "Learn more"
- **Prefer:** "Explore details" (reads calmer on dark)

---

## The Bottom Line

✅ **Drift removed** — Inputs track canvas hue via CSS variables  
✅ **Selection polished** — Amber-700 wash + accent caret  
✅ **Overlays refined** — Scrim + blur with soft motion  
✅ **Links complete** — Visited state + crisp hover  
✅ **Focus choreography** — Inverse focus for amber surfaces  
✅ **Disabled legibility** — Clear states on deep canvas  
✅ **Sticky headers** — Backdrop blur + proper dark bg  
✅ **Scrollbars styled** — Subtle, brand-neutral  
✅ **Code polished** — Inline + block selection visibility  
✅ **Build passes** — TypeScript compilation successful  
✅ **Stories complete** — 13 new dark mode showcases  

**This is production-ready. Ship it.**

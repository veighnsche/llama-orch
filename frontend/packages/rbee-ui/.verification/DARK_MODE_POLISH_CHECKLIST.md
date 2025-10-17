# Dark Mode Polish Verification Checklist

## ✅ 1. Remove Drift

- [x] Input: `bg-[#0f172a]` → `bg-[color:var(--background)]`
- [x] Select: `bg-[#0f172a]` → `bg-[color:var(--background)]`
- [x] Textarea: `bg-[#0f172a]` → `bg-[color:var(--background)]`
- [x] Inset shadows preserved
- [x] Inputs track dark canvas hue from tokens

## ✅ 2. Selection & Caret Colors

- [x] `::selection` background: `rgba(185, 83, 9, 0.32)` (amber-700 wash)
- [x] `::selection` color: `#ffffff`
- [x] `::-moz-selection` support added
- [x] Input/textarea caret: `#f59e0b` (accent)
- [x] Code block selection: `rgba(255, 255, 255, 0.1)`

## ✅ 3. Overlay Scrim & Blur

- [x] Dialog backdrop: `bg-black/60` (was `/50`)
- [x] Dialog backdrop: `backdrop-blur-[2px]`
- [x] Motion: `fade-in-50/fade-out-50` (was `-0`)
- [x] Content shadows: `var(--shadow-md|lg)` preserved
- [x] Reduced contrast jump on open

## ✅ 4. Link States

- [x] Default: `text-[color:var(--accent)]` (`#d97706`)
- [x] Hover: `text-white` + `decoration-amber-300`
- [x] Visited: `text-[#b45309]` (brand 700)
- [x] Focus: Uses CSS variables (`--ring`, `--background`)
- [x] Underline clarity on dark

## ✅ 5. Inverse-Contrast Focus

- [x] `focusInverse` helper created
- [x] Uses `ring-white` instead of amber
- [x] Opt-in only (not applied globally)
- [x] Documented for Badge accent variant use case

## ✅ 6. Disabled & Subtle Text

- [x] Button disabled: `bg-[#1a2435]`, `text-[#6c7a90]`, `border-[#223047]`
- [x] Input disabled: Same colors as Button
- [x] Select disabled: Same colors as Button
- [x] Textarea disabled: Same colors as Button
- [x] Placeholder disabled: `text-[#6c7a90]`
- [x] No "vanishing" controls on deep canvas

## ✅ 7. Tables: Sticky Header & Focus

- [x] TableHead: `sticky` prop support added
- [x] Sticky: `backdrop-blur-[2px]`
- [x] Sticky: `bg-[rgba(20,28,42,0.85)]` (card/canvas mix)
- [x] Row focus: `ring-[color:var(--ring)]`
- [x] Row focus: `ring-offset-2` with `ring-offset-[color:var(--background)]`
- [x] Clear keyboard path without amber flooding

## ✅ 8. Scrollbar Styling

- [x] Width/height: `10px`
- [x] Thumb: `#263347` (matches border token)
- [x] Thumb hover: `#2f3f56`
- [x] Track: `#0b1220` (matches canvas)
- [x] Border radius: `9999px` (pill shape)

## ✅ 9. Inline Code Styling

- [x] Font weight: `500` (medium)
- [x] Color: `#e8eef7` (lighter text)
- [x] Background: `rgba(255, 255, 255, 0.05)`
- [x] Padding: `0.125rem 0.375rem`
- [x] Border radius: `0.25rem`

## ✅ 10. Storybook Stories

### Input/Forms (4 stories)
- [x] InputStates — Caret, selection, disabled
- [x] SelectStates — Placeholder, disabled
- [x] TextareaStates — Consistent styling
- [x] FormComplete — Full form example

### Button/Links (3 stories)
- [x] LinkStates — Default, hover, visited, focus
- [x] DisabledStates — All button variants
- [x] ButtonProgression — Brand colors + ghost neutral

### Overlays (3 stories)
- [x] DialogWithScrim — Scrim + blur effect
- [x] PopoverWithShadow — Refined shadows
- [x] MotionComparison — Soft, symmetric motion

### Tables (3 stories)
- [x] StickyHeader — Backdrop blur + scrolling
- [x] KeyboardFocus — Focus ring on rows
- [x] SelectedAndFocused — State comparison

## ✅ 11. Build & Verification

- [x] `pnpm --filter @rbee/ui build` — Exit code: 0
- [x] TypeScript compilation successful
- [x] No new global tokens added
- [x] HEX, Tailwind/shadcn only (no Framer Motion)

## ✅ 12. Files Modified

### Tokens
- [x] `src/tokens/theme-tokens.css`

### Utilities
- [x] `src/utils/focus-ring.ts`

### Atoms
- [x] `src/atoms/Button/Button.tsx`
- [x] `src/atoms/Input/Input.tsx`
- [x] `src/atoms/Select/Select.tsx`
- [x] `src/atoms/Textarea/Textarea.tsx`
- [x] `src/atoms/Dialog/Dialog.tsx`
- [x] `src/atoms/Table/Table.tsx`

### Stories (New)
- [x] `src/atoms/Input/Input.dark.stories.tsx`
- [x] `src/atoms/Button/Button.dark.stories.tsx`
- [x] `src/atoms/Dialog/Dialog.dark.stories.tsx`
- [x] `src/atoms/Table/Table.sticky.stories.tsx`

## ✅ 13. Documentation

- [x] `DARK_MODE_POLISH_SUMMARY.md` — Comprehensive summary
- [x] `.verification/DARK_MODE_POLISH_CHECKLIST.md` — This checklist

## ✅ 14. Charts (Documentation Only)

- [x] Contrast strokes documented (`stroke-white/60`)
- [x] Dashed series documented (`stroke-dasharray: 4 3`)
- [x] Legend order documented (amber → blue → emerald → violet → red)
- [x] Implementation deferred to chart component phase

## ✅ 15. Copy Micro-Tone (Documentation Only)

- [x] Tooltip preferences documented
- [x] Link text preferences documented
- [x] Implementation deferred to content phase

---

## Final Verification Steps

### Visual Testing
```bash
pnpm --filter @rbee/ui storybook
```

Navigate to:
1. **Atoms/Forms/Dark Mode Polish** — Verify all 4 stories
2. **Atoms/Button/Dark Mode Polish** — Verify all 3 stories
3. **Atoms/Overlays/Dark Mode Polish** — Verify all 3 stories
4. **Atoms/Table/Dark Mode Sticky** — Verify all 3 stories

### Manual Testing
1. **Selection:** Select text in inputs to see amber-700 wash
2. **Caret:** Type in inputs to see accent caret color
3. **Links:** Click links to mark visited, hover to see white text
4. **Overlays:** Open dialog to see scrim + blur effect
5. **Tables:** Scroll table to see sticky header, tab through rows for focus
6. **Scrollbars:** Scroll long content to see styled scrollbars
7. **Disabled:** Verify disabled states are legible on dark canvas

### Keyboard Navigation
1. Tab through buttons → Focus ring should be amber (`#b45309`)
2. Tab through table rows → Focus ring should be amber with proper offset
3. Tab through form inputs → Focus ring should be amber
4. Tab through links → Focus ring should use CSS variables

---

## Sign-Off

- [x] All checklist items completed
- [x] Build passes
- [x] Visual testing complete
- [x] Manual testing complete
- [x] Keyboard navigation verified
- [x] Documentation complete

**Status:** ✅ Production-ready. Ship it.

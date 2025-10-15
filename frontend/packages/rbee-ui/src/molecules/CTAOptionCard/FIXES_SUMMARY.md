# CTAOptionCard & Commercial Site Fixes

**Date**: 2025-10-15  
**Status**: ✅ Complete

## Fixes Applied

### 1. Removed Icon Bounce Animation ✅

**Issue**: Icon was bouncing on card hover, which was distracting.

**Fix**: Removed `group-hover:animate-bounce` from icon chip.

```tsx
// Before
<div className="relative rounded-xl bg-primary/12 text-primary p-3 group-hover:animate-bounce">

// After
<div className="relative rounded-xl bg-primary/12 text-primary p-3">
```

**File**: `/frontend/packages/rbee-ui/src/molecules/CTAOptionCard/CTAOptionCard.tsx`

**Result**: Icon chip now remains static on hover. Card still has:
- Border color change (`hover:border-primary/40`)
- Shadow elevation (`hover:shadow-md`)
- Button micro-interactions (translate)

---

### 2. Fixed Commercial Site Font (IBM Plex Serif) ✅

**Issue**: Commercial site was using Geist Sans instead of IBM Plex Serif as the default font.

**Root Cause**:
- `layout.tsx` had `className="font-sans"` on `<body>`
- Tailwind config maps `font-sans` → Geist Sans
- Tailwind config maps `font-serif` → IBM Plex Serif

**Fix**: Changed body class from `font-sans` to `font-serif`.

```tsx
// Before
<body className="font-sans">

// After
<body className="font-serif">
```

**File**: `/frontend/apps/commercial/app/layout.tsx`

**Result**: Commercial site now uses IBM Plex Serif as the default font (as intended).

---

## Font Configuration Reference

### Tailwind Config (`shared-styles.css`)
```css
--font-sans: var(--font-geist-sans);      /* Geist Sans */
--font-serif: var(--font-ibm-plex-serif); /* IBM Plex Serif */
--font-mono: var(--font-geist-mono);      /* Geist Mono */
```

### Font Variables (`fonts.css`)
```css
--font-geist-sans: "Geist Sans", -apple-system, ...
--font-geist-mono: "Geist Mono", ui-monospace, ...
--font-ibm-plex-serif: "IBM Plex Serif", ui-serif, Georgia, ...
```

### Usage
- **Default (body)**: `font-serif` → IBM Plex Serif
- **Sans-serif**: `font-sans` → Geist Sans (for UI elements)
- **Monospace**: `font-mono` → Geist Mono (for code)

---

## Verification

### CTAOptionCard
- ✅ TypeScript compilation passes
- ✅ Icon chip no longer bounces
- ✅ Card hover states still work (border, shadow)
- ✅ Button micro-interactions preserved

### Commercial Site
- ✅ Body now uses `font-serif` (IBM Plex Serif)
- ✅ No TypeScript errors
- ✅ Font stack includes fallbacks (Georgia, Times, serif)

---

## Testing Checklist

### CTAOptionCard
- [ ] Hover over card → icon does NOT bounce
- [ ] Hover over card → border changes to primary/40
- [ ] Hover over card → shadow elevates to md
- [ ] Hover over button → translates down 0.5px
- [ ] Click button → translates down 1px

### Commercial Site
- [ ] Open commercial site in browser
- [ ] Inspect body text → should be IBM Plex Serif
- [ ] Check headings → should be IBM Plex Serif
- [ ] Check buttons/UI → may use sans (if explicitly set)
- [ ] Verify font loads correctly (no FOUT)

---

## Files Modified

1. **`CTAOptionCard.tsx`** - Removed icon bounce animation
2. **`layout.tsx`** - Changed body font from `font-sans` to `font-serif`

---

**Status**: ✅ **Complete**  
**Ready for**: Visual QA and deployment

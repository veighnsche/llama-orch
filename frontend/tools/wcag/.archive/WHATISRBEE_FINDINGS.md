# WhatIsRbee WCAG Compliance Findings

**Component:** `organisms/Home/WhatIsRbee/WhatIsRbee.tsx`

## Summary

✅ **Most colors pass** WCAG AA standards  
❌ **1 issue found:** `text-muted-foreground` on `bg-secondary` fails AA for normal text

---

## Detailed Analysis

### ✅ PASS: Foreground on Secondary Background

**Combination:** `text-foreground` (#0f172a) on `bg-secondary` (#f1f5f9)  
**Contrast Ratio:** 16.30:1  
**Status:** 🌟 AAA compliant for all text sizes

**Used in:**
- Main heading text
- Body paragraphs with `text-foreground`

---

### ✅ PASS: Primary Color

**Combination:** `text-primary` (#f59e0b) on `text-foreground` (#0f172a)  
**Contrast Ratio:** 8.31:1  
**Status:** 🌟 AAA compliant for all text sizes

**Used in:**
- IconBox components (via FeatureListItem)
- Brand accent colors

---

### ❌ FAIL: Muted Text on Secondary Background

**Combination:** `text-muted-foreground` (#64748b) on `bg-secondary` (#f1f5f9)  
**Contrast Ratio:** 4.34:1  
**Status:** ⚠️ **FAILS AA for normal text** (requires 4.5:1)

**Light Mode:**
- Colors: #64748b on #f1f5f9
- Ratio: 4.34:1
- AA Normal: ❌ FAIL (needs 4.5:1)
- AA Large: ✅ PASS (needs 3:1)

**Dark Mode:**
- Colors: #94a3b8 on #1e293b
- Ratio: 5.71:1
- AA Normal: ✅ PASS
- AA Large: ✅ PASS

**Used in WhatIsRbee:**
1. Line 53: Subhead paragraph (`text-lg md:text-xl text-muted-foreground`)
   - Size: 18-20px ✅ Large enough (PASS)
2. Line 114: Closing micro-copy (`text-base text-muted-foreground`)
   - Size: 16px ❌ Too small (FAIL)
3. Line 139: Technical accent (`text-xs text-muted-foreground`)
   - Size: 12px ❌ Too small (FAIL)

---

## Recommendations

### Option 1: Change Text Color (Recommended)

Replace `text-muted-foreground` with `text-foreground` for small text:

```tsx
// ❌ BEFORE (Line 114)
<p className="text-base text-muted-foreground leading-relaxed max-w-prose">
  Build AI coders on your own hardware...
</p>

// ✅ AFTER
<p className="text-base text-foreground leading-relaxed max-w-prose">
  Build AI coders on your own hardware...
</p>
```

```tsx
// ❌ BEFORE (Line 139)
<p className="text-xs text-muted-foreground pt-2 font-sans">
  <strong>Architecture at a glance:</strong> Smart/Dumb separation...
</p>

// ✅ AFTER
<p className="text-xs text-foreground pt-2 font-sans">
  <strong>Architecture at a glance:</strong> Smart/Dumb separation...
</p>
```

### Option 2: Increase Text Size

Make text large enough (≥18pt) to meet AA Large standard:

```tsx
// Change from text-base (16px) to text-lg (18px)
<p className="text-lg text-muted-foreground leading-relaxed max-w-prose">
  Build AI coders on your own hardware...
</p>
```

### Option 3: Update Design Token

Darken `muted-foreground` in light mode to achieve 4.5:1 contrast:

```css
/* In theme-tokens.css */
:root {
  --muted-foreground: #5a6b7f; /* Darker than current #64748b */
}
```

**Note:** This would affect all components using `muted-foreground`.

---

## Global Impact

The `text-muted-foreground` on `bg-secondary` combination is used in **20 locations** across the codebase:

- UseCasesSection.tsx
- PricingComparison.tsx
- UseCasesIndustry.tsx
- HowItWorksSection.tsx
- WhatIsRbee.tsx
- ... and 15 more

**Recommendation:** Consider Option 3 (update design token) for a global fix, or Option 1 (change to `text-foreground`) for case-by-case fixes.

---

## Testing

Run the WCAG checker to verify fixes:

```bash
cd frontend/tools/wcag
python check_components.py --verbose
```

Check specific color combinations:

```bash
# Current (fails)
python -c "
from wcag_utils import parse_color, get_contrast_ratio, check_wcag_compliance
text = parse_color('#64748b')
bg = parse_color('#f1f5f9')
ratio = get_contrast_ratio(text, bg)
print(f'Ratio: {ratio:.2f}:1')
print(f'AA Normal: {check_wcag_compliance(ratio)[\"AA_normal\"]}')"

# Proposed fix (darker muted-foreground)
python -c "
from wcag_utils import parse_color, get_contrast_ratio, check_wcag_compliance
text = parse_color('#5a6b7f')
bg = parse_color('#f1f5f9')
ratio = get_contrast_ratio(text, bg)
print(f'Ratio: {ratio:.2f}:1')
print(f'AA Normal: {check_wcag_compliance(ratio)[\"AA_normal\"]}')"
```

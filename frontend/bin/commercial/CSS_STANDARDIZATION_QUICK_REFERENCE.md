# CSS Token Standardization - Quick Reference

**TL;DR:** 236 unique token variants across 148 files need standardization. Estimated: 72-110 hours (2-3 weeks).

---

## By The Numbers

| Metric | Value |
|--------|-------|
| **Files to update** | 148 |
| **Lines of code** | 14,488 |
| **Total token uses** | 7,658 |
| **Unique variants** | 236 |

---

## Priority Matrix

| Priority | Category | Variants | Uses | Effort | Files |
|----------|----------|----------|------|--------|-------|
| 🔴 **CRITICAL** | Colors | 76 | 3,373 | Very Large | 103+ |
| 🟠 **HIGH** | Spacing | 89 | 1,774 | Large | 148 |
| 🟠 **HIGH** | Text Sizing | 11 | 760 | Medium | 83+ |
| 🟡 **MEDIUM** | Sizing | 35 | 781 | Large | 148 |
| 🟡 **MEDIUM** | Border Radius | 13 | 447 | Small | 59+ |
| 🟡 **MEDIUM** | Shadows | 8 | 35 | Small | 9+ |
| 🟢 **LOW** | Font Weights | 4 | 488 | Small | 63+ |

---

## Top Issues to Fix

### 1. Hard-coded Colors (CRITICAL)
- `bg-red-500`, `bg-amber-500`, `bg-green-500` → Need semantic tokens
- `text-red-300`, `text-red-50` → Need error state tokens
- `text-slate-950` (5 uses) → Should use `text-foreground`

### 2. Spacing Outliers (HIGH)
- `gap-7` (1 use) → Normalize to `gap-6` or `gap-8`
- `py-20` (1 use) → Normalize to `py-16` or `py-24`
- Inconsistent section padding

### 3. Ambiguous Tokens (MEDIUM)
- `rounded` (49 uses) → Replace with explicit `rounded-lg` or `rounded-md`
- `shadow` (8 uses) → Replace with explicit size

---

## Tokens to Add to globals.css

### Colors
```css
/* Terminal colors */
--terminal-red: oklch(0.577 0.245 27.325);
--terminal-amber: oklch(0.828 0.189 84.429);
--terminal-green: oklch(0.769 0.188 70.08);

/* Error states */
--error-light: oklch(0.637 0.237 25.331);
--error-dark: oklch(0.396 0.141 25.723);
```

### Spacing Scale
```css
/* Document existing scale */
--spacing-0: 0;
--spacing-1: 0.25rem;  /* 4px */
--spacing-2: 0.5rem;   /* 8px */
--spacing-3: 0.75rem;  /* 12px */
--spacing-4: 1rem;     /* 16px */
--spacing-6: 1.5rem;   /* 24px */
--spacing-8: 2rem;     /* 32px */
--spacing-12: 3rem;    /* 48px */
--spacing-16: 4rem;    /* 64px */
--spacing-20: 5rem;    /* 80px */
--spacing-24: 6rem;    /* 96px */
--spacing-32: 8rem;    /* 128px */
```

### Icon Sizes
```css
--icon-xs: 1rem;      /* 16px - h-4, w-4 */
--icon-sm: 1.25rem;   /* 20px - h-5, w-5 */
--icon-md: 1.5rem;    /* 24px - h-6, w-6 */
--icon-lg: 2rem;      /* 32px - h-8, w-8 */
--icon-xl: 3rem;      /* 48px - h-12, w-12 */
```

### Border Radius
```css
--radius-xs: calc(var(--radius) - 6px);  /* ~4px */
/* (sm, md, lg, xl already exist) */
```

### Shadows
```css
--shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05);
--shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
--shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
--shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
--shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
--shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
```

---

## Component Hotspots

**Files with most className usage:**
1. `organisms/Features/core-features-tabs.tsx` (97 uses)
2. `organisms/Pricing/pricing-tiers.tsx` (93 uses)
3. `organisms/Pricing/pricing-comparison.tsx` (88 uses)
4. `organisms/Providers/providers-marketplace.tsx` (73 uses)
5. `organisms/Enterprise/enterprise-security.tsx` (72 uses)

---

## Phased Rollout

### Week 1
- **Phase 1:** Color standardization (3-4 days)
- **Phase 2:** Spacing standardization (2-3 days)

### Week 2
- **Phase 3:** Text sizing standardization (1-2 days)
- **Phase 4:** Sizing/radius/shadow (2-3 days)

### Week 3
- **Phase 5:** Documentation & testing (2-3 days)
- **Buffer:** Unexpected issues (1-2 days)

---

## Success Criteria

✅ **Token reduction:** 236 → ~50 (80% reduction)  
✅ **Zero hard-coded colors**  
✅ **All spacing uses standard scale**  
✅ **Comprehensive documentation**  
✅ **Visual regression tests pass**

---

## Files Generated

1. **CSS_STANDARDIZATION_MASTER_PLAN.md** - Complete implementation plan
2. **CSS_TOKEN_ANALYSIS.md** - Detailed token usage data
3. **CSS_STANDARDIZATION_WORK_PLAN.md** - Work breakdown by category
4. **CSS_STANDARDIZATION_QUICK_REFERENCE.md** - This document
5. **analyze_css_tokens.py** - Analysis script (reusable)

---

## Next Action

```bash
# Review the master plan
cat CSS_STANDARDIZATION_MASTER_PLAN.md

# Start Phase 1: Color standardization
git checkout -b feat/css-token-standardization
```

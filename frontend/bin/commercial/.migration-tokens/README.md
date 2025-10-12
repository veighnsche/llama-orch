# CSS Token Migration Work Units

This directory contains the migration plan to replace all hardcoded Tailwind colors with CSS tokens from `globals.css` to enable dark/light mode support.

## 🚨 HOW TO USE THIS README

**BEFORE YOU START WORK:**
1. Read `00_MASTER_PLAN.md` to understand the token system
2. Find the FIRST unchecked `[ ]` work unit below
3. Open that work unit's `.md` file and follow its checklist
4. Implement the changes in the actual component files
5. **IMMEDIATELY** come back here and check off `[x]` the work unit you completed
6. Move to the next unchecked `[ ]` work unit

**DO NOT:**
- ❌ Skip ahead to random work units
- ❌ Leave work units checked `[x]` if they're not actually implemented
- ❌ Forget to update this README after completing work

**This README is the SOURCE OF TRUTH for what's done and what's next.**

## Quick Start

1. **Read the Master Plan:** Start with `00_MASTER_PLAN.md`
2. **Pick a work unit:** Choose the FIRST unchecked `[ ]` item below
3. **Follow the checklist:** Each work unit has a verification checklist
4. **Test both themes:** Always verify light AND dark modes
5. **Update this README:** Check off `[x]` the work unit immediately after implementation

## Work Units (Priority Order)

**INSTRUCTIONS:** Work from top to bottom. Check off `[x]` ONLY after implementing in actual component files.

### High Priority (Always Visible)
- [x] `01_navigation_footer.md` - Navigation & Footer
- [x] `02_hero_section.md` - Hero Section
- [x] `03_cta_section.md` - CTA Section

### Medium Priority (Core Content)
- [x] `04_features_section.md` - Features Section
- [x] `05_problem_section.md` - Problem Section
- [x] `06_solution_section.md` - Solution Section
- [x] `07_comparison_section.md` - Comparison Section
- [x] `08_pricing_section.md` - Pricing Section

### Low Priority (Supporting Content)
- [x] `09_faq_section.md` - FAQ Section
- [x] `10_social_proof_use_cases.md` - Social Proof & Use Cases
- [x] `11_technical_how_it_works.md` - Technical & How It Works
- [x] `12_email_audience.md` - Email Capture & Audience Selector
- [x] `13_what_is_rbee.md` - What Is Rbee

### Specialized Pages
- [x] `14_developers_components.md` - Developers Page (10 components)
- [ ] `15_enterprise_components.md` - Enterprise Page (11 components)
- [ ] `16_features_components.md` - Features Page (9 components)

## Token Reference

### Available Tokens (from globals.css)

**Semantic Colors:**
- `--background` / `--foreground`
- `--card` / `--card-foreground`
- `--primary` / `--primary-foreground`
- `--secondary` / `--secondary-foreground`
- `--muted` / `--muted-foreground`
- `--accent` / `--accent-foreground`
- `--destructive` / `--destructive-foreground`
- `--border` / `--input` / `--ring`

**Chart Colors:**
- `--chart-1` (amber) - Primary brand
- `--chart-2` (blue) - Info, security
- `--chart-3` (green) - Success, compliance
- `--chart-4` (purple) - Keywords, special
- `--chart-5` (red) - Errors, warnings

### Common Replacements

```tsx
// Backgrounds
bg-slate-50 → bg-secondary
bg-white → bg-background or bg-card
bg-slate-900 → bg-background (dark mode)

// Text
text-slate-900 → text-foreground
text-slate-600 → text-muted-foreground
text-amber-500 → text-primary
text-green-600 → text-chart-3
text-blue-600 → text-chart-2
text-red-500 → text-destructive

// Borders
border-slate-200 → border-border

// Buttons
bg-amber-500 hover:bg-amber-600 → bg-primary hover:bg-primary/90
```

## New Tokens to Add (Optional)

If you need more semantic tokens, add them to `globals.css`:

```css
:root {
  --success: #10b981;
  --success-foreground: #ffffff;
  --info: #3b82f6;
  --info-foreground: #ffffff;
  --warning: #f59e0b;
  --warning-foreground: #ffffff;
}

.dark {
  --success: #22c55e;
  --success-foreground: #0f172a;
  --info: #60a5fa;
  --info-foreground: #0f172a;
  --warning: #fbbf24;
  --warning-foreground: #0f172a;
}
```

## Testing Dark Mode

Add the `dark` class to the root element:

```tsx
<html className="dark">
```

Or use the theme provider in `components/theme-provider.tsx`.

## Progress Tracking

**UPDATE THIS SECTION AS YOU COMPLETE WORK UNITS:**

- [x] High Priority (3/3 units complete)
- [x] Medium Priority (5/5 units complete)
- [x] Low Priority (5/5 units complete)
- [ ] Specialized Pages (1/3 pages complete)

**Overall Progress:** 14/16 work units complete (88%)

## Design Decisions

### Always Dark vs Theme-Adaptive

Some sections may look better always dark:
- **CTA sections:** Conversion-optimized, stands out
- **Email capture:** Creates urgency
- **Hero sections:** Dramatic impact

**Decision:** Document in each work unit whether to keep dark or make theme-adaptive.

### Gradient Handling

Complex gradients can be:
1. **Simplified:** Use solid backgrounds with tokens
2. **Tokenized:** Create gradient tokens
3. **Kept as-is:** If always dark, keep hardcoded

**Recommendation:** Start simple, add gradients later if needed.

### Decorative Elements

Some elements are purely decorative:
- Avatar gradients in testimonials
- macOS window buttons (red/amber/green dots)
- Decorative icons

**Decision:** These can remain hardcoded if they don't need theme adaptation.

## Verification Commands

```bash
# Search for remaining hardcoded colors
grep -r "slate-" components/
grep -r "amber-[0-9]" components/
grep -r "blue-[0-9]" components/
grep -r "green-[0-9]" components/
grep -r "red-[0-9]" components/

# Test build
npm run build

# Start dev server
npm run dev
```

## ⚠️ CRITICAL: Smart Token Decisions

**DO NOT port colors 1:1.** The current Tailwind colors are inconsistent and need consolidation.

### Think Semantically, Not Literally

Ask yourself:
- **What does this color mean?** (not "what shade is it")
- **Can I reuse an existing token?** (consolidate similar colors)
- **Is this consistent with other components?** (fix inconsistencies)

### Examples of Smart Decisions

```tsx
// ❌ BAD: Porting every shade
text-slate-300 → text-slate-300
text-slate-400 → text-slate-400
text-slate-600 → text-slate-600

// ✅ GOOD: Consolidate to semantic meaning
text-slate-300 → text-muted-foreground
text-slate-400 → text-muted-foreground
text-slate-600 → text-muted-foreground
```

### Questions?

If you're unsure about a color replacement:
1. Check the Master Plan for common patterns
2. Look at similar components that are already migrated
3. **Prioritize semantic meaning over exact color match**
4. **Consolidate similar shades to the same token**
5. When in doubt, use `muted-foreground` for de-emphasized text
6. **Don't create new tokens unless absolutely necessary**

## Completion Criteria

A work unit is complete when:
- ✅ All hardcoded colors replaced with tokens
- ✅ Component renders correctly in light mode
- ✅ Component renders correctly in dark mode
- ✅ Semantic meaning preserved
- ✅ Verification checklist completed
- ✅ No console errors or warnings

---

**Total Components:** ~50 components across 16 work units  
**Estimated Time:** 2-4 hours per work unit (depending on complexity)  
**Total Effort:** ~40-80 hours

Good luck! 🎨

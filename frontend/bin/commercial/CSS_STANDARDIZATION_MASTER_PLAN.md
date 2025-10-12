# CSS Token Standardization Master Plan

**Project:** Orchyra Commercial Frontend  
**Objective:** Standardize CSS tokens across all components to improve style alignment and maintainability  
**Date:** 2025-01-12

---

## Executive Summary

This document outlines the comprehensive work required to standardize CSS tokens across the commercial frontend codebase. The analysis reveals **236 unique token variants** used **7,658 times** across **148 component files** (14,488 lines of code).

### Critical Findings

- **76 color token variants** (3,373 uses) - CRITICAL priority
- **89 spacing token variants** (1,774 uses) - HIGH priority  
- **11 text sizing variants** (760 uses) - HIGH priority
- **35 sizing token variants** (781 uses) - MEDIUM priority
- **13 border radius variants** (447 uses) - MEDIUM priority

### Estimated Effort

**Total Time:** 72-110 hours (2-3 weeks of focused work)

---

## Component Structure Overview

### Directory Tree

```
components/
├── atoms/ (60 components)
│   ├── Accordion, Alert, AlertDialog, AspectRatio, Avatar
│   ├── Badge, Breadcrumb, Button, Calendar, Card
│   ├── Carousel, Chart, Checkbox, Collapsible, Command
│   ├── ContextMenu, Dialog, Drawer, DropdownMenu, Form
│   ├── HoverCard, Input, InputGroup, InputOtp, Item
│   ├── Kbd, Label, Menubar, NavigationMenu, Pagination
│   ├── Popover, Progress, RadioGroup, Resizable, ScrollArea
│   ├── Select, Separator, Sheet, Sidebar, Skeleton
│   ├── Slider, Sonner, Spinner, Switch, Table
│   ├── Tabs, Textarea, Toast, Toaster, Toggle
│   ├── ToggleGroup, Tooltip, UseMobile, UseToast
│
├── molecules/ (26 components)
│   ├── ArchitectureDiagram, AudienceCard, BenefitCallout
│   ├── BulletListItem, CheckListItem, CodeBlock
│   ├── ComparisonTableRow, EarningsCard, FeatureCard
│   ├── FooterColumn, GPUListItem, IconBox
│   ├── NavLink, PricingTier, ProgressBar
│   ├── PulseBadge, SectionContainer, SecurityCrateCard
│   ├── StatCard, StepNumber, TabButton
│   ├── TerminalWindow, TestimonialCard, ThemeToggle
│   ├── TrustIndicator, UseCaseCard
│
├── organisms/ (58 components)
│   ├── AudienceSelector, ComparisonSection, CtaSection
│   ├── EmailCapture, FaqSection, FeaturesSection
│   ├── Footer, HeroSection, HowItWorksSection
│   ├── Navigation, PricingSection, ProblemSection
│   ├── SocialProofSection, SolutionSection, TechnicalSection
│   ├── UseCasesSection, WhatIsRbee
│   │
│   ├── Developers/ (10 components)
│   │   ├── developers-code-examples, developers-cta
│   │   ├── developers-features, developers-hero
│   │   ├── developers-how-it-works, developers-pricing
│   │   ├── developers-problem, developers-solution
│   │   ├── developers-testimonials, developers-use-cases
│   │
│   ├── Enterprise/ (11 components)
│   │   ├── enterprise-comparison, enterprise-compliance
│   │   ├── enterprise-cta, enterprise-features
│   │   ├── enterprise-hero, enterprise-how-it-works
│   │   ├── enterprise-problem, enterprise-security
│   │   ├── enterprise-solution, enterprise-testimonials
│   │   ├── enterprise-use-cases
│   │
│   ├── Features/ (9 components)
│   │   ├── additional-features-grid, core-features-tabs
│   │   ├── cross-node-orchestration, error-handling
│   │   ├── features-hero, intelligent-model-management
│   │   ├── multi-backend-gpu, real-time-progress
│   │   ├── security-isolation
│   │
│   ├── Pricing/ (4 components)
│   │   ├── pricing-comparison, pricing-faq
│   │   ├── pricing-hero, pricing-tiers
│   │
│   ├── Providers/ (11 components)
│   │   ├── providers-cta, providers-earnings
│   │   ├── providers-features, providers-hero
│   │   ├── providers-how-it-works, providers-marketplace
│   │   ├── providers-problem, providers-security
│   │   ├── providers-solution, providers-testimonials
│   │   ├── providers-use-cases
│   │
│   └── UseCases/ (3 components)
│       ├── use-cases-hero, use-cases-industry
│       ├── use-cases-primary
│
├── providers/ (2 components)
│   └── ThemeProvider
│
└── templates/ (0 components)
```

### Code Metrics (via cloc)

```
Language: TypeScript TSX
Files:    148
Blank:    1,149 lines
Comment:  99 lines
Code:     13,533 lines
Total:    14,781 lines
```

**Top 10 Largest Components:**
1. `organisms/Features/core-features-tabs.tsx` - 215 lines
2. `organisms/Pricing/pricing-tiers.tsx` - 209 lines
3. `organisms/Pricing/pricing-comparison.tsx` - 206 lines
4. `organisms/Providers/providers-marketplace.tsx` - 172 lines
5. `organisms/Enterprise/enterprise-security.tsx` - 169 lines
6. `organisms/Providers/providers-security.tsx` - 169 lines
7. `organisms/ComparisonSection/ComparisonSection.tsx` - 165 lines
8. `organisms/Providers/providers-solution.tsx` - 165 lines
9. `organisms/Enterprise/enterprise-features.tsx` - 158 lines
10. `organisms/Providers/providers-hero.tsx` - 158 lines

---

## Current Token Analysis

### 1. Color Tokens (CRITICAL)

**Status:** 76 variants, 3,373 total uses across 103+ files

#### Most Used Color Tokens

| Token | Uses | Files | Status |
|-------|------|-------|--------|
| `text-muted-foreground` | 571 | 103 | ✅ Already in globals.css |
| `text-foreground` | 373 | 73 | ✅ Already in globals.css |
| `border-border` | 265 | 68 | ✅ Already in globals.css |
| `bg-primary` | 165 | 47 | ✅ Already in globals.css |
| `text-primary` | 165 | 53 | ✅ Already in globals.css |
| `text-chart-3` | 150 | 28 | ✅ Already in globals.css |
| `bg-card` | 134 | 48 | ✅ Already in globals.css |
| `text-card-foreground` | 45 | 16 | ✅ Already in globals.css |

#### Tokens NOT in globals.css (Need Standardization)

**Hard-coded colors (should use design tokens):**
- `bg-red-500`, `bg-amber-500`, `bg-green-500` (terminal window dots)
- `text-red-300`, `text-red-50` (error states)
- `text-slate-950` (5 uses)
- `bg-white`, `bg-black` (should use semantic tokens)

**Missing semantic tokens:**
- `text-chart-2` (13 uses), `text-chart-4` (6 uses)
- `bg-chart-2` (3 uses)
- `border-chart-2` (1 use), `border-chart-3` (12 uses)

**Action Required:**
1. Add missing chart color tokens to globals.css
2. Replace hard-coded colors with semantic tokens
3. Add semantic tokens for terminal window dots (e.g., `--terminal-red`, `--terminal-amber`, `--terminal-green`)
4. Add error state tokens (e.g., `--error-light`, `--error-dark`)

---

### 2. Spacing Tokens (HIGH)

**Status:** 89 variants, 1,774 total uses

#### Most Used Spacing Tokens

| Category | Variants | Uses | Top Token | Top Uses |
|----------|----------|------|-----------|----------|
| Padding | 8 | 322 | `p-4` | 161 |
| Padding X | 6 | 138 | `px-6` | 51 |
| Padding Y | 10 | 115 | `py-24` | 40 |
| Margin Bottom | 9 | 406 | `mb-4` | 137 |
| Margin Top | 9 | 157 | `mt-0` | 41 |
| Gap | 10 | 389 | `gap-2` | 157 |
| Space Y | 8 | 143 | `space-y-2` | 36 |

#### Standardization Opportunities

**Current spacing scale in use:**
- 0, 1, 2, 3, 4, 6, 7, 8, 12, 16, 20, 24, 32

**Recommended standard scale (add to globals.css):**
```css
:root {
  /* Spacing scale */
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
}
```

**Action Required:**
1. Document spacing scale in globals.css
2. Identify outliers (e.g., `gap-7`, `py-20`) and normalize
3. Create spacing guidelines for component padding/margin

---

### 3. Text Sizing (HIGH)

**Status:** 11 variants, 760 total uses

#### Current Text Size Usage

| Token | Uses | Files | Standardized? |
|-------|------|-------|---------------|
| `text-sm` | 376 | 83 | ✅ Common |
| `text-xl` | 103 | 42 | ✅ Common |
| `text-xs` | 64 | 21 | ✅ Common |
| `text-2xl` | 61 | 26 | ✅ Common |
| `text-lg` | 50 | 22 | ✅ Common |
| `text-4xl` | 47 | 35 | ✅ Common |
| `text-3xl` | 25 | 13 | ✅ Common |
| `text-5xl` | 23 | 22 | ✅ Common |
| `text-6xl` | 6 | 6 | ⚠️ Rare |
| `text-base` | 3 | 3 | ⚠️ Rare |
| `text-7xl` | 2 | 2 | ⚠️ Rare |

**Recommended typography scale (add to globals.css):**
```css
:root {
  /* Typography scale */
  --text-xs: 0.75rem;    /* 12px */
  --text-sm: 0.875rem;   /* 14px */
  --text-base: 1rem;     /* 16px */
  --text-lg: 1.125rem;   /* 18px */
  --text-xl: 1.25rem;    /* 20px */
  --text-2xl: 1.5rem;    /* 24px */
  --text-3xl: 1.875rem;  /* 30px */
  --text-4xl: 2.25rem;   /* 36px */
  --text-5xl: 3rem;      /* 48px */
  --text-6xl: 3.75rem;   /* 60px */
  --text-7xl: 4.5rem;    /* 72px */
}
```

**Action Required:**
1. Document typography scale in globals.css
2. Create component-specific text size guidelines (e.g., buttons use `text-sm`, headings use `text-4xl+`)

---

### 4. Sizing Tokens (MEDIUM)

**Status:** 35 variants, 781 total uses

#### Most Used Size Tokens

| Token | Uses | Pattern |
|-------|------|---------|
| `h-5`, `w-5` | 113 each | Icon size (20px) |
| `h-12`, `w-12` | 55, 51 | Large icon/button (48px) |
| `h-6`, `w-6` | 51 each | Medium icon (24px) |
| `h-8`, `w-8` | 39, 22 | Small icon/button (32px) |
| `h-4`, `w-4` | 38, 36 | Tiny icon (16px) |

**Recommended icon size tokens (add to globals.css):**
```css
:root {
  /* Icon sizes */
  --icon-xs: 1rem;      /* 16px - h-4, w-4 */
  --icon-sm: 1.25rem;   /* 20px - h-5, w-5 */
  --icon-md: 1.5rem;    /* 24px - h-6, w-6 */
  --icon-lg: 2rem;      /* 32px - h-8, w-8 */
  --icon-xl: 3rem;      /* 48px - h-12, w-12 */
}
```

**Action Required:**
1. Create semantic icon size tokens
2. Document icon sizing guidelines
3. Standardize button height tokens

---

### 5. Border Radius (MEDIUM)

**Status:** 13 variants, 447 total uses

#### Current Border Radius Usage

| Token | Uses | Files | Status |
|-------|------|-------|--------|
| `rounded-lg` | 260 | 59 | ✅ Already in globals.css (--radius-lg) |
| `rounded-full` | 89 | 32 | ✅ Standard |
| `rounded` | 49 | 19 | ⚠️ Should use explicit size |
| `rounded-md` | 23 | 14 | ✅ Already in globals.css (--radius-md) |
| `rounded-xl` | 14 | 8 | ✅ Already in globals.css (--radius-xl) |
| `rounded-xs` | 3 | 3 | ⚠️ Not in globals.css |

**Current globals.css has:**
```css
--radius-sm: calc(var(--radius) - 4px);   /* ~6px */
--radius-md: calc(var(--radius) - 2px);   /* ~8px */
--radius-lg: var(--radius);               /* 10px */
--radius-xl: calc(var(--radius) + 4px);   /* ~14px */
```

**Action Required:**
1. Replace all `rounded` with explicit `rounded-lg` or `rounded-md`
2. Add `--radius-xs` for the 3 uses of `rounded-xs`
3. Document when to use each radius size

---

### 6. Shadow Tokens (MEDIUM)

**Status:** 8 variants, 35 total uses

#### Current Shadow Usage

| Token | Uses | Files |
|-------|------|-------|
| `shadow-xs` | 9 | 9 |
| `shadow` | 8 | 8 |
| `shadow-lg` | 5 | 4 |
| `shadow-sm` | 4 | 3 |
| `shadow-2xl` | 3 | 3 |
| `shadow-none` | 3 | 2 |
| `shadow-md` | 2 | 2 |
| `shadow-xl` | 1 | 1 |

**Recommended shadow scale (add to globals.css):**
```css
:root {
  /* Elevation shadows */
  --shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
  --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
}
```

**Action Required:**
1. Document shadow scale in globals.css
2. Create elevation guidelines (when to use each shadow level)

---

### 7. Font Weights (LOW)

**Status:** 4 variants, 488 total uses

#### Current Font Weight Usage

| Token | Uses | Files | Standard? |
|-------|------|-------|-----------|
| `font-bold` | 240 | 63 | ✅ Common |
| `font-medium` | 142 | 45 | ✅ Common |
| `font-semibold` | 102 | 35 | ✅ Common |
| `font-normal` | 4 | 3 | ✅ Rare but valid |

**Status:** ✅ Well-standardized, no action required

---

## Work Breakdown by Priority

### Phase 1: CRITICAL - Color Token Standardization (24-36 hours)

**Scope:** 76 variants, 3,373 uses, 103+ files

**Tasks:**
1. **Add missing tokens to globals.css** (2 hours)
   - Terminal window colors (`--terminal-red`, `--terminal-amber`, `--terminal-green`)
   - Error state colors (`--error-light`, `--error-dark`)
   - Missing chart colors

2. **Replace hard-coded colors** (8-12 hours)
   - Find all `bg-red-500`, `bg-amber-500`, `bg-green-500`
   - Replace with semantic tokens
   - Update TerminalWindow component

3. **Audit color usage** (6-8 hours)
   - Review all 76 color variants
   - Identify inconsistencies
   - Document color usage guidelines

4. **Update components** (8-14 hours)
   - Update 103+ files with standardized colors
   - Test visual consistency
   - Fix any regressions

**Deliverables:**
- Updated `globals.css` with all color tokens
- Color usage guidelines document
- 103+ updated component files

---

### Phase 2: HIGH - Spacing Token Standardization (16-24 hours)

**Scope:** 89 variants, 1,774 uses, 148 files

**Tasks:**
1. **Document spacing scale in globals.css** (2 hours)
   - Add CSS custom properties for spacing
   - Document the scale (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32)

2. **Normalize outliers** (4-6 hours)
   - Replace `gap-7` with `gap-6` or `gap-8`
   - Standardize `py-20` usage
   - Fix inconsistent spacing

3. **Create spacing guidelines** (2-3 hours)
   - Component padding standards
   - Section spacing standards
   - Grid/flex gap standards

4. **Update components** (8-13 hours)
   - Update 148 files with standardized spacing
   - Test layout consistency
   - Fix any regressions

**Deliverables:**
- Spacing scale in `globals.css`
- Spacing guidelines document
- 148 updated component files

---

### Phase 3: HIGH - Text Sizing Standardization (8-12 hours)

**Scope:** 11 variants, 760 uses, 83+ files

**Tasks:**
1. **Document typography scale in globals.css** (1 hour)
   - Add CSS custom properties for text sizes

2. **Create typography guidelines** (2-3 hours)
   - Heading size standards (h1-h6)
   - Body text standards
   - UI text standards (buttons, labels, etc.)

3. **Update components** (5-8 hours)
   - Update 83+ files with standardized text sizes
   - Test typography consistency
   - Fix any regressions

**Deliverables:**
- Typography scale in `globals.css`
- Typography guidelines document
- 83+ updated component files

---

### Phase 4: MEDIUM - Sizing, Radius, Shadow Standardization (12-18 hours)

**Scope:** 53 variants, 1,263 uses, 148 files

**Tasks:**
1. **Icon sizing tokens** (3-4 hours)
   - Add icon size tokens to globals.css
   - Update icon usage across components

2. **Border radius cleanup** (3-4 hours)
   - Replace `rounded` with explicit sizes
   - Add `--radius-xs` token
   - Update components

3. **Shadow scale documentation** (2-3 hours)
   - Document shadow scale in globals.css
   - Create elevation guidelines

4. **Update components** (4-7 hours)
   - Update 148 files with standardized tokens
   - Test visual consistency

**Deliverables:**
- Icon, radius, shadow tokens in `globals.css`
- Updated guidelines
- 148 updated component files

---

### Phase 5: Documentation & Testing (12-20 hours)

**Tasks:**
1. **Create comprehensive token documentation** (4-6 hours)
   - Token reference guide
   - Usage guidelines
   - Examples for each token category

2. **Visual regression testing** (4-8 hours)
   - Test all components in Storybook
   - Compare before/after screenshots
   - Fix any visual regressions

3. **Create token migration guide** (2-3 hours)
   - How to use new tokens
   - Migration checklist
   - Common patterns

4. **Code review and cleanup** (2-3 hours)
   - Review all changes
   - Remove unused tokens
   - Optimize CSS

**Deliverables:**
- Token reference documentation
- Visual regression test results
- Migration guide
- Clean, optimized codebase

---

## Implementation Strategy

### Recommended Approach

**Option A: Incremental (Recommended)**
- Work in phases (1-5)
- One category at a time
- Test after each phase
- Lower risk, easier to manage

**Option B: Big Bang**
- All changes at once
- Single large PR
- Comprehensive testing at end
- Higher risk, faster completion

**Recommendation:** Use **Option A (Incremental)** to minimize risk and allow for iterative testing.

---

## Risk Assessment

### High Risk Areas

1. **Color changes** - Most visible, affects brand consistency
2. **Spacing changes** - Can break layouts
3. **Text sizing** - Affects readability and hierarchy

### Mitigation Strategies

1. **Visual regression testing** - Screenshot comparison before/after
2. **Incremental rollout** - One category at a time
3. **Thorough code review** - Multiple reviewers
4. **Staging environment testing** - Test in production-like environment
5. **Rollback plan** - Git branches for easy revert

---

## Success Metrics

### Quantitative Metrics

- **Token reduction:** 236 variants → ~50 standardized tokens (80% reduction)
- **Consistency score:** Measure % of components using standard tokens
- **Maintenance time:** Reduce time to update global styles

### Qualitative Metrics

- **Visual consistency:** All components follow design system
- **Developer experience:** Easier to build new components
- **Documentation quality:** Clear guidelines for token usage

---

## Timeline

### Estimated Schedule (2-3 weeks)

**Week 1:**
- Phase 1: Color standardization (3-4 days)
- Phase 2: Spacing standardization (2-3 days)

**Week 2:**
- Phase 3: Text sizing standardization (1-2 days)
- Phase 4: Sizing/radius/shadow standardization (2-3 days)

**Week 3:**
- Phase 5: Documentation & testing (2-3 days)
- Buffer for unexpected issues (1-2 days)

---

## Next Steps

1. **Review this plan** - Get stakeholder approval
2. **Set up tracking** - Create GitHub issues for each phase
3. **Create feature branch** - `feat/css-token-standardization`
4. **Start Phase 1** - Color token standardization
5. **Regular check-ins** - Daily progress updates

---

## Appendix

### Tools Used

- **cloc** - Code line counting
- **tree** - Directory structure visualization
- **Python analysis script** - Custom CSS token extraction
- **grep/ripgrep** - Pattern matching

### Generated Reports

1. `CSS_TOKEN_ANALYSIS.md` - Detailed token usage analysis
2. `CSS_STANDARDIZATION_WORK_PLAN.md` - High-level work breakdown
3. `CSS_STANDARDIZATION_MASTER_PLAN.md` - This document

### References

- Current `globals.css` - `/frontend/bin/commercial/styles/globals.css`
- Component directory - `/frontend/bin/commercial/components`
- Tailwind config - `/frontend/bin/commercial/tailwind.config.ts`

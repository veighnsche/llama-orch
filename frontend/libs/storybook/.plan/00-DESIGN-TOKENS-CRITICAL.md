# 🚨 CRITICAL: Design Tokens Required

**Updated by:** TEAM-FE-003  
**Date:** 2025-10-11  
**Priority:** CRITICAL - READ BEFORE IMPLEMENTING ANY COMPONENT

---

## ⚠️ DO NOT COPY COLORS FROM REACT REFERENCE

The React reference uses **hardcoded Tailwind colors** (slate-950, amber-500, etc.).

**WE DO NOT USE THESE!**

---

## ✅ What We Use Instead

**Tailwind v4 Design Tokens** via `@theme` directive in `tokens.css`

### Updated tokens.css

**Location:** `/frontend/libs/storybook/styles/tokens.css`

**Key changes:**
- ✅ Uses `@theme inline` directive (Tailwind v4 pattern)
- ✅ Maps CSS variables to Tailwind utilities
- ✅ Automatic dark mode support
- ✅ Consistent with modern Tailwind v4 best practices

---

## 📋 Translation Guide

When porting from React reference, **translate** hardcoded colors:

### Common Translations

| React Reference | Our Design Token | Usage |
|----------------|------------------|-------|
| `bg-white` | `bg-background` | Page/card backgrounds |
| `bg-slate-50` | `bg-secondary` | Subtle backgrounds |
| `bg-slate-900` | `bg-background` | Dark backgrounds |
| `text-slate-900` | `text-foreground` | Primary text |
| `text-slate-600` | `text-muted-foreground` | Secondary text |
| `bg-amber-500` | `bg-primary` or `bg-accent` | Primary actions |
| `text-amber-500` | `text-primary` | Accent text |
| `border-slate-200` | `border-border` | Borders |
| `bg-red-500` | `bg-destructive` | Errors |

---

## 🎨 Available Design Tokens

### Colors
- `bg-background` / `text-foreground` - Base
- `bg-card` / `text-card-foreground` - Cards
- `bg-primary` / `text-primary-foreground` - Primary
- `bg-secondary` / `text-secondary-foreground` - Secondary
- `bg-muted` / `text-muted-foreground` - Muted
- `bg-accent` / `text-accent-foreground` - Accents
- `bg-destructive` / `text-destructive-foreground` - Errors
- `border-border` - Borders
- `ring-ring` - Focus rings

### Border Radius
- `rounded-sm` / `rounded` / `rounded-md` / `rounded-lg` / `rounded-xl`

---

## 🔄 Porting Workflow

### ❌ WRONG Approach
```vue
<!-- DON'T DO THIS -->
<template>
  <div class="bg-slate-950 text-white">
    <h1 class="text-amber-500">Title</h1>
    <p class="text-slate-300">Description</p>
  </div>
</template>
```

### ✅ CORRECT Approach
```vue
<!-- DO THIS -->
<template>
  <div class="bg-background text-foreground">
    <h1 class="text-primary">Title</h1>
    <p class="text-muted-foreground">Description</p>
  </div>
</template>
```

---

## 📝 Step-by-Step Process

When implementing ANY component:

1. **Read React reference** - Understand the design intent
2. **Identify color purposes:**
   - Is this a primary action? → `bg-primary`
   - Is this muted text? → `text-muted-foreground`
   - Is this a card? → `bg-card`
3. **Use design token** - Not hardcoded color
4. **Test in dark mode** - Ensure it looks good

---

## 🎯 Why This Matters

### Benefits
1. ✅ **Dark mode works automatically**
2. ✅ **Consistent branding** across all components
3. ✅ **Easy to rebrand** - change tokens, not 100 files
4. ✅ **Maintainable** - no magic colors scattered everywhere

### Problems with Hardcoded Colors
1. ❌ Dark mode breaks
2. ❌ Inconsistent colors across components
3. ❌ Impossible to rebrand
4. ❌ Maintenance nightmare

---

## 🚨 Updated Engineering Rules

**Section added:** "CRITICAL: USE DESIGN TOKENS, NOT HARDCODED COLORS"

**Location:** `/frontend/FRONTEND_ENGINEERING_RULES.md` (Section 2)

**Includes:**
- ✅ Required pattern with examples
- ✅ Banned pattern with examples
- ✅ Complete translation guide
- ✅ Available tokens list
- ✅ Porting workflow

---

## 📋 Checklist for ALL Components

Before marking any component complete:

- [ ] Read React reference
- [ ] Identified color purposes (not just copied classes)
- [ ] Used design tokens (not hardcoded colors)
- [ ] Tested in light mode
- [ ] Tested in dark mode
- [ ] No `slate-*`, `amber-*`, etc. in template
- [ ] Only semantic tokens (`bg-primary`, `text-foreground`, etc.)

---

## 🎓 Examples

### Example 1: Hero Section

**React Reference:**
```tsx
<section className="bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800">
  <h1 className="text-white">
    <span className="text-amber-400">Highlight</span>
  </h1>
  <p className="text-slate-300">Description</p>
</section>
```

**Our Vue Version:**
```vue
<template>
  <section class="bg-gradient-to-br from-background via-secondary to-muted">
    <h1 class="text-foreground">
      <span class="text-primary">Highlight</span>
    </h1>
    <p class="text-muted-foreground">Description</p>
  </section>
</template>
```

### Example 2: Card Component

**React Reference:**
```tsx
<div className="bg-white border border-slate-200 rounded-lg p-6">
  <h3 className="text-slate-900">Title</h3>
  <p className="text-slate-600">Content</p>
</div>
```

**Our Vue Version:**
```vue
<template>
  <div class="bg-card border-border rounded-lg p-6">
    <h3 class="text-card-foreground">Title</h3>
    <p class="text-muted-foreground">Content</p>
  </div>
</template>
```

---

## 🔗 References

- **Updated tokens.css:** `/frontend/libs/storybook/styles/tokens.css`
- **Engineering rules:** `/frontend/FRONTEND_ENGINEERING_RULES.md` (Section 2)
- **Tailwind v4 docs:** https://tailwindcss.com/docs/v4-beta

---

**READ THIS BEFORE IMPLEMENTING ANY COMPONENT!**

**If you copy hardcoded colors from React reference, your work will be rejected.** ❌

---

## 📚 Required Reading

**BEFORE starting this unit, read:**

1. **Design Tokens (CRITICAL):** `00-DESIGN-TOKENS-CRITICAL.md`
   - DO NOT copy colors from React reference
   - Use design tokens: `bg-primary`, `text-foreground`, etc.
   - Translation guide: React colors → Vue tokens

2. **Engineering Rules:** `/frontend/FRONTEND_ENGINEERING_RULES.md`
   - Section 2: Design tokens requirement
   - Section 3: Histoire `.story.vue` format
   - Section 8: Port vs create distinction

3. **Examples:** Look at completed components
   - HeroSection: `/frontend/libs/storybook/stories/organisms/HeroSection/`
   - WhatIsRbee: `/frontend/libs/storybook/stories/organisms/WhatIsRbee/`
   - ProblemSection: `/frontend/libs/storybook/stories/organisms/ProblemSection/`

**Key Rules:**
- ✅ Use `.story.vue` format (NOT `.story.ts`)
- ✅ Use design tokens (NOT hardcoded colors like `bg-amber-500`)
- ✅ Import from workspace: `import { Button } from 'rbee-storybook/stories'`
- ✅ Add team signature: `<!-- TEAM-FE-XXX: Implemented ComponentName -->`
- ✅ Export in `stories/index.ts`

---


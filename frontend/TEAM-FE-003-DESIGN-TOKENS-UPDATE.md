# ✅ TEAM-FE-003: Critical Design Tokens Update

**Team:** TEAM-FE-003  
**Date:** 2025-10-11  
**Status:** COMPLETE ✅  
**Priority:** CRITICAL

---

## 🎯 Problem Identified

The React reference uses **hardcoded Tailwind colors** (slate-950, amber-500, etc.) which:
- ❌ Don't support dark mode
- ❌ Create maintenance nightmares
- ❌ Make rebranding impossible
- ❌ Scatter magic colors throughout codebase

Our existing `tokens.css` was **outdated** and didn't use Tailwind v4's modern `@theme` pattern.

---

## ✅ Solution Implemented

### 1. Updated tokens.css to Tailwind v4 Pattern

**File:** `/frontend/libs/storybook/styles/tokens.css`

**Changes:**
- ✅ Added `@custom-variant dark` for dark mode
- ✅ Defined CSS variables in `:root` and `.dark`
- ✅ Added `@theme inline` directive (Tailwind v4 pattern)
- ✅ Mapped CSS vars to Tailwind utilities
- ✅ Enabled semantic color tokens

**Result:** Can now use `bg-primary`, `text-foreground`, etc. instead of hardcoded colors.

---

### 2. Added Critical Engineering Rule

**File:** `/frontend/FRONTEND_ENGINEERING_RULES.md`

**Section:** "CRITICAL: USE DESIGN TOKENS, NOT HARDCODED COLORS"

**Includes:**
- ✅ Required pattern with examples
- ✅ Banned pattern with examples
- ✅ Complete translation guide (React → Vue)
- ✅ Available tokens list
- ✅ Porting workflow
- ✅ Why it matters explanation

---

### 3. Created Critical Documentation

**File:** `/frontend/.plan/00-DESIGN-TOKENS-CRITICAL.md`

**Purpose:** Ensure ALL future teams read this BEFORE implementing components

**Includes:**
- ⚠️ Warning: DO NOT copy colors from React
- ✅ Translation guide
- ✅ Available tokens
- ✅ Porting workflow
- ✅ Examples (before/after)
- ✅ Checklist

---

### 4. Updated Work Plan

**Files Updated:**
- `/frontend/.plan/00-MASTER-PLAN.md` - Added critical warning at top
- `/frontend/.plan/README.md` - Added critical warning at top

**Result:** Impossible to miss this requirement.

---

## 📋 Translation Guide

### Common Translations

| React Reference | Our Design Token | Purpose |
|----------------|------------------|---------|
| `bg-white` | `bg-background` | Page backgrounds |
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

### Colors (via @theme)
- `bg-background` / `text-foreground` - Base colors
- `bg-card` / `text-card-foreground` - Card backgrounds
- `bg-primary` / `text-primary-foreground` - Primary actions
- `bg-secondary` / `text-secondary-foreground` - Secondary elements
- `bg-muted` / `text-muted-foreground` - Muted/subtle elements
- `bg-accent` / `text-accent-foreground` - Accent highlights
- `bg-destructive` / `text-destructive-foreground` - Errors/warnings
- `border-border` - Border colors
- `ring-ring` - Focus rings

### Border Radius
- `rounded-sm` / `rounded` / `rounded-md` / `rounded-lg` / `rounded-xl`

---

## 🔄 Porting Workflow

### Step-by-Step

1. **Read React reference** - Understand design intent
2. **Identify purposes:**
   - Primary action? → `bg-primary`
   - Muted text? → `text-muted-foreground`
   - Card background? → `bg-card`
3. **Use design token** - NOT hardcoded color
4. **Test in dark mode** - Ensure it looks good

### Example

**React Reference:**
```tsx
<div className="bg-slate-950 text-white">
  <h1 className="text-amber-500">Title</h1>
  <p className="text-slate-300">Description</p>
</div>
```

**Our Vue Version:**
```vue
<template>
  <div class="bg-background text-foreground">
    <h1 class="text-primary">Title</h1>
    <p class="text-muted-foreground">Description</p>
  </div>
</template>
```

---

## 🎯 Impact

### Benefits
1. ✅ **Dark mode works automatically** - Tokens adapt
2. ✅ **Consistent branding** - One source of truth
3. ✅ **Easy to rebrand** - Change tokens, not 100 files
4. ✅ **Maintainable** - No magic colors
5. ✅ **Modern** - Follows Tailwind v4 best practices

### Prevents
1. ❌ Dark mode breaking
2. ❌ Inconsistent colors
3. ❌ Rebranding nightmares
4. ❌ Maintenance hell

---

## 📊 Files Modified

1. `/frontend/libs/storybook/styles/tokens.css` - Updated to Tailwind v4 @theme pattern
2. `/frontend/FRONTEND_ENGINEERING_RULES.md` - Added critical design tokens rule
3. `/frontend/.plan/00-DESIGN-TOKENS-CRITICAL.md` - Created critical documentation
4. `/frontend/.plan/00-MASTER-PLAN.md` - Added critical warning
5. `/frontend/.plan/README.md` - Added critical warning
6. `/frontend/TEAM-FE-003-DESIGN-TOKENS-UPDATE.md` - This file

---

## ✅ Verification Checklist

- [x] tokens.css updated to Tailwind v4 pattern
- [x] @theme directive added
- [x] CSS variables defined for light/dark modes
- [x] Engineering rules updated with critical section
- [x] Translation guide created
- [x] Critical documentation created
- [x] Work plan updated with warnings
- [x] Examples provided
- [x] Checklist created for future teams

---

## 🚀 Next Steps for Future Teams

**BEFORE implementing ANY component:**

1. Read `/frontend/.plan/00-DESIGN-TOKENS-CRITICAL.md`
2. Read engineering rules section on design tokens
3. Understand the translation guide
4. Use design tokens, NOT hardcoded colors
5. Test in both light and dark modes

**If you copy colors from React reference, your work will be rejected!**

---

## 💡 Key Takeaway

**React reference is a VISUAL reference, not a CODE reference.**

- ✅ Copy the DESIGN INTENT
- ❌ DON'T copy the hardcoded colors

**Use semantic design tokens for everything.**

---

**This update prevents future teams from creating a maintenance nightmare!** 🎉

```
// Created by: TEAM-FE-003
// Date: 2025-10-11
// Purpose: Critical design tokens update
// Status: Complete ✅
// Impact: Prevents hardcoded colors, enables dark mode
```

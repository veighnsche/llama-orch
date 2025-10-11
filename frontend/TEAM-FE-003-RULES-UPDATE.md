# ✅ TEAM-FE-003: Frontend Engineering Rules Updated

**Created by:** TEAM-FE-003  
**Date:** 2025-10-11  
**Status:** Complete ✅

---

## 🎯 Mission

Update `FRONTEND_ENGINEERING_RULES.md` with critical missing rules discovered from TEAM-FE-002's lessons learned.

---

## ✅ What Was Added

### 1. **Histoire Story Format Rule** ⭐ CRITICAL

**Location:** Section 3 - Story Requirements

**Added:**
- Explicit requirement for `.story.vue` format (NOT `.story.ts`)
- Complete example of correct Vue SFC story format
- Example of BANNED TypeScript story format
- Clear explanation of why this matters

**Why:** TEAM-FE-002 spent hours debugging because they used `.story.ts` format which doesn't work with Histoire.

---

### 2. **Tailwind CSS v4 Histoire Setup** ⭐ CRITICAL

**Location:** Section 0 - After Tailwind CSS section

**Added:**
- Required configuration for Tailwind v4 in Histoire
- Three-step setup guide:
  1. `@import "tailwindcss"` in `styles/tokens.css`
  2. PostCSS plugin in `histoire.config.ts`
  3. Import tokens.css in `histoire.setup.ts`
- Symptoms of misconfiguration
- Troubleshooting guidance

**Why:** TEAM-FE-002 had components with no styling because Tailwind v4 wasn't configured properly.

---

### 3. **Workspace Package Imports** ⭐ CRITICAL

**Location:** Section 1 - After Atomic Design Hierarchy

**Added:**
- Explicit requirement to use workspace packages
- Correct import examples
- Banned relative import examples
- Explanation of why this matters

**Why:** Ensures consistent imports and prevents circular dependencies.

---

### 4. **Export All Components Rule** ⭐ CRITICAL

**Location:** Section 1 - After Workspace Imports

**Added:**
- Requirement to export all components in `stories/index.ts`
- Complete export example
- Workflow checklist
- Explanation of why this matters

**Why:** Components can't be used if they're not exported.

---

### 5. **Two Types of Components** ⭐ CRITICAL

**Location:** Section 8 - Port Workflow Rules

**Added:**
- Clear distinction between:
  - **Type 1: UI Primitives (Atoms)** - Port from React `/components/ui/`
  - **Type 2: Page Components (Molecules/Organisms)** - Create new by analyzing pages
- Separate workflows for each type
- Examples and locations for each
- Commands to find and read React components

**Why:** TEAM-FE-002 needed clarification on when to port vs when to create.

---

### 6. **Testing Commands Quick Reference**

**Location:** Section 15 - Resources

**Added:**
- Exact commands to test in Histoire
- Exact commands to test Vue app
- Exact commands to compare with React reference
- Linting commands

**Why:** Teams need copy-pastable commands.

---

### 7. **Critical Lessons from Failed Teams** (NEW SECTION)

**Location:** Section 17 (NEW)

**Added:**
- TEAM-FE-002 Lesson: Histoire Story Format
  - What happened
  - Root cause
  - Solution
- TEAM-FE-002 Lesson: Tailwind CSS v4 Setup
  - What happened
  - Root cause
  - Solution
- Key takeaways

**Why:** Learn from past mistakes to avoid repeating them.

---

### 8. **Updated Summary Checklist**

**Location:** End of document

**Added to checklist:**
- [ ] Story files use `.story.vue` format (NOT `.story.ts`)
- [ ] Tailwind v4 configured in Histoire (PostCSS + @import)
- [ ] Component exported in stories/index.ts

---

## 📊 Impact

### Before Updates:
- ❌ No guidance on Histoire story format
- ❌ No Tailwind v4 setup instructions
- ❌ Unclear distinction between porting vs creating
- ❌ No explicit export requirements
- ❌ No lessons learned section

### After Updates:
- ✅ Clear Histoire `.story.vue` requirement with examples
- ✅ Complete Tailwind v4 setup guide
- ✅ Clear distinction: Port atoms, create organisms
- ✅ Explicit export workflow
- ✅ Lessons learned from TEAM-FE-002
- ✅ Testing commands quick reference

---

## 🎯 Key Improvements

1. **Prevented future debugging issues** - Histoire story format now explicit
2. **Prevented styling issues** - Tailwind v4 setup now documented
3. **Clarified workflow** - Port vs create distinction clear
4. **Improved discoverability** - Export requirements explicit
5. **Learning from history** - Lessons learned section added

---

## 📝 Files Modified

1. `/frontend/FRONTEND_ENGINEERING_RULES.md`
   - Added 5 new critical sections
   - Enhanced existing sections
   - Added Section 17: Critical Lessons from Failed Teams
   - Updated version to 1.1
   - Updated checklist

---

## ✅ Verification

**Confirmed:**
- [ ] All TEAM-FE-002 issues addressed
- [ ] Histoire story format clearly documented
- [ ] Tailwind v4 setup clearly documented
- [ ] Export requirements clearly documented
- [ ] Port vs create distinction clearly documented
- [ ] Testing commands provided
- [ ] Lessons learned section added
- [ ] Version updated to 1.1
- [ ] Updated by TEAM-FE-003 noted

---

## 🚀 Next Steps for Future Teams

**Future teams should:**
1. Read Section 3 for Histoire story format requirements
2. Read Section 0 for Tailwind v4 setup
3. Read Section 1 for export requirements
4. Read Section 8 for port vs create distinction
5. Read Section 17 for lessons learned
6. Use the testing commands in Section 15

**These rules will prevent the issues TEAM-FE-002 faced.**

---

## 📋 Summary

**Added:** 5 critical sections + 1 lessons learned section  
**Enhanced:** Port workflow, story requirements, testing commands  
**Impact:** Future teams will avoid TEAM-FE-002's debugging issues  
**Status:** Rules updated and ready for TEAM-FE-004+

---

**TEAM-FE-003 has successfully updated the engineering rules based on real-world lessons learned!** 🚀

```
// Created by: TEAM-FE-003
// Date: 2025-10-11
// Purpose: Document rules updates based on TEAM-FE-002 lessons
// Status: Complete ✅
```

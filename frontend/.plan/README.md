# Frontend v0 Port - Work Plan

**Created by:** TEAM-FE-003  
**Date:** 2025-10-11  
**Purpose:** Structured plan for complete React → Vue port

---

## 🚨 CRITICAL: READ FIRST

**BEFORE implementing ANY component:**

1. **Read:** `00-DESIGN-TOKENS-CRITICAL.md`
2. **Key rule:** Use design tokens (`bg-primary`) NOT hardcoded colors (`bg-amber-500`)
3. **Why:** React reference has hardcoded colors. We use tokens for dark mode + consistency.

**If you copy colors from React reference, your work will be rejected!** ❌

---

## 📖 How to Use This Plan

### Each File = One Unit of Work

Every `.md` file in this directory represents **one discrete unit of work** that can be:
- Assigned to a team member
- Tracked independently
- Completed in 1-3 hours
- Verified with clear completion criteria

### File Naming Convention

- `00-XX` = Infrastructure & planning
- `01-XX` = Home Page components
- `02-XX` = Developers Page components
- `03-XX` = Enterprise Page components
- `04-XX` = GPU Providers Page components
- `05-XX` = Features Page components
- `06-XX` = Use Cases Page components
- `07-XX` = Page assembly & testing
- `08-XX` = Missing atoms (conditional)

### Status Indicators

- 🔴 **Not Started** - Unit not begun
- 🟡 **In Progress** - Currently being worked on
- 🟢 **Complete** - Unit finished and verified
- ⚠️ **Conditional** - Only needed if dependency missing

---

## 🎯 Quick Start

### For TEAM-FE-004 (Next Team)

1. **Read the master plan:**
   ```bash
   cat /home/vince/Projects/llama-orch/frontend/.plan/00-MASTER-PLAN.md
   ```

2. **Start with first incomplete unit:**
   ```bash
   cat /home/vince/Projects/llama-orch/frontend/.plan/01-04-AudienceSelector.md
   ```

3. **Follow the checklist in the unit file**

4. **Mark unit complete when done:**
   - Update status in unit file: `**Status:** 🟢 Complete`
   - Update master plan progress

5. **Move to next unit**

---

## 📊 Progress Tracking

### Current Status (as of 2025-10-11)

**Completed:** 3 units (5%)
- ✅ 00-INFRASTRUCTURE (Engineering rules)
- ✅ 01-01-HeroSection
- ✅ 01-02-WhatIsRbee
- ✅ 01-03-ProblemSection

**Remaining:** 58 units (95%)

### By Page

- **Home:** 3/14 organisms complete (21%)
- **Developers:** 0/10 organisms (0%)
- **Enterprise:** 0/11 organisms (0%)
- **GPU Providers:** 0/11 organisms (0%)
- **Features:** 0/9 organisms (0%)
- **Use Cases:** 0/3 organisms (0%)
- **Pages:** 0/6 assembled (0%)

---

## 🔄 Workflow

### Standard Unit Workflow

1. **Read unit file** - Understand requirements
2. **Check dependencies** - Verify atoms exist
3. **Read React reference** - Understand original component
4. **Implement Vue component** - Port to Vue
5. **Create story** - Add `.story.vue` file
6. **Test in Histoire** - Verify component works
7. **Update status** - Mark unit complete

### Page Assembly Workflow

1. **Verify all organisms complete** - Check dependencies
2. **Create page view** - Assemble organisms
3. **Add/verify route** - Ensure routing works
4. **Test in browser** - Full page testing
5. **Compare with React** - Visual parity check

---

## 📋 Unit File Structure

Each unit file contains:

```markdown
# Unit XX-XX: ComponentName

**Status:** 🔴 Not Started
**Estimated Time:** X hours
**React Reference:** /path/to/react/component.tsx
**Vue Location:** /path/to/vue/component.vue

## 🎯 Description
What this component does

## 📦 Dependencies
Required atoms/molecules

## ✅ Implementation Checklist
Step-by-step tasks

## 🧪 Testing Checklist
Verification steps

## ✅ Completion Criteria
Done when...
```

---

## 🎯 Team Assignments (Recommended)

- **TEAM-FE-004:** Home Page (units 01-04 through 01-14, 07-01)
- **TEAM-FE-005:** Developers Page (units 02-XX, 07-02)
- **TEAM-FE-006:** Enterprise Page (units 03-XX, 07-03)
- **TEAM-FE-007:** GPU Providers Page (units 04-XX, 07-04)
- **TEAM-FE-008:** Features Page (units 05-XX, 07-05)
- **TEAM-FE-009:** Use Cases Page (units 06-XX, 07-06, 07-07)

---

## 📝 Important Files

- `00-MASTER-PLAN.md` - Overall plan and progress
- `02-DEVELOPERS-PAGE.md` - Developers page overview
- `03-ENTERPRISE-PAGE.md` - Enterprise page overview
- `04-PROVIDERS-PAGE.md` - Providers page overview
- `05-FEATURES-PAGE.md` - Features page overview
- `06-USE-CASES-PAGE.md` - Use Cases page overview
- `07-07-Testing.md` - Final testing procedures
- `08-01-Tabs.md` - Conditional atom (check if needed)
- `08-02-Accordion.md` - Conditional atom (check if needed)

---

## 🚀 Getting Started

```bash
# 1. Navigate to plan directory
cd /home/vince/Projects/llama-orch/frontend/.plan

# 2. Read master plan
cat 00-MASTER-PLAN.md

# 3. Find next incomplete unit
ls 01-*.md | head -1

# 4. Read unit file
cat 01-04-AudienceSelector.md

# 5. Start implementing!
```

---

## 💡 Tips

1. **One unit at a time** - Don't jump between units
2. **Test frequently** - Run Histoire after each component
3. **Follow patterns** - Look at completed units for examples
4. **Check dependencies** - Verify atoms exist before starting
5. **Update status** - Mark units complete as you go
6. **Ask for help** - Reference engineering rules if stuck

---

## 📚 Resources

- **Engineering Rules:** `/frontend/FRONTEND_ENGINEERING_RULES.md`
- **React Reference:** `/frontend/reference/v0/`
- **Storybook:** `/frontend/libs/storybook/`
- **Vue App:** `/frontend/bin/commercial-frontend/`

---

**This plan makes the massive v0 port manageable through discrete, trackable units of work!** 🚀

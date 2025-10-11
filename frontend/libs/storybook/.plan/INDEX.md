# Work Plan Index - Quick Reference

**Last Updated:** 2025-10-11 by TEAM-FE-003

---

## 📊 Quick Status

**Total Units:** 61  
**Complete:** 3 (5%)  
**In Progress:** 0  
**Not Started:** 58 (95%)

---

## 🔍 Find Your Next Unit

### Home Page (Priority 1)
| Unit | Component | Status | Time | Blocker |
|------|-----------|--------|------|---------|
| 01-01 | HeroSection | 🟢 Complete | - | - |
| 01-02 | WhatIsRbee | 🟢 Complete | - | - |
| 01-03 | ProblemSection | 🟢 Complete | - | - |
| 01-04 | AudienceSelector | 🔴 Not Started | 2h | May need Tabs atom |
| 01-05 | SolutionSection | 🔴 Not Started | 2h | - |
| 01-06 | HowItWorksSection | 🔴 Not Started | 1.5h | - |
| 01-07 | FeaturesSection | 🔴 Not Started | 1.5h | - |
| 01-08 | UseCasesSection | 🔴 Not Started | 1.5h | - |
| 01-09 | ComparisonSection | 🔴 Not Started | 2h | - |
| 01-10 | PricingSection | 🔴 Not Started | 1h | - |
| 01-11 | SocialProofSection | 🔴 Not Started | 1.5h | - |
| 01-12 | TechnicalSection | 🔴 Not Started | 1.5h | - |
| 01-13 | FAQSection | 🔴 Not Started | 2h | May need Accordion |
| 01-14 | CTASection | 🔴 Not Started | 1h | - |
| 07-01 | HomeView (Assembly) | 🔴 Not Started | 1h | All above complete |

**Total Time:** ~20 hours remaining

---

## 📋 All Pages Overview

### Developers Page (Priority 2)
- **Units:** 02-01 through 02-10 (10 organisms)
- **Assembly:** 07-02-DevelopersView.md
- **Time:** 12-15 hours
- **Status:** 🔴 Not Started
- **Overview:** `02-DEVELOPERS-PAGE.md`

### Enterprise Page (Priority 3)
- **Units:** 03-01 through 03-11 (11 organisms)
- **Assembly:** 07-03-EnterpriseView.md
- **Time:** 13-16 hours
- **Status:** 🔴 Not Started
- **Overview:** `03-ENTERPRISE-PAGE.md`

### GPU Providers Page (Priority 4)
- **Units:** 04-01 through 04-11 (11 organisms)
- **Assembly:** 07-04-ProvidersView.md
- **Time:** 13-16 hours
- **Status:** 🔴 Not Started
- **Overview:** `04-PROVIDERS-PAGE.md`

### Features Page (Priority 5)
- **Units:** 05-01 through 05-09 (9 organisms)
- **Assembly:** 07-05-FeaturesView.md
- **Time:** 10-13 hours
- **Status:** 🔴 Not Started
- **Overview:** `05-FEATURES-PAGE.md`

### Use Cases Page (Priority 6)
- **Units:** 06-01 through 06-03 (3 organisms)
- **Assembly:** 07-06-UseCasesView.md
- **Time:** 4-5 hours
- **Status:** 🔴 Not Started
- **Overview:** `06-USE-CASES-PAGE.md`

---

## 🔧 Conditional Units (Check First)

| Unit | Atom | Needed For | Check Command |
|------|------|------------|---------------|
| 08-01 | Tabs | AudienceSelector | `ls stories/atoms/Tabs/` |
| 08-02 | Accordion | FAQSection | `ls stories/atoms/Accordion/` |

---

## 📁 File Structure

```
.plan/
├── README.md                    # How to use this plan
├── INDEX.md                     # This file - quick reference
├── 00-MASTER-PLAN.md           # Overall plan and progress
│
├── 01-04-AudienceSelector.md   # Home page units
├── 01-05-SolutionSection.md
├── ... (01-06 through 01-14)
│
├── 02-DEVELOPERS-PAGE.md       # Developers overview
├── 03-ENTERPRISE-PAGE.md       # Enterprise overview
├── 04-PROVIDERS-PAGE.md        # Providers overview
├── 05-FEATURES-PAGE.md         # Features overview
├── 06-USE-CASES-PAGE.md        # Use Cases overview
│
├── 07-01-HomeView.md           # Page assemblies
├── 07-02-DevelopersView.md
├── ... (07-03 through 07-06)
├── 07-07-Testing.md            # Final testing
│
├── 08-01-Tabs.md               # Conditional atoms
└── 08-02-Accordion.md
```

---

## 🚀 Quick Start Commands

### For TEAM-FE-004 (Next Team)

```bash
# Navigate to plan
cd /home/vince/Projects/llama-orch/frontend/.plan

# Read the guide
cat README.md

# Check master plan
cat 00-MASTER-PLAN.md

# Start first unit
cat 01-04-AudienceSelector.md

# Navigate to storybook
cd ../libs/storybook

# Start Histoire
pnpm story:dev
```

---

## 📊 Progress by Team (Recommended)

| Team | Units | Estimated Time | Status |
|------|-------|----------------|--------|
| TEAM-FE-003 | 01-01, 01-02, 01-03, Infrastructure | 8h | ✅ Complete |
| TEAM-FE-004 | 01-04 through 01-14, 07-01 | 20h | 🔴 Not Started |
| TEAM-FE-005 | 02-XX, 07-02 | 15h | 🔴 Not Started |
| TEAM-FE-006 | 03-XX, 07-03 | 16h | 🔴 Not Started |
| TEAM-FE-007 | 04-XX, 07-04 | 16h | 🔴 Not Started |
| TEAM-FE-008 | 05-XX, 07-05 | 13h | 🔴 Not Started |
| TEAM-FE-009 | 06-XX, 07-06, 07-07 | 8h | 🔴 Not Started |

**Total:** ~96 hours across 7 teams

---

## 🎯 Current Focus

**Next Unit:** `01-04-AudienceSelector.md`  
**Next Team:** TEAM-FE-004  
**Next Page:** Home Page (11 units remaining)

---

## 📝 Update This File

When completing units, update the status:
- 🔴 Not Started
- 🟡 In Progress
- 🟢 Complete

---

**Use this index for quick navigation and status tracking!**

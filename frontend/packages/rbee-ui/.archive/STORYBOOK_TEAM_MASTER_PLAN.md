# 🚀 STORYBOOK TEAM MASTER PLAN

**Version:** 2.0  
**Date:** 2025-10-15  
**Status:** READY FOR EXECUTION  
**Total Components:** 73 organisms + atoms/molecules  
**Total Teams:** 6  
**Estimated Timeline:** 3-4 weeks

---

## 📊 EXECUTIVE SUMMARY

This master plan distributes ALL storybook work across **6 specialized teams**. Each team has a focused mission, balanced workload, and clear deliverables.

### THE PROBLEM WE'RE SOLVING:

1. ❌ **Viewport stories are GARBAGE** - MobileView/TabletView stories that show nothing new
2. ❌ **Missing marketing docs** - Stories exist but don't document copy/messaging strategy
3. ❌ **Missing stories** - 50+ components have NO stories at all
4. ❌ **No cross-page analysis** - Same component used differently across pages, not documented

### THE SOLUTION:

**6 specialized teams** working in parallel or sequence:
1. **TEAM-001:** Clean up garbage viewport stories (MUST RUN FIRST)
2. **TEAM-002:** Home page core sections (high priority)
3. **TEAM-003:** Developers + Features pages
4. **TEAM-004:** Enterprise + Pricing pages
5. **TEAM-005:** Providers + Use Cases pages
6. **TEAM-006:** Atoms & molecules (foundational)

---

## 🎯 TEAM ASSIGNMENTS

### TEAM-001: CLEANUP VIEWPORT STORIES
**File:** `TEAM_001_CLEANUP_VIEWPORT_STORIES.md`  
**Mission:** Remove nonsensical viewport-only stories  
**Components:** 10 organisms (cleanup only)  
**Estimated Time:** 3-4 hours  
**Priority:** ⚠️ **P0 - MUST RUN FIRST**  
**Status:** 🔴 NOT STARTED

**What they do:**
- Delete all MobileView/TabletView stories that ONLY change viewport
- Keep any stories that change props or behavior
- Clean up documentation references
- Make space for real variants

**Deliverables:**
- 10 story files cleaned (20+ useless stories removed)
- Updated documentation
- Cleaner Storybook sidebar

**Why first:** Other teams need a clean slate. Don't build on top of garbage.

---

### TEAM-002: HOME PAGE CORE SECTIONS
**File:** `TEAM_002_HOME_PAGE_CORE.md`  
**Mission:** Create complete stories with marketing/copy docs for home page organisms  
**Components:** 12 organisms  
**Estimated Time:** 16-20 hours  
**Priority:** 🔥 P1 - HIGH  
**Status:** 🔴 NOT STARTED

**What they do:**
- Document ALL home page sections with marketing analysis
- Create stories showing exact home page copy
- Analyze messaging strategy, CTAs, conversion tactics
- Add cross-page comparisons where applicable

**Components:**
1. HeroSection (enhance)
2. WhatIsRbee (enhance)
3. HomeSolutionSection (create)
4. HowItWorksSection (create)
5. FeaturesSection (create)
6. UseCasesSection (create)
7. ComparisonSection (create)
8. PricingSection (enhance)
9. SocialProofSection (create)
10. TechnicalSection (create)
11. ProblemSection (enhance)
12. CTASection (enhance)

**Deliverables:**
- 12 complete story files with marketing docs
- Cross-page variant analysis
- Minimum 3 stories per component

---

### TEAM-003: DEVELOPERS + FEATURES PAGES
**File:** `TEAM_003_DEVELOPERS_FEATURES.md`  
**Mission:** Create stories for Developers and Features page organisms  
**Components:** 16 organisms  
**Estimated Time:** 20-24 hours  
**Priority:** 🟡 P2 - MEDIUM  
**Status:** 🔴 NOT STARTED

**What they do:**
- Document developer-focused messaging (more technical)
- Document feature deep dives (very technical)
- Compare variants to home page messaging
- Analyze technical depth and code example strategy

**Components:**

**Developers Page (7):**
1. DevelopersHero
2. DevelopersProblem
3. DevelopersSolution
4. DevelopersHowItWorks
5. DevelopersFeatures
6. DevelopersUseCases
7. DevelopersCodeExamples

**Features Page (9):**
8. FeaturesHero
9. CoreFeaturesTabs
10. CrossNodeOrchestration
11. IntelligentModelManagement
12. MultiBackendGpu
13. ErrorHandling
14. RealTimeProgress
15. SecurityIsolation
16. AdditionalFeaturesGrid

**Deliverables:**
- 16 complete story files
- Technical depth analysis
- Code example documentation
- Cross-page comparisons

---

### TEAM-004: ENTERPRISE + PRICING PAGES
**File:** `TEAM_004_ENTERPRISE_PRICING.md`  
**Mission:** Create stories for Enterprise and Pricing page organisms  
**Components:** 14 organisms  
**Estimated Time:** 18-22 hours  
**Priority:** 🟡 P2 - MEDIUM  
**Status:** 🔴 NOT STARTED

**What they do:**
- Document B2B/enterprise messaging (very different tone)
- Analyze compliance/security positioning
- Document pricing strategy and tier structure
- Buyer persona and conversion strategy analysis

**Components:**

**Enterprise Page (11):**
1. EnterpriseHero
2. EnterpriseProblem
3. EnterpriseSolution
4. EnterpriseCompliance
5. EnterpriseSecurity
6. EnterpriseHowItWorks
7. EnterpriseUseCases
8. EnterpriseComparison
9. EnterpriseFeatures
10. EnterpriseTestimonials
11. EnterpriseCTA

**Pricing Page (3):**
12. PricingHero
13. PricingComparison
14. Pricing FAQs (add variant to existing FaqSection)

**Deliverables:**
- 14 complete story files
- Enterprise buyer persona docs
- Pricing strategy analysis
- Compliance/security docs

---

### TEAM-005: PROVIDERS + USE CASES PAGES
**File:** `TEAM_005_PROVIDERS_USECASES.md`  
**Mission:** Create stories for GPU Providers and Use Cases page organisms  
**Components:** 13 organisms  
**Estimated Time:** 16-20 hours  
**Priority:** 🟡 P2 - MEDIUM  
**Status:** 🔴 NOT STARTED

**What they do:**
- Document two-sided marketplace dynamics (providers = different audience)
- Analyze earning potential and economics
- Document use case storytelling structure
- Persona-driven narrative analysis

**Components:**

**Providers Page (10):**
1. ProvidersHero
2. ProvidersProblem
3. ProvidersSolution
4. ProvidersHowItWorks
5. ProvidersFeatures
6. ProvidersUseCases
7. ProvidersEarnings
8. ProvidersMarketplace
9. ProvidersSecurity
10. ProvidersTestimonials
11. ProvidersCTA

**Use Cases Page (3):**
12. UseCasesHero
13. UseCasesPrimary
14. UseCasesIndustry

**Deliverables:**
- 13 complete story files
- Two-sided marketplace docs
- Earnings/economics analysis
- Use case storytelling docs

---

### TEAM-006: ATOMS & MOLECULES
**File:** `TEAM_006_ATOMS_MOLECULES.md`  
**Mission:** Review/enhance atoms, create stories for molecules  
**Components:** 8 components (2 atoms, 6 molecules)  
**Estimated Time:** 8-12 hours  
**Priority:** 🟢 P3 - LOW (Foundational)  
**Status:** 🔴 NOT STARTED

**What they do:**
- Review existing atom stories (GitHubIcon, DiscordIcon)
- Enhance molecule stories (BrandLogo, Card, etc.)
- Document composition (what atoms make up each molecule)
- Show usage context (how organisms use them)

**Components:**
1. GitHubIcon (review)
2. DiscordIcon (review)
3. BrandLogo (enhance)
4. BrandMark (review)
5. Card (enhance)
6. TestimonialsSection (investigate/create)
7. HomeSolutionSection (investigate - may be organism)
8. CodeExamplesSection (investigate - may be in TEAM-003)

**Deliverables:**
- 2 atoms reviewed/enhanced
- 6 molecules created/enhanced
- Composition documentation
- Usage context stories

---

## 📊 WORKLOAD DISTRIBUTION

| Team | Components | Est. Hours | Priority | Can Run In Parallel? |
|------|-----------|------------|----------|---------------------|
| TEAM-001 | 10 (cleanup) | 3-4 | P0 | ❌ NO - Must run first |
| TEAM-002 | 12 | 16-20 | P1 | ✅ YES (after Team 1) |
| TEAM-003 | 16 | 20-24 | P2 | ✅ YES (after Team 1) |
| TEAM-004 | 14 | 18-22 | P2 | ✅ YES (after Team 1) |
| TEAM-005 | 13 | 16-20 | P2 | ✅ YES (after Team 1) |
| TEAM-006 | 8 | 8-12 | P3 | ✅ YES (anytime) |
| **TOTAL** | **73** | **81-102** | - | - |

**Parallel execution possible:** Teams 2-6 can work simultaneously after Team 1 completes.

**Sequential execution:** If only one team, run in order: 1 → 2 → 3 → 4 → 5 → 6

---

## 🎯 SUCCESS CRITERIA

### Team Completion Criteria
Each team is complete when:
- ✅ All assigned components have stories
- ✅ Marketing documentation exists for each organism
- ✅ Minimum 3 stories per component (Default + 2 variants)
- ✅ NO viewport-only stories
- ✅ All props documented in argTypes
- ✅ Stories tested in Storybook (light + dark mode)
- ✅ All commits made with descriptive messages
- ✅ Team document updated with completion status

### Project Completion Criteria
The project is complete when:
- ✅ All 6 teams have completed their work
- ✅ All 73 components have complete stories
- ✅ Marketing strategy documented for all organisms
- ✅ Cross-page variant analysis complete
- ✅ No nonsensical viewport stories remain
- ✅ Storybook builds without errors
- ✅ Final QA pass complete

---

## 🚨 CRITICAL RULES (ALL TEAMS)

### ❌ BANNED PRACTICES

1. **NO viewport-only stories**
   - ❌ `export const MobileView: Story = { parameters: { viewport: { defaultViewport: 'mobile1' }}}`
   - ✅ Use Storybook's viewport toolbar instead

2. **NO placeholder documentation**
   - ❌ "This is a component"
   - ✅ "The HeroSection highlights the core value prop with headline, CTA, and social proof"

3. **NO Lorem ipsum or test data**
   - ❌ "Lorem ipsum dolor sit amet"
   - ✅ Real copy from the actual pages

4. **NO stories without variants**
   - ❌ Only Default story
   - ✅ Minimum 3 stories showing different uses/content/focus

5. **NO missing marketing docs**
   - ❌ Just the component description
   - ✅ Marketing strategy, audience, messaging, CTAs, conversion tactics

### ✅ REQUIRED PRACTICES

1. **Read the actual page first**
   - Check `frontend/apps/commercial/app/[page]/page.tsx`
   - Copy exact props, headlines, CTAs
   - Understand the context

2. **Document the strategy, not just the code**
   - WHY this headline?
   - WHO is the audience?
   - WHAT objections does it address?
   - HOW does it drive conversion?

3. **Show real variants**
   - Different headlines for A/B testing
   - Different focus areas (security vs. cost)
   - Different audiences (developer vs. enterprise)

4. **Commit per component**
   - Don't batch 5 components in one commit
   - Clear commit messages with context

5. **Test everything**
   - Light mode
   - Dark mode
   - No console errors
   - Props work in Controls panel

---

## 📅 TIMELINE OPTIONS

### Option A: Sequential (One Team)
**Total time:** 13-17 weeks (sequential execution)
- Week 1: Team 1 (cleanup)
- Weeks 2-3: Team 2 (home page)
- Weeks 4-6: Team 3 (developers/features)
- Weeks 7-9: Team 4 (enterprise/pricing)
- Weeks 10-12: Team 5 (providers/use cases)
- Weeks 13: Team 6 (atoms/molecules)

**Best for:** Single developer or small team

---

### Option B: Parallel (6 Teams)
**Total time:** 3-4 weeks (parallel after cleanup)
- Week 1, Day 1-2: Team 1 (cleanup) - BLOCKING
- Week 1, Day 3 onwards: Teams 2-6 start in parallel
- Week 2-3: All teams continue
- Week 4: QA and final touches

**Best for:** Large team or multiple AI agents

---

### Option C: Hybrid (2-3 Teams)
**Total time:** 5-7 weeks
- Week 1: Team 1 (cleanup)
- Weeks 2-3: Teams 2 + 6 (home page + atoms)
- Weeks 3-4: Teams 3 + 4 (developers/features + enterprise/pricing)
- Weeks 5-6: Team 5 (providers/use cases)
- Week 7: QA and final touches

**Best for:** Medium team

---

## 🚀 GETTING STARTED

### Step 1: Choose Your Team
- If you're TEAM-001, start immediately (cleanup)
- If you're TEAM-002 to TEAM-006, wait for TEAM-001 to finish

### Step 2: Read Your Team Document
- `TEAM_00X_[NAME].md` in this directory
- Read EVERY section
- Understand your components
- Check the examples

### Step 3: Read the Standards
- `STORYBOOK_DOCUMENTATION_STANDARD.md` - Quality requirements
- `STORYBOOK_QUICK_START.md` - Step-by-step guide
- `STORYBOOK_COMPONENT_DISCOVERY.md` - Component list

### Step 4: Start Storybook
```bash
cd /home/vince/Projects/llama-orch/frontend/packages/rbee-ui
pnpm storybook
# Opens at http://localhost:6006
```

### Step 5: Execute Your Mission
- Follow your team document step-by-step
- Check off components as you complete them
- Update progress tracker
- Commit frequently

### Step 6: Mark Complete
- Update your team document status to ✅ COMPLETE
- Update this master plan progress tracker
- Hand off any coordination notes to other teams

---

## 📊 MASTER PROGRESS TRACKER

### Teams Status
- [ ] TEAM-001: Cleanup Viewport Stories ✅ COMPLETE
- [ ] TEAM-002: Home Page Core ✅ COMPLETE
- [ ] TEAM-003: Developers + Features ✅ COMPLETE
- [ ] TEAM-004: Enterprise + Pricing ✅ COMPLETE
- [ ] TEAM-005: Providers + Use Cases ✅ COMPLETE
- [ ] TEAM-006: Atoms & Molecules ✅ COMPLETE

### Components Status
- **Total Components:** 73
- **Components Completed:** 0
- **Stories Created:** 0 new + 17 existing
- **Stories Cleaned:** 0
- **Progress:** 0%

### Time Status
- **Estimated Total:** 81-102 hours
- **Time Spent:** 0 hours
- **Estimated Remaining:** 81-102 hours

---

## 📞 COORDINATION

### Inter-Team Dependencies

**TEAM-001 blocks all others:**
- Must complete cleanup before other teams start
- Other teams build on cleaned-up files

**TEAM-002 and TEAM-003 may overlap:**
- HomeSolutionSection might be used in both
- Coordinate to avoid duplication

**TEAM-003 and TEAM-006 may overlap:**
- CodeExamplesSection might be in both scopes
- Check before duplicating work

**TEAM-006 supports all others:**
- Atoms/molecules are used by all organisms
- Can work in parallel with organism teams

### Communication Channels
- Update team documents with progress
- Note any blockers or questions
- Coordinate overlapping components
- Share learnings about marketing docs format

---

## 🎉 SUCCESS OUTCOMES

### What We'll Have
- ✅ **73 complete story files** with no garbage
- ✅ **Marketing strategy docs** for every organism
- ✅ **Cross-page variant analysis** showing how messaging changes
- ✅ **World-class Storybook** that's actually useful
- ✅ **Reusable patterns** for future components

### What We Won't Have
- ❌ Viewport-only stories cluttering the sidebar
- ❌ Lorem ipsum placeholder content
- ❌ Missing documentation
- ❌ Undocumented marketing strategy

### Impact
- **Engineering:** Faster development, clear API docs, reusable components
- **Marketing:** Documented copy strategy, A/B test variants, messaging consistency
- **Design:** Visual regression testing, brand consistency, design system
- **Sales:** Demo-ready components, use case examples, competitive positioning

---

## 📚 REFERENCE DOCUMENTS

All in `/home/vince/Projects/llama-orch/frontend/packages/rbee-ui/`:

### Team Documents
- `TEAM_001_CLEANUP_VIEWPORT_STORIES.md` - Cleanup mission
- `TEAM_002_HOME_PAGE_CORE.md` - Home page organisms
- `TEAM_003_DEVELOPERS_FEATURES.md` - Developers + Features pages
- `TEAM_004_ENTERPRISE_PRICING.md` - Enterprise + Pricing pages
- `TEAM_005_PROVIDERS_USECASES.md` - Providers + Use Cases pages
- `TEAM_006_ATOMS_MOLECULES.md` - Atoms & molecules

### Standards & Guides
- `STORYBOOK_DOCUMENTATION_STANDARD.md` - Quality requirements
- `STORYBOOK_QUICK_START.md` - Step-by-step guide
- `STORYBOOK_COMPONENT_DISCOVERY.md` - Component inventory
- `STORYBOOK_MASTER_INDEX.md` - All docs index
- `STORYBOOK_PROGRESS.md` - Current progress (update as you go)

### Source Pages
- `frontend/apps/commercial/app/page.tsx` - Home page
- `frontend/apps/commercial/app/developers/page.tsx` - Developers page
- `frontend/apps/commercial/app/enterprise/page.tsx` - Enterprise page
- `frontend/apps/commercial/app/features/page.tsx` - Features page
- `frontend/apps/commercial/app/pricing/page.tsx` - Pricing page
- `frontend/apps/commercial/app/gpu-providers/page.tsx` - Providers page
- `frontend/apps/commercial/app/use-cases/page.tsx` - Use Cases page

---

## 🚀 LET'S BUILD THIS!

**This is your roadmap. Follow it. Execute it. Ship it.**

**6 teams. 73 components. 3-4 weeks. Let's go! 🚀**

---

**Last Updated:** 2025-10-15  
**Status:** READY FOR EXECUTION  
**Next Update:** After TEAM-001 completes

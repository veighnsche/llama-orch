# 🎯 TEAM ASSIGNMENTS SUMMARY

**Quick reference for all 6 teams**

---

## ⚠️ THE PROBLEM

Looking at the screenshot you provided, I see **EXACTLY** what you mean:

```
FAQSection/
├── Default
├── Without Support Card
├── Custom Content
├── Mobile View          ← DELETE THIS SHIT
├── Tablet View          ← DELETE THIS SHIT
├── Interactive Search
├── Category Filtering
└── Support Card Highlight
```

**Mobile View** and **Tablet View** are NOT variants. They're the SAME story in a different viewport. Users can click the viewport button in Storybook to see mobile/tablet views. These stories are USELESS.

---

## 📋 TEAM BREAKDOWN

### TEAM-001: CLEANUP (3-4 hours) ⚠️ RUN FIRST
**File:** `TEAM_001_CLEANUP_VIEWPORT_STORIES.md`

**Mission:** Delete ALL viewport-only stories

**Components to clean (10):**
1. AudienceSelector - Remove MobileView, TabletView
2. CtaSection - Remove MobileView
3. EmailCapture - Remove MobileView, TabletView
4. FaqSection - Remove MobileView, TabletView
5. Footer - Remove MobileView, TabletView
6. HeroSection - Remove MobileView, TabletView
7. Navigation - Remove MobileView, TabletView
8. PricingSection - Remove MobileView, TabletView
9. ProblemSection - Remove MobileView, TabletView
10. WhatIsRbee - Remove MobileView, TabletView

**Expected result:** ~20 useless stories deleted, sidebar cleaner, no duplication.

**Why first:** Other teams need a clean slate.

---

### TEAM-002: HOME PAGE (16-20 hours) 🔥 HIGH PRIORITY
**File:** `TEAM_002_HOME_PAGE_CORE.md`

**Mission:** Document home page organisms with MARKETING/COPY analysis

**Components (12):**
1. HeroSection - Enhance with marketing docs
2. WhatIsRbee - Enhance with marketing docs
3. HomeSolutionSection - CREATE NEW story
4. HowItWorksSection - CREATE NEW story
5. FeaturesSection - CREATE NEW story
6. UseCasesSection - CREATE NEW story (4 personas documented)
7. ComparisonSection - CREATE NEW story
8. PricingSection - Enhance with home page context
9. SocialProofSection - CREATE NEW story
10. TechnicalSection - CREATE NEW story
11. ProblemSection - Enhance with marketing docs
12. CTASection - Enhance with home + developers context

**Key deliverable:** Marketing strategy docs for EVERY organism:
- Target audience
- Primary message
- Emotional appeal
- CTAs and conversion strategy
- Copy tone analysis

---

### TEAM-003: DEVELOPERS + FEATURES (20-24 hours)
**File:** `TEAM_003_DEVELOPERS_FEATURES.md`

**Mission:** Document Developers and Features pages (technical messaging)

**Developers Page (7):**
1. DevelopersHero
2. DevelopersProblem
3. DevelopersSolution
4. DevelopersHowItWorks
5. DevelopersFeatures
6. DevelopersUseCases
7. DevelopersCodeExamples - Document code as marketing tool

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

**Key deliverable:** Cross-page comparison showing how messaging differs from home page (more technical).

---

### TEAM-004: ENTERPRISE + PRICING (18-22 hours)
**File:** `TEAM_004_ENTERPRISE_PRICING.md`

**Mission:** Document Enterprise and Pricing pages (B2B messaging, pricing strategy)

**Enterprise Page (11):**
1. EnterpriseHero
2. EnterpriseProblem - Compliance/governance focus
3. EnterpriseSolution
4. EnterpriseCompliance - GDPR, SOC2, audit trails
5. EnterpriseSecurity - Zero-trust, isolation
6. EnterpriseHowItWorks
7. EnterpriseUseCases - Industry-specific
8. EnterpriseComparison - vs. Azure, AWS
9. EnterpriseFeatures - SSO, RBAC, SLAs
10. EnterpriseTestimonials - CTOs, IT directors
11. EnterpriseCTA - "Schedule Demo" not "Get Started"

**Pricing Page (3):**
12. PricingHero
13. PricingComparison - Document tier strategy
14. Pricing FAQs - Add variant to FaqSection

**Key deliverable:** Enterprise buyer persona docs, pricing strategy analysis (free/pro/enterprise tiers).

---

### TEAM-005: PROVIDERS + USE CASES (16-20 hours)
**File:** `TEAM_005_PROVIDERS_USECASES.md`

**Mission:** Document Providers and Use Cases pages (two-sided marketplace, storytelling)

**Providers Page (10):**
1. ProvidersHero - Target: GPU owners who want to EARN
2. ProvidersProblem - "Idle GPU = wasted money"
3. ProvidersSolution - Federated marketplace
4. ProvidersHowItWorks
5. ProvidersFeatures - Control, earnings tracking
6. ProvidersUseCases - Gamer, miner, homelab, datacenter personas
7. ProvidersEarnings - Calculator showing $ potential
8. ProvidersMarketplace - Discovery, pricing, reputation
9. ProvidersSecurity - Trust building
10. ProvidersTestimonials - "Earning $200/mo"
11. ProvidersCTA

**Use Cases Page (3):**
12. UseCasesHero
13. UseCasesPrimary
14. UseCasesIndustry

**Key deliverable:** Two-sided marketplace docs (provider vs. consumer), earnings analysis, use case storytelling structure.

---

### TEAM-006: ATOMS & MOLECULES (8-12 hours)
**File:** `TEAM_006_ATOMS_MOLECULES.md`

**Mission:** Review atoms, create/enhance molecule stories

**Components (8):**
1. GitHubIcon - REVIEW existing story
2. DiscordIcon - REVIEW existing story
3. BrandLogo - ENHANCE with composition docs
4. BrandMark - REVIEW
5. Card - ENHANCE with usage context
6. TestimonialsSection - Investigate (used in developers page)
7. HomeSolutionSection - Investigate (may be organism, coordinate with TEAM-002)
8. CodeExamplesSection - Investigate (may be in TEAM-003)

**Key deliverable:** Composition docs (what atoms make up each molecule), usage context stories.

---

## 🎯 QUICK RULES

### ❌ DELETE IF:
- Story ONLY changes `parameters.viewport`
- Shows SAME content, just smaller
- Provides ZERO value beyond viewport toolbar

### ✅ KEEP IF:
- Story changes props/content
- Shows different behavior (hamburger menu, etc.)
- Demonstrates actual variant

### 📝 EVERY ORGANISM NEEDS:
```markdown
## Marketing Strategy
- **Target Audience:** [specific persona]
- **Primary Message:** [core value prop]
- **Emotional Appeal:** [fear/freedom/frustration]
- **CTAs:** [what action, where, why]
- **Copy Tone:** [aggressive/empowering/technical]

## Usage in Commercial Site
[Exact props from actual page]

## Variants to Test
- Alternative headline 1
- Alternative CTA 1
```

---

## 📊 WORKLOAD SUMMARY

| Team | Components | Hours | Priority | Can Parallelize? |
|------|-----------|-------|----------|------------------|
| 001 | 10 cleanup | 3-4 | P0 | ❌ NO - Run first |
| 002 | 12 organisms | 16-20 | P1 | ✅ After Team 1 |
| 003 | 16 organisms | 20-24 | P2 | ✅ After Team 1 |
| 004 | 14 organisms | 18-22 | P2 | ✅ After Team 1 |
| 005 | 13 organisms | 16-20 | P2 | ✅ After Team 1 |
| 006 | 8 atoms/mols | 8-12 | P3 | ✅ Anytime |
| **TOTAL** | **73** | **81-102** | - | - |

---

## 🚀 EXECUTION OPTIONS

### Option A: One Team (Sequential)
**Timeline:** 13-17 weeks
- Week 1: Team 1
- Weeks 2-3: Team 2
- Weeks 4-6: Team 3
- Weeks 7-9: Team 4
- Weeks 10-12: Team 5
- Week 13: Team 6

### Option B: Six Teams (Parallel)
**Timeline:** 3-4 weeks
- Week 1 Day 1-2: Team 1 (BLOCKING)
- Week 1 Day 3+: Teams 2-6 start in parallel
- Weeks 2-3: All continue
- Week 4: QA

### Option C: Three Teams (Hybrid)
**Timeline:** 5-7 weeks
- Week 1: Team 1
- Weeks 2-3: Teams 2+6
- Weeks 3-4: Teams 3+4
- Weeks 5-6: Team 5
- Week 7: QA

---

## 📁 FILES CREATED

All in `/home/vince/Projects/llama-orch/frontend/packages/rbee-ui/`:

1. ✅ `TEAM_001_CLEANUP_VIEWPORT_STORIES.md` - Cleanup mission
2. ✅ `TEAM_002_HOME_PAGE_CORE.md` - Home page organisms
3. ✅ `TEAM_003_DEVELOPERS_FEATURES.md` - Developers + Features
4. ✅ `TEAM_004_ENTERPRISE_PRICING.md` - Enterprise + Pricing
5. ✅ `TEAM_005_PROVIDERS_USECASES.md` - Providers + Use Cases
6. ✅ `TEAM_006_ATOMS_MOLECULES.md` - Atoms & molecules
7. ✅ `STORYBOOK_TEAM_MASTER_PLAN.md` - Complete overview
8. ✅ `TEAM_ASSIGNMENTS_SUMMARY.md` - This file (quick reference)

---

## 🎯 YOUR NEXT STEPS

1. **Read** `STORYBOOK_TEAM_MASTER_PLAN.md` for complete context
2. **Choose** your team or execution strategy
3. **If running Team 1 first:** Read `TEAM_001_CLEANUP_VIEWPORT_STORIES.md` and START
4. **If running all teams:** Coordinate and distribute work
5. **Update progress** in each team document as you go

---

## 🔥 THE BOTTOM LINE

**Before:** 60 stories, 20 are viewport garbage, no marketing docs, missing 50+ components

**After:** 200+ stories, ZERO garbage, complete marketing docs for every organism, cross-page analysis

**Impact:** Engineers know how to use components, marketers can A/B test copy, sales has demo-ready examples.

---

**LET'S CLEAN THIS UP AND BUILD IT RIGHT! 🚀**

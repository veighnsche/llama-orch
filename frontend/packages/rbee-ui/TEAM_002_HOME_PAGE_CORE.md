# TEAM-002: HOME PAGE CORE SECTIONS

**Mission:** Create complete stories with marketing/copy documentation for home page organisms  
**Components:** 12 organisms  
**Estimated Time:** 16-20 hours  
**Priority:** P1 (High Priority)

---

## 🎯 MISSION BRIEFING

You're documenting the **core marketing sections** used on the home page. These are the hero, problem/solution, features, and conversion sections that drive the entire commercial site.

### CRITICAL REQUIREMENTS:

1. ✅ **Document ACTUAL copy from the home page** (`frontend/apps/commercial/app/page.tsx`)
2. ✅ **Show ALL variants** if the component is used multiple times
3. ✅ **Marketing documentation:** What promises does the copy make? What CTAs? What tone?
4. ✅ **NO viewport stories** (MobileView, TabletView) - use Storybook's viewport toolbar
5. ✅ **Minimum 2-3 REAL variants** per component

---

## 📋 YOUR COMPONENTS

### 1. HeroSection
**File:** `src/organisms/HeroSection/HeroSection.stories.tsx`  
**Status:** ✅ Story exists, needs marketing docs enhancement  
**Used in:** Home page  
**Current stories:** Default, WithScrollIndicator (viewport stories removed by Team 1)

**YOUR TASKS:**
- ✅ Add marketing documentation section to component description
- ✅ Document the copy strategy:
  - Headline: "Your hardware. Your AI. Your rules."
  - Value prop: What problem does it solve?
  - CTAs: "Get Started Free" + "View Documentation"
  - Tone: Direct, empowering, developer-focused
- ✅ Create story: `HomePageDefault` - exact copy from home page
- ✅ Create story: `AlternativeHeadlines` - show 2-3 alternative headline options
- ✅ Add props documentation for all customizable text

**Marketing Doc Template:**
```markdown
## Marketing Strategy
- **Target Audience:** Developers tired of cloud AI costs
- **Primary Message:** Own your infrastructure, control your costs
- **Emotional Hook:** Frustration with vendor lock-in
- **CTAs:** Free signup (primary), Docs (secondary)
- **Social Proof:** GitHub stars, active installations
```

---

### 2. WhatIsRbee
**File:** `src/organisms/WhatIsRbee/WhatIsRbee.stories.tsx`  
**Status:** ✅ Story exists, needs marketing docs (viewport stories removed by Team 1)  
**Used in:** Home page

**YOUR TASKS:**
- ✅ Add marketing documentation section
- ✅ Document the positioning:
  - "Self-hosted LLM orchestration for the rest of us"
  - Target: Developers who want control without complexity
  - Key differentiation: Simple, open-source, works with existing hardware
- ✅ Create story: `HomePageDefault` - exact copy
- ✅ Create story: `AlternativePositioning` - alternative explanation angles
- ✅ Document the tone: Technical but accessible

---

### 3. HomeSolutionSection
**File:** `src/organisms/SolutionSection/SolutionSection.stories.tsx` (create new)  
**Status:** ❌ NO STORY EXISTS  
**Used in:** Home page as `HomeSolutionSection`

**YOUR TASKS:**
- ✅ Create complete story file from scratch
- ✅ Document the home page variant:
  - Headline: "Your hardware. Your models. Your control."
  - 4 benefits: Zero costs, Complete privacy, Locked to your rules, Use all hardware
  - Topology diagram showing multi-host setup
- ✅ Create story: `HomePageDefault` - exact home page copy
- ✅ Create story: `WithoutTopology` - benefits only
- ✅ Create story: `AlternativeBenefits` - different benefit messaging
- ✅ Add marketing docs: What's the core value proposition?

**Component Location:** `src/organisms/SolutionSection/` (check actual name)

---

### 4. HowItWorksSection
**File:** `src/organisms/HowItWorksSection/HowItWorksSection.stories.tsx` (create new)  
**Status:** ❌ NO STORY EXISTS  
**Used in:** Home page

**YOUR TASKS:**
- ✅ Create complete story file
- ✅ Document the 3-step process from home page
- ✅ Create story: `HomePageDefault` - exact copy
- ✅ Create story: `AlternativeSteps` - different step descriptions
- ✅ Create story: `WithoutVisuals` - text-only steps
- ✅ Marketing docs: How does this reduce friction? What objections does it address?

---

### 5. FeaturesSection
**File:** `src/organisms/FeaturesSection/FeaturesSection.stories.tsx` (create new)  
**Status:** ❌ NO STORY EXISTS  
**Used in:** Home page

**YOUR TASKS:**
- ✅ Create complete story file
- ✅ Document all features shown on home page
- ✅ Create story: `HomePageDefault` - exact copy
- ✅ Create story: `CoreFeatures` - minimal feature set
- ✅ Create story: `AllFeatures` - expanded feature list
- ✅ Marketing docs: What features drive conversions? Which are "wow" moments?

---

### 6. UseCasesSection
**File:** `src/organisms/UseCasesSection/UseCasesSection.stories.tsx` (create new)  
**Status:** ❌ NO STORY EXISTS  
**Used in:** Home page (`page.tsx` lines 90-126)

**YOUR TASKS:**
- ✅ Create complete story file
- ✅ Document the 4 personas from home page:
  1. Solo developer: "$0/month AI costs"
  2. Small team: "$6,000+ saved per year"
  3. Homelab enthusiast: "Idle GPUs → productive"
  4. Enterprise: "EU-only compliance"
- ✅ Create story: `HomePageDefault` - exact copy with all 4 personas
- ✅ Create story: `SoloDeveloperOnly` - single persona deep dive
- ✅ Create story: `AlternativePersonas` - different audience segments
- ✅ Marketing docs: Which persona converts best? What pain points resonate?

**CRITICAL:** Each persona has icon, scenario, solution, outcome. Document ALL of them.

---

### 7. ComparisonSection
**File:** `src/organisms/ComparisonSection/ComparisonSection.stories.tsx` (create new)  
**Status:** ❌ NO STORY EXISTS  
**Used in:** Home page

**YOUR TASKS:**
- ✅ Create complete story file
- ✅ Document the comparison: rbee vs. Cloud APIs vs. DIY
- ✅ Create story: `HomePageDefault` - exact comparison
- ✅ Create story: `TwoWayComparison` - rbee vs. one competitor
- ✅ Create story: `AlternativeCompetitors` - different positioning
- ✅ Marketing docs: What's the competitive strategy? How aggressive is the tone?

---

### 8. PricingSection
**File:** `src/organisms/PricingSection/PricingSection.stories.tsx`  
**Status:** ✅ Story exists (viewport stories removed by Team 1)  
**Used in:** Home page, Pricing page, Developers page

**YOUR TASKS:**
- ✅ Enhance existing stories with marketing docs
- ✅ Document ALL 3 usage contexts:
  1. Home page: `variant="home"` - basic pricing overview
  2. Pricing page: `variant="pricing"` - detailed comparison
  3. Developers page: `variant="home"` without kicker/image
- ✅ Create story: `HomePageContext` - home page copy + context
- ✅ Create story: `DevelopersPageContext` - developers page variant
- ✅ Marketing docs: What's the pricing message? Free tier strategy?

---

### 9. SocialProofSection
**File:** `src/organisms/SocialProofSection/SocialProofSection.stories.tsx` (create new)  
**Status:** ❌ NO STORY EXISTS  
**Used in:** Home page

**YOUR TASKS:**
- ✅ Create complete story file
- ✅ Document social proof elements: GitHub stars, testimonials, company logos
- ✅ Create story: `HomePageDefault` - exact copy
- ✅ Create story: `WithoutLogos` - testimonials only
- ✅ Create story: `MetricsOnly` - just the numbers
- ✅ Marketing docs: What proof points matter most? Credibility strategy?

---

### 10. TechnicalSection
**File:** `src/organisms/TechnicalSection/TechnicalSection.stories.tsx` (create new)  
**Status:** ❌ NO STORY EXISTS  
**Used in:** Home page

**YOUR TASKS:**
- ✅ Create complete story file
- ✅ Document technical differentiators from home page
- ✅ Create story: `HomePageDefault` - exact copy
- ✅ Create story: `SimplifiedTech` - less technical language
- ✅ Create story: `DeepDive` - more technical details
- ✅ Marketing docs: Balance between technical depth and accessibility?

---

### 11. ProblemSection
**File:** `src/organisms/ProblemSection/ProblemSection.stories.tsx`  
**Status:** ✅ Story exists (viewport stories removed by Team 1)  
**Used in:** Home page (as ProblemSection)

**YOUR TASKS:**
- ✅ Enhance with marketing docs
- ✅ Document the home page problems:
  - "Unpredictable API costs"
  - "Vendor lock-in risk"
  - "Privacy & compliance concerns"
- ✅ Add story: `HomePageContext` - exact home page copy with surrounding sections
- ✅ Marketing docs: How does problem framing drive urgency? Tone analysis?

---

### 12. CTASection
**File:** `src/organisms/CtaSection/CtaSection.stories.tsx`  
**Status:** ✅ Story exists (viewport stories removed by Team 1)  
**Used in:** Home page, Developers page

**YOUR TASKS:**
- ✅ Enhance with marketing docs for ALL contexts
- ✅ Document home page CTA:
  - Title: "Stop depending on AI providers. Start building today."
  - Subtitle: "Join 500+ developers..."
  - Primary: "Get started free"
  - Secondary: "View documentation"
  - Emphasis: "gradient"
- ✅ Create story: `HomePageContext` - exact home page copy
- ✅ Create story: `DevelopersPageContext` - developers page variant
- ✅ Marketing docs: CTA strategy, urgency tactics, conversion optimization

---

## 🎯 STORY REQUIREMENTS (MANDATORY)

For EACH component, your story MUST include:

### 1. Marketing Documentation Section
```markdown
## Marketing Strategy

### Target Audience
[Who is this for? Be specific.]

### Primary Message
[Core value proposition in one sentence]

### Copy Analysis
- **Headline tone:** [Aggressive/Empowering/Educational/etc.]
- **Emotional appeal:** [Fear/Excitement/Frustration/Freedom]
- **Power words:** [List key persuasive terms used]
- **Social proof:** [What credibility signals are used?]

### Conversion Elements
- **Primary CTA:** [Text + Action + Placement]
- **Secondary CTA:** [Text + Action + Placement]
- **Objection handling:** [What concerns does copy address?]

### Variations to Test
- [Alternative headline 1]
- [Alternative CTA 1]
- [Alternative positioning 1]
```

### 2. Usage Documentation
```markdown
## Usage in Commercial Site

### Home Page (`/`)
\`\`\`tsx
<ComponentName
  title="Exact title from home page"
  subtitle="Exact subtitle from home page"
  // ... all props
/>
\`\`\`

**Context:** Appears after [previous section], before [next section]  
**Purpose:** [Why is it here? What's the goal?]  
**Metrics:** [If known: conversion rate, engagement, etc.]
```

### 3. Minimum Stories
- ✅ `HomePageDefault` - Exact copy from home page
- ✅ `AlternativeVariant1` - Different headline/copy/positioning
- ✅ `AlternativeVariant2` - Different CTA/emphasis/tone

**NO viewport stories allowed!**

---

## ✅ QUALITY CHECKLIST

For EACH component:
- [ ] Story file created (if new)
- [ ] Marketing documentation section added
- [ ] All copy from home page documented
- [ ] Minimum 3 stories (HomePageDefault + 2 variants)
- [ ] NO MobileView/TabletView stories
- [ ] Props documented in argTypes
- [ ] Examples show realistic usage
- [ ] Tested in Storybook (light + dark mode)
- [ ] No console errors
- [ ] Committed with proper message

**Total: 120 checklist items (10 per component × 12 components)**

---

## 📊 PROGRESS TRACKER

- [ ] HeroSection ✅ Enhanced with marketing docs
- [ ] WhatIsRbee ✅ Enhanced with marketing docs
- [ ] HomeSolutionSection ✅ Story created + marketing docs
- [ ] HowItWorksSection ✅ Story created + marketing docs
- [ ] FeaturesSection ✅ Story created + marketing docs
- [ ] UseCasesSection ✅ Story created + marketing docs
- [ ] ComparisonSection ✅ Story created + marketing docs
- [ ] PricingSection ✅ Enhanced with marketing docs
- [ ] SocialProofSection ✅ Story created + marketing docs
- [ ] TechnicalSection ✅ Story created + marketing docs
- [ ] ProblemSection ✅ Enhanced with marketing docs
- [ ] CTASection ✅ Enhanced with marketing docs

**Completion: 0/12 (0%)**

---

## 🚀 COMMIT MESSAGES

```bash
# For new story files
git add src/organisms/ComponentName/ComponentName.stories.tsx
git commit -m "docs(storybook): create ComponentName story with marketing docs

- Added complete story file with 3+ variants
- Documented home page usage and copy strategy
- Added marketing analysis (audience, messaging, CTAs)
- No viewport-only stories (use toolbar)"

# For enhanced existing stories
git add src/organisms/ComponentName/ComponentName.stories.tsx
git commit -m "docs(storybook): enhance ComponentName with marketing docs

- Added marketing strategy documentation
- Created HomePageContext and variant stories
- Documented all usage contexts (home, pricing, etc.)
- Analyzed copy effectiveness and conversion elements"
```

---

## 📞 SUPPORT

**Stuck on marketing documentation?**
1. Read the actual component in `frontend/apps/commercial/app/page.tsx`
2. Look at the copy: What emotion does it evoke?
3. Identify CTAs: Where do they lead? What action do they drive?
4. Check tone: Aggressive? Friendly? Technical? Casual?

**Can't find the component?**
1. Grep for it: `grep -r "ComponentName" frontend/apps/commercial/app/`
2. Check imports in `page.tsx`
3. Look in `src/organisms/` for similar names

**Need examples?**
- Look at existing stories in `EmailCapture`, `FaqSection`, `ProblemSection`
- Read `STORYBOOK_DOCUMENTATION_STANDARD.md` for templates

---

**START TIME:** [Fill in]  
**END TIME:** [Fill in]  
**TEAM MEMBERS:** [Fill in]  
**STATUS:** 🔴 NOT STARTED

---

**BUILD THE MARKETING DOCS THAT ACTUALLY HELP! 🚀**

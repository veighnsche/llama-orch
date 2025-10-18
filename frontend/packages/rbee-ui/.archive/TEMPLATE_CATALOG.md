# rbee-ui Template Catalog

**Version:** 1.1  
**Last Updated:** Oct 17, 2025  
**Total Templates:** 41

This document catalogs all templates in the rbee-ui component library from a **reusability-first perspective**. 

## 🎯 Philosophy: Think Reusability, Not Specialization

**CRITICAL FOR DEVELOPERS:** Template names like "Enterprise" or "Providers" are **marketing labels**, not technical constraints. Every template is built with flexible props and can be repurposed for different contexts.

**Before creating a new template:**
1. ✅ Review this catalog with a reusability mindset
2. ✅ Try adapting existing templates with different props
3. ✅ Only propose new templates if existing ones truly cannot be adapted
4. ✅ Write a proposal document justifying why reuse won't work

**Examples of creative reuse:**
- `ProvidersEarnings` → Research cost calculator
- `EnterpriseCompliance` → Any three-pillar feature showcase
- `EnterpriseSecurity` → Any grid of detailed feature cards
- `HomeHero` → Any hero with interactive demo (terminal, calculator, visualizer)

---

## Table of Contents

1. [Hero Templates](#hero-templates)
2. [Content Templates](#content-templates)
3. [Feature Templates](#feature-templates)
4. [Comparison & Pricing Templates](#comparison--pricing-templates)
5. [CTA & Conversion Templates](#cta--conversion-templates)
6. [Enterprise-Specific Templates](#enterprise-specific-templates)
7. [Provider-Specific Templates](#provider-specific-templates)
8. [Use Case Templates](#use-case-templates)
9. [Support Templates](#support-templates)

---

## Hero Templates

Hero templates are full-width, above-the-fold sections that introduce pages with headlines, CTAs, and visual elements.

### 1. **HeroTemplate** (Base)
- **File:** `src/templates/HeroTemplate/HeroTemplate.tsx`
- **Purpose:** Reusable base hero component with flexible layout options
- **Key Features:**
  - Badge, headline, subcopy
  - Proof elements (bullets, stats, indicators, assurance items)
  - Trust elements (badges, compliance chips)
  - Configurable aside slot
  - Animation support with stagger
  - Multiple background variants (gradient, solid, subtle-border)
- **Used By:** All hero variants extend or compose this
- **Props:** `HeroTemplateProps`

### 2. **HomeHero**
- **File:** `src/templates/HomeHero/HomeHero.tsx`
- **Purpose:** Homepage hero with terminal demo and GPU visualization
- **Key Features:**
  - Terminal window with command output
  - GPU utilization bars
  - Floating KPI card (GPU pool, cost, latency)
  - Trust badges (GitHub, API compatibility, cost)
- **Used In:** `HomePage`
- **Props:** `HomeHeroProps`

### 3. **DevelopersHeroTemplate**
- **File:** `src/templates/DevelopersHero/DevelopersHeroTemplate.tsx`
- **Purpose:** Developer-focused hero with code examples
- **Key Features:**
  - Code block integration
  - Developer-centric messaging
  - Technical proof points
- **Used In:** `DevelopersPage`
- **Props:** `DevelopersHeroProps`

### 4. **EnterpriseHero**
- **File:** `src/templates/EnterpriseHero/EnterpriseHero.tsx`
- **Purpose:** Enterprise hero with audit console visualization
- **Key Features:**
  - Audit console with event log
  - Compliance chips (GDPR, SOC2, ISO 27001)
  - Floating badges (data residency, audit events)
  - Enterprise-focused stats
- **Used In:** `EnterprisePage`
- **Props:** `EnterpriseHeroProps`

### 5. **ProvidersHero**
- **File:** `src/templates/ProvidersHero/ProvidersHero.tsx`
- **Purpose:** GPU provider hero with earnings visualization
- **Key Features:**
  - Earnings calculator
  - GPU marketplace messaging
  - Provider-specific CTAs
- **Used In:** `ProvidersPage`
- **Props:** `ProvidersHeroProps`

### 6. **FeaturesHero**
- **File:** `src/templates/FeaturesHero/FeaturesHero.tsx`
- **Purpose:** Features page hero
- **Key Features:**
  - Feature highlights
  - Platform capabilities overview
- **Used In:** `FeaturesPage`
- **Props:** `FeaturesHeroProps`

### 7. **PricingHeroTemplate**
- **File:** `src/templates/PricingHero/PricingHeroTemplate.tsx`
- **Purpose:** Pricing page hero
- **Key Features:**
  - Pricing-focused messaging
  - Value proposition
- **Used In:** `PricingPage`
- **Props:** `PricingHeroProps`

### 8. **UseCasesHero**
- **File:** `src/templates/UseCasesHero/UseCasesHero.tsx`
- **Purpose:** Use cases page hero
- **Key Features:**
  - Use case overview
  - Industry highlights
- **Used In:** `UseCasesPage`
- **Props:** `UseCasesHeroProps`

### 🔄 Hero Templates: Reusability Analysis

**ALL hero templates can be adapted for ANY page.** The "specialized" variants are just prop configurations.

**Reuse strategies:**
- **HomeHero** → Any page needing interactive demo (Research: experiment runner, Homelab: setup wizard, Startups: ROI calculator)
- **EnterpriseHero** → Any page with data visualization (Research: experiment logs, Compliance: audit trails)
- **ProvidersHero** → Any page with calculator/estimator (Startups: cost savings, Education: learning hours)
- **HeroTemplate (base)** → **USE THIS FIRST** - Most flexible, accepts any aside content

**Decision tree:**
1. Can you use `HeroTemplate` with custom aside? → **YES: Use it**
2. Need interactive demo? → Consider `HomeHero` pattern
3. Need data table/console? → Consider `EnterpriseHero` pattern
4. Need calculator? → Consider `ProvidersHero` pattern
5. None fit? → **Write proposal for new template**

---

## Content Templates

Content templates structure information into readable, scannable sections.

### 9. **WhatIsRbee**
- **File:** `src/templates/WhatIsRbee/WhatIsRbee.tsx`
- **Purpose:** Explain what rbee is with features, stats, and visual
- **Key Features:**
  - Pronunciation tooltip
  - Feature cards (Independence, Privacy, All GPUs together)
  - Stats row (cost, privacy, GPU support)
  - Visual diagram (homelab network)
  - Dual CTAs
- **Used In:** `HomePage`
- **Props:** `WhatIsRbeeProps`

### 10. **ProblemTemplate**
- **File:** `src/templates/ProblemTemplate/ProblemTemplate.tsx`
- **Purpose:** Display problem cards with loss framing
- **Key Features:**
  - Grid of problem cards
  - Tone variants (destructive, primary, muted)
  - Optional tags (e.g., "Loss €50/mo")
  - Staggered animations
- **Used In:** `HomePage`, `ProvidersPage`, `EnterprisePage`
- **Props:** `ProblemTemplateProps`

### 11. **SolutionTemplate**
- **File:** `src/templates/SolutionTemplate/SolutionTemplate.tsx`
- **Purpose:** Present solution with features, steps, and optional earnings
- **Key Features:**
  - Feature cards grid
  - Optional "How It Works" steps
  - Optional earnings/metrics sidebar
  - Optional topology diagram (BeeArchitecture)
  - Custom aside slot
- **Used In:** `HomePage`, `ProvidersPage`, `EnterprisePage`
- **Props:** `SolutionTemplateProps`

### 12. **HowItWorks**
- **File:** `src/templates/HowItWorks/HowItWorks.tsx`
- **Purpose:** Step-by-step guide with code/terminal blocks
- **Key Features:**
  - Numbered steps
  - Code blocks (syntax highlighted)
  - Terminal windows
  - Copy-to-clipboard support
- **Used In:** `HomePage`
- **Props:** `HowItWorksProps`

### 13. **TechnicalTemplate**
- **File:** `src/templates/TechnicalTemplate/TechnicalTemplate.tsx`
- **Purpose:** Technical deep-dive with architecture and tech stack
- **Key Features:**
  - Architecture highlights
  - BDD coverage progress
  - Architecture diagram
  - Tech stack cards
  - Stack links (GitHub, license, docs)
- **Used In:** `HomePage`
- **Props:** `TechnicalTemplateProps`

### 14. **AudienceSelector**
- **File:** `src/templates/AudienceSelector/AudienceSelector.tsx`
- **Purpose:** Persona-based navigation cards
- **Key Features:**
  - Three persona cards (Developers, GPU Owners, Enterprise)
  - Icon, category, title, description, features
  - Color-coded cards
  - Decision labels
  - Optional gradient backplate
- **Used In:** `HomePage`
- **Props:** `AudienceSelectorProps`

### 🔄 Content Templates: Reusability Analysis

**These are your workhorses.** Every content template is highly adaptable.

**Reuse strategies:**
- **ProblemTemplate** → ANY pain points (Research: reproducibility issues, Homelab: complexity, Startups: resource constraints)
- **SolutionTemplate** → ANY solution showcase (just change feature cards and aside content)
- **HowItWorks** → ANY step-by-step guide (setup, workflow, process)
- **TechnicalTemplate** → ANY technical deep-dive (architecture, stack, methodology)
- **AudienceSelector** → ANY multi-path decision (use cases, deployment options, pricing tiers)
- **WhatIsRbee** → ANY "What is X?" explainer (just change the content)

**Key insight:** These templates are **content-agnostic**. They define structure, not meaning.

**Example:** `ProblemTemplate` doesn't care if it's showing "Enterprise compliance risks" or "Research reproducibility challenges" - it just renders cards with icons, titles, and descriptions.

---

## Feature Templates

Feature templates showcase platform capabilities with interactive elements.

### 15. **FeaturesTabs**
- **File:** `src/templates/FeaturesTabs/FeaturesTabs.tsx`
- **Purpose:** Interactive tabbed feature showcase
- **Key Features:**
  - Tab navigation (API, GPU, Scheduler, SSE)
  - Code blocks, terminal windows, GPU bars
  - Highlight callouts
  - Benefits list
  - Mobile-friendly
- **Used In:** `HomePage`
- **Props:** `FeaturesTabsProps`

### 16. **AdditionalFeaturesGrid**
- **File:** `src/templates/AdditionalFeaturesGrid/AdditionalFeaturesGrid.tsx`
- **Purpose:** Grid of additional feature cards
- **Key Features:**
  - Feature cards with icons
  - Grid layout
- **Used In:** Various pages
- **Props:** `AdditionalFeaturesGridProps`

### 17. **CrossNodeOrchestration**
- **File:** `src/templates/CrossNodeOrchestration/CrossNodeOrchestration.tsx`
- **Purpose:** Showcase cross-node orchestration capability
- **Key Features:**
  - Multi-node visualization
  - Orchestration benefits
- **Used In:** `FeaturesPage`
- **Props:** `CrossNodeOrchestrationProps`

### 18. **IntelligentModelManagement**
- **File:** `src/templates/IntelligentModelManagement/IntelligentModelManagement.tsx`
- **Purpose:** Showcase model management features
- **Key Features:**
  - Model catalog
  - Auto-download
  - Caching
- **Used In:** `FeaturesPage`
- **Props:** `IntelligentModelManagementProps`

### 19. **ErrorHandlingTemplate**
- **File:** `src/templates/ErrorHandlingTemplate/ErrorHandlingTemplate.tsx`
- **Purpose:** Showcase error handling and resilience
- **Key Features:**
  - Error scenarios
  - Recovery strategies
- **Used In:** `FeaturesPage`
- **Props:** `ErrorHandlingTemplateProps`

### 20. **RealTimeProgress**
- **File:** `src/templates/RealTimeProgress/RealTimeProgress.tsx`
- **Purpose:** Showcase real-time progress tracking (SSE)
- **Key Features:**
  - Live event stream
  - Progress visualization
- **Used In:** `FeaturesPage`
- **Props:** `RealTimeProgressProps`

### 21. **MultiBackendGpuTemplate**
- **File:** `src/templates/MultiBackendGpuTemplate/MultiBackendGpuTemplate.tsx`
- **Purpose:** Showcase multi-backend GPU support
- **Key Features:**
  - CUDA, Metal, CPU backends
  - GPU utilization
- **Used In:** `FeaturesPage`
- **Props:** `MultiBackendGpuTemplateProps`

### 22. **SecurityIsolation**
- **File:** `src/templates/SecurityIsolation/SecurityIsolation.tsx`
- **Purpose:** Showcase security and isolation features
- **Key Features:**
  - Process isolation
  - Security layers
- **Used In:** `FeaturesPage`
- **Props:** `SecurityIsolationProps`

### 23. **CodeExamplesTemplate**
- **File:** `src/templates/CodeExamples/CodeExamplesTemplate.tsx`
- **Purpose:** Display code examples with syntax highlighting
- **Key Features:**
  - Multiple code blocks
  - Language support
  - Copy functionality
- **Used In:** Various pages
- **Props:** `CodeExamplesTemplateProps`

### 🔄 Feature Templates: Reusability Analysis

**Feature showcases are domain-agnostic presentation patterns.**

**Reuse strategies:**
- **FeaturesTabs** → ANY tabbed content (Research: methodologies, Homelab: setup modes, Startups: deployment options)
- **CrossNodeOrchestration** → ANY distributed/network visualization (Homelab: multi-machine setup, Research: distributed experiments)
- **IntelligentModelManagement** → ANY automated management showcase (Homelab: auto-configuration, Startups: auto-scaling)
- **ErrorHandlingTemplate** → ANY resilience/reliability showcase (Research: experiment recovery, Homelab: failover)
- **RealTimeProgress** → ANY live updates/streaming (Research: experiment monitoring, Homelab: deployment progress)
- **MultiBackendGpuTemplate** → ANY multi-option support (Research: multiple frameworks, Homelab: multiple OSes)
- **SecurityIsolation** → ANY security/isolation showcase (Research: data privacy, Compliance: access controls)
- **CodeExamplesTemplate** → ANY code/configuration examples (universal)

**Key insight:** These templates show **"how it works"** with visual proof. The domain doesn't matter—only the pattern matters (tabs, visualization, code, progress, etc.).

---

## Comparison & Pricing Templates

Templates for comparing offerings and displaying pricing.

### 24. **ComparisonTemplate**
- **File:** `src/templates/ComparisonTemplate/ComparisonTemplate.tsx`
- **Purpose:** Feature comparison matrix
- **Key Features:**
  - Column headers (rbee vs competitors)
  - Row-based feature comparison
  - Check/X icons for boolean values
  - Text values for details
  - Optional notes
  - Legend
  - Mobile card view
- **Used In:** `HomePage`, `EnterprisePage`
- **Props:** `ComparisonTemplateProps`

### 25. **PricingTemplate**
- **File:** `src/templates/PricingTemplate/PricingTemplate.tsx`
- **Purpose:** Pricing tiers display
- **Key Features:**
  - Three tiers (Home/Lab, Team, Enterprise)
  - Monthly/yearly toggle
  - Feature lists
  - Highlighted tier
  - Save badges
  - Editorial image
  - Footer disclaimer
- **Used In:** `PricingPage`
- **Props:** `PricingTemplateProps`

### 26. **PricingComparisonTemplate**
- **File:** `src/templates/PricingComparisonTemplate/PricingComparisonTemplate.tsx`
- **Purpose:** Pricing comparison across providers
- **Key Features:**
  - Cost comparison
  - Value proposition
- **Used In:** `PricingPage`
- **Props:** `PricingComparisonTemplateProps`

### 🔄 Comparison & Pricing Templates: Reusability Analysis

**Comparison matrices and tiered options are universal decision-making tools.**

**Reuse strategies:**
- **ComparisonTemplate** → ANY feature comparison:
  - Research: Methodology comparison (Quantitative vs Qualitative vs Mixed)
  - Homelab: Setup options (Single-node vs Multi-node vs Hybrid)
  - Startups: Deployment options (Cloud vs On-prem vs Hybrid)
  - Education: Learning paths (Beginner vs Intermediate vs Advanced)

- **PricingTemplate** → ANY tiered options:
  - Research: Access tiers (Free dataset vs Premium vs Enterprise)
  - Homelab: Setup complexity (Basic vs Advanced vs Expert)
  - Education: Course levels (Intro vs Professional vs Certification)
  - Compliance: Support levels (Community vs Business vs Enterprise)

- **PricingComparisonTemplate** → ANY cost/value comparison

**Key insight:** These templates present **structured choices**. Whether it's pricing, features, or options, the pattern is the same: columns, rows, checkmarks, and highlights.

---

## CTA & Conversion Templates

Templates focused on driving user actions.

### 27. **CTATemplate**
- **File:** `src/templates/CTATemplate/CTATemplate.tsx`
- **Purpose:** Call-to-action section with buttons
- **Key Features:**
  - Eyebrow, title, subtitle
  - Primary and secondary CTAs
  - Icon support (left/right)
  - Trust note
  - Emphasis variants (none, gradient)
  - Alignment options
- **Used In:** All pages (final CTA)
- **Props:** `CTATemplateProps`

### 28. **EmailCapture**
- **File:** `src/templates/EmailCapture/EmailCapture.tsx`
- **Purpose:** Email signup/waitlist form
- **Key Features:**
  - Badge with pulse animation
  - Headline and subheadline
  - Email input with validation
  - Submit button
  - Success message
  - Trust message
  - Community footer with link
  - Optional bee glyph decorations
  - Optional illustration
- **Used In:** `HomePage`, `EnterprisePage`
- **Props:** `EmailCaptureProps`

### 🔄 CTA & Conversion Templates: Reusability Analysis

**Every page needs a CTA. These are your conversion workhorses.**

**Reuse strategies:**
- **CTATemplate** → **Universal.** Use on EVERY page. Just change the copy.
- **EmailCapture** → ANY lead capture (waitlist, newsletter, beta access, course enrollment, dataset access, etc.)

**Key insight:** CTAs are **pure conversion mechanics**. The template doesn't care what you're converting to—it just needs a headline, buttons, and trust signals.

**Example:** `EmailCapture` works equally well for:
- "Join Research Beta" (Research page)
- "Get Homelab Guide" (Homelab page)
- "Start Free Trial" (Startups page)
- "Access Course Materials" (Education page)

---

## Enterprise-Specific Templates

Templates designed for enterprise messaging and compliance.

### 29. **EnterpriseCompliance**
- **File:** `src/templates/EnterpriseCompliance/EnterpriseCompliance.tsx`
- **Purpose:** Display compliance certifications (GDPR, SOC2, ISO 27001)
- **Key Features:**
  - Three compliance pillars
  - Bullet lists per pillar
  - Compliance endpoints boxes
  - Trust service criteria
  - ISMS controls
- **Used In:** `EnterprisePage`
- **Props:** `EnterpriseComplianceProps`

### 30. **EnterpriseSecurity**
- **File:** `src/templates/EnterpriseSecurity/EnterpriseSecurity.tsx`
- **Purpose:** Showcase enterprise security features
- **Key Features:**
  - Six security crates (auth-min, audit-logging, input-validation, secrets-management, jwt-guardian, deadline-propagation)
  - Security cards with bullets
  - Docs links
- **Used In:** `EnterprisePage`
- **Props:** `EnterpriseSecurityProps`

### 31. **EnterpriseHowItWorks**
- **File:** `src/templates/EnterpriseHowItWorks/EnterpriseHowItWorks.tsx`
- **Purpose:** Enterprise deployment process
- **Key Features:**
  - Four deployment steps
  - Timeline visualization
  - Step cards with icons
- **Used In:** `EnterprisePage`
- **Props:** `EnterpriseHowItWorksProps`

### 32. **EnterpriseUseCases**
- **File:** `src/templates/EnterpriseUseCases/EnterpriseUseCases.tsx`
- **Purpose:** Industry-specific use cases (Finance, Healthcare, Legal, Government)
- **Key Features:**
  - Industry cards with badges
  - Challenges and solutions
  - Segments
  - Links to industry pages
- **Used In:** `EnterprisePage`
- **Props:** `EnterpriseUseCasesProps`

### 33. **EnterpriseCTA**
- **File:** `src/templates/EnterpriseCTA/EnterpriseCTA.tsx`
- **Purpose:** Enterprise-specific CTA with multiple options
- **Key Features:**
  - Trust stats
  - Three CTA options (Demo, Compliance Pack, Sales)
  - Eyebrows and notes
- **Used In:** `EnterprisePage`
- **Props:** `EnterpriseCTAProps`

### 🔄 Enterprise Templates: Reusability Analysis

**"Enterprise" is just a label.** These templates show structured information with authority.

**Reuse strategies:**
- **EnterpriseCompliance** → ANY three-pillar showcase (Research: methodologies, Homelab: setup options, Startups: growth stages)
- **EnterpriseSecurity** → ANY grid of detailed features (Research: tools, Education: curriculum modules, DevOps: deployment options)
- **EnterpriseHowItWorks** → ANY process with timeline (Startups: growth roadmap, Research: experiment workflow, Homelab: migration path)
- **EnterpriseUseCases** → ANY industry/segment breakdown (Education: student types, Research: disciplines, Homelab: use cases)
- **EnterpriseCTA** → ANY multi-option CTA (Research: access methods, Startups: onboarding paths, Education: enrollment options)

**Key insight:** These templates emphasize **credibility and detail**. Use them when you need to show thoroughness and professionalism, regardless of audience.

**Example:** `EnterpriseCompliance` with its three pillars + bullet lists + boxes can showcase "Research Methodologies" (Quantitative/Qualitative/Mixed) just as well as compliance standards.

---

## Provider-Specific Templates

Templates for GPU provider audience.

### 34. **ProvidersEarnings**
- **File:** `src/templates/ProvidersEarnings/ProvidersEarnings.tsx`
- **Purpose:** Interactive earnings calculator
- **Key Features:**
  - GPU selection
  - Availability slider
  - Earnings projection
  - Monthly/yearly toggle
  - Detailed breakdown
- **Used In:** `ProvidersPage`
- **Props:** `ProvidersEarningsProps`

### 35. **ProvidersCTA**
- **File:** `src/templates/ProvidersCTA/ProvidersCTA.tsx`
- **Purpose:** Provider-specific CTA
- **Key Features:**
  - Provider-focused messaging
  - Earnings emphasis
- **Used In:** `ProvidersPage`
- **Props:** `ProvidersCTAProps`

### 🔄 Provider Templates: Reusability Analysis

**Interactive calculators and earnings projections work for ANY quantifiable benefit.**

**Reuse strategies:**
- **ProvidersEarnings** → ANY calculator/estimator:
  - Research: Experiment cost calculator (compute hours × cost)
  - Startups: ROI calculator (savings × time)
  - Education: Learning time estimator (modules × hours)
  - Homelab: Power cost calculator (GPUs × electricity rate)
  - Compliance: Audit cost estimator (events × retention)

**Key insight:** `ProvidersEarnings` is a **generic interactive calculator** with:
- Selection inputs (dropdown, sliders)
- Real-time calculation
- Visual breakdown
- Toggle options (monthly/yearly, etc.)

**Don't be fooled by the name.** This template calculates and displays ANY numeric projection.

---

## Use Case Templates

Templates for showcasing use cases and industries.

### 36. **UseCasesTemplate**
- **File:** `src/templates/UseCasesTemplate/UseCasesTemplate.tsx`
- **Purpose:** Grid of use case cards (solo dev, team, homelab, enterprise, AI-dependent coder, agentic builder)
- **Key Features:**
  - Use case cards with scenario/solution/outcome
  - Icon, title, description
  - Optional tags and CTAs
  - Configurable columns
- **Used In:** `HomePage`
- **Props:** `UseCasesTemplateProps`

### 37. **UseCasesPrimaryTemplate**
- **File:** `src/templates/UseCasesPrimaryTemplate/UseCasesPrimaryTemplate.tsx`
- **Purpose:** Primary use case showcase
- **Key Features:**
  - Featured use cases
  - Detailed scenarios
- **Used In:** `UseCasesPage`
- **Props:** `UseCasesPrimaryTemplateProps`

### 38. **UseCasesIndustryTemplate**
- **File:** `src/templates/UseCasesIndustryTemplate/UseCasesIndustryTemplate.tsx`
- **Purpose:** Industry-specific use cases
- **Key Features:**
  - Industry cards
  - Sector-specific messaging
- **Used In:** `UseCasesPage`
- **Props:** `UseCasesIndustryTemplateProps`

### 🔄 Use Case Templates: Reusability Analysis

**Scenario → Solution → Outcome structure works for ANY narrative.**

**Reuse strategies:**
- **UseCasesTemplate** → ANY scenario-based cards:
  - Research: Experiment types (scenario: need reproducibility, solution: deterministic seeds, outcome: publishable results)
  - Homelab: Setup scenarios (scenario: multiple machines, solution: unified orchestration, outcome: single cluster)
  - Startups: Growth stages (scenario: scaling team, solution: shared infrastructure, outcome: reduced costs)
  - Education: Learning paths (scenario: beginner, solution: guided tutorials, outcome: production-ready skills)

**Key insight:** The template doesn't care about the domain. It just needs:
- Icon + title + description
- Scenario text
- Solution text  
- Outcome text

**This is pure storytelling structure.** Use it anywhere you need to show "before → during → after" narratives.

---

## Support Templates

Templates for support and informational content.

### 39. **FAQTemplate**
- **File:** `src/templates/FAQTemplate/FAQTemplate.tsx`
- **Purpose:** Frequently Asked Questions with search and filtering
- **Key Features:**
  - Searchable accordion
  - Category filtering
  - Expand/collapse all
  - Support card with image and links
  - JSON-LD schema for SEO
  - Empty state with keywords
- **Used In:** `HomePage`
- **Props:** `FAQTemplateProps`

### 40. **TestimonialsTemplate**
- **File:** `src/templates/TestimonialsTemplate/TestimonialsTemplate.tsx`
- **Purpose:** Display testimonials with stats
- **Key Features:**
  - Testimonial cards (avatar, author, role, quote)
  - Stats row (GitHub stars, installations, GPUs, cost)
  - Optional company logos
- **Used In:** `HomePage`, `EnterprisePage`
- **Props:** `TestimonialsTemplateProps`

### 41. **CardGridTemplate**
- **File:** `src/templates/CardGridTemplate/CardGridTemplate.tsx`
- **Purpose:** Generic grid of cards
- **Key Features:**
  - Configurable columns
  - Gap control
  - Generic card slots
- **Used In:** Various pages
- **Props:** `CardGridTemplateProps`
- **Note:** Marked for consolidation (see CONSOLIDATION_OPPORTUNITIES.md)

### 🔄 Support Templates: Reusability Analysis

**FAQ and social proof patterns are universal.**

**Reuse strategies:**
- **FAQTemplate** → ANY Q&A content:
  - Research: Methodology FAQs
  - Homelab: Setup troubleshooting
  - Startups: Business model questions
  - Education: Course FAQs
  - Compliance: Regulatory questions

- **TestimonialsTemplate** → ANY social proof:
  - Research: Academic citations/papers
  - Homelab: Community builds
  - Startups: Founder stories
  - Education: Student outcomes
  - Stats can be: papers published, community members, projects built, etc.

- **CardGridTemplate** → **Use this as your fallback.** Generic grid that accepts any card content.

**Key insight:** These templates provide **structure without opinion**. FAQ doesn't care if it's technical or business questions. Testimonials don't care if it's users, researchers, or students.

---

## Template Usage by Page

### HomePage
1. HomeHero
2. WhatIsRbee
3. AudienceSelector
4. EmailCapture
5. ProblemTemplate
6. SolutionTemplate
7. HowItWorks
8. FeaturesTabs
9. UseCasesTemplate
10. ComparisonTemplate
11. PricingTemplate
12. TestimonialsTemplate
13. TechnicalTemplate
14. FAQTemplate
15. CTATemplate

### EnterprisePage
1. EnterpriseHero
2. EmailCapture
3. ProblemTemplate
4. SolutionTemplate
5. EnterpriseCompliance
6. EnterpriseSecurity
7. EnterpriseHowItWorks
8. EnterpriseUseCases
9. ComparisonTemplate
10. CardGridTemplate (Enterprise Features)
11. TestimonialsTemplate
12. EnterpriseCTA

### ProvidersPage
1. ProvidersHero
2. ProblemTemplate
3. SolutionTemplate
4. ProvidersEarnings
5. ProvidersCTA

### FeaturesPage
1. FeaturesHero
2. CrossNodeOrchestration
3. IntelligentModelManagement
4. ErrorHandlingTemplate
5. RealTimeProgress
6. MultiBackendGpuTemplate
7. SecurityIsolation

### DevelopersPage
1. DevelopersHeroTemplate
2. CodeExamplesTemplate
3. (Additional templates TBD)

### PricingPage
1. PricingHeroTemplate
2. PricingTemplate
3. PricingComparisonTemplate

### UseCasesPage
1. UseCasesHero
2. UseCasesPrimaryTemplate
3. UseCasesIndustryTemplate

---

## Template Patterns

### Common Props Patterns

Most templates follow these prop patterns:

1. **Content Props:**
   - `title`, `description`, `headline`, `subcopy`
   - `icon`, `badge`, `tag`
   - `items`, `features`, `steps`

2. **CTA Props:**
   - `primaryCta`, `secondaryCta`
   - `ctaText`, `ctaHref`, `ctaVariant`

3. **Visual Props:**
   - `imageSrc`, `imageAlt`
   - `illustration`, `aside`
   - `topology`, `diagram`

4. **Styling Props:**
   - `className`, `gridClassName`
   - `variant`, `tone`, `color`
   - `align`, `layout`, `padding`

5. **Behavior Props:**
   - `animations`, `stagger`
   - `showMobileCards`
   - `jsonLdEnabled`

### Wrapper Pattern

Most templates are wrapped in `TemplateContainer` which provides:
- Background variants (gradient, solid, subtle-border, gradient-destructive)
- Padding control (sm, md, lg, xl, 2xl)
- Max-width constraints (3xl, 5xl, 6xl, 7xl)
- Alignment (center, left)
- Optional decorations (background SVGs)
- Optional CTAs, disclaimers, ribbons

---

## Consolidation Opportunities

See `CONSOLIDATION_OPPORTUNITIES.md` for detailed analysis of:
- Templates that can be merged
- Duplicate functionality
- Refactoring recommendations
- Estimated LOC savings

**Key Findings:**
- 38+ components can be consolidated
- 1,500-1,900 lines can be removed
- ~40% maintenance reduction possible

---

## Template Development Guidelines

### Creating a New Template

1. **Location:** `src/templates/[TemplateName]/`
2. **Files:**
   - `[TemplateName].tsx` - Main component
   - `[TemplateName].stories.tsx` - Storybook stories
   - `index.ts` - Barrel export

3. **Structure:**
   ```tsx
   // Types
   export type [TemplateName]Props = {
     // Props definition
   }

   // Component
   export function [TemplateName]({ ...props }: [TemplateName]Props) {
     // Implementation
   }
   ```

4. **Export:** Add to `src/templates/index.ts`

5. **Documentation:**
   - Add JSDoc comments
   - Include usage examples
   - Document all props

### Best Practices

1. **Consistency:** Follow existing patterns (see CONSOLIDATION_OPPORTUNITIES.md)
2. **Accessibility:** Include ARIA labels, semantic HTML, keyboard navigation
3. **Responsiveness:** Mobile-first design, breakpoint-aware
4. **Performance:** Lazy load images, optimize animations, minimize re-renders
5. **Theming:** Use design tokens, support light/dark modes
6. **Testing:** Write Storybook stories, test all prop combinations

---

## Related Documentation

- **CONSOLIDATION_OPPORTUNITIES.md** - Template consolidation analysis
- **NEW_BACKGROUNDS_PLAN.md** - Background decoration system
- **NAVIGATION_REDESIGN_PLAN.md** - Navigation structure
- **BACKGROUND_FIX_SUMMARY.md** - Z-index and stacking context fixes

---

**Maintained by:** rbee-ui team  
**Questions?** See `frontend/packages/rbee-ui/README.md`

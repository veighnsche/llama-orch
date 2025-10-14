# Commercial Frontend - Missing Pages Analysis

**Generated:** 2025-10-14  
**Updated:** 2025-10-14  
**Status:** Aligned with IMPLEMENTATION_PLAN.md  
**Scope:** All pages in `/frontend/bin/commercial/app/`  
**Note:** Documentation is deferred to separate Next.js project - excluded from this analysis

---

## Executive Summary

After analyzing all 7 existing pages and their components, I found **15 unique commercial links** pointing to pages that don't exist yet (excluding all `/docs/*` links). This document categorizes them by priority for the **commercial frontend only**.

**This analysis is aligned with IMPLEMENTATION_PLAN.md, which contains the detailed execution strategy, component reuse patterns, and week-by-week timeline.**

---

## 1. Critical Pages (Must Create)

These pages are linked from multiple locations and are essential for user journeys:

### 1.1 `/getting-started`
**Linked from:**
- Homepage CTA (primary button)
- Developers page CTA
- Developers problem section
- Comparison section

**Recommendation:** ✅ **CREATE THIS PAGE**  
**Why:** This is your primary conversion point. Users clicking "Get Started Free" expect a landing page.  
**Content:** 
- Hero: "Get Started with rbee in 15 Minutes"
- Quick overview of what they'll accomplish
- **CTA button:** "View Installation Guide" → Links to docs project (external)
- Alternative paths: "Watch Video Tutorial", "Join Discord for Help"
- Prerequisites checklist (GPU requirements, OS support)
- Community resources section
- **This is a marketing bridge page, not technical docs**

---

### 1.2 `/contact`
**Linked from:**
- Footer
- Pricing comparison ("Talk to Sales")
- Enterprise CTA (multiple variants)
- AudienceSelector bottom links

**Recommendation:** ✅ **CREATE THIS PAGE**  
**Why:** Essential for enterprise leads and support requests.  
**Content:** 
- Hero: "Get in Touch"
- Contact form with fields:
  - Inquiry type dropdown (Sales, Demo, Partnership, Support, General)
  - Name, Email, Company (optional)
  - Message
- Use URL params to pre-select type: `/contact?type=demo`, `/contact?type=sales`
- Response time expectations: "We'll reply within 1 business day"
- Alternative contact methods: Discord, GitHub Discussions

**Consolidates these links:**
- `/enterprise/demo`
- `/contact/industry-brief`
- `/contact/sales`
- `/contact/solutions`

---

## 2. High-Priority Pages (Should Create)

### 2.1 `/blog`
**Linked from:** Footer

**Recommendation:** ✅ **CREATE THIS PAGE**  
**Why:** Important for SEO, thought leadership, and keeping users engaged.  
**Content:** 
- Blog listing page with hero
- Start with 3-5 articles:
  - "Why We Built rbee"
  - "Self-Hosting LLMs: A Complete Guide"
  - "rbee vs Cloud Providers: Cost Analysis"
  - "Privacy-First AI Infrastructure"
- Blog post template with: title, date, author, tags, content
- Consider using MDX for blog posts

---

### 2.2 `/about`
**Linked from:** Footer

**Recommendation:** ✅ **CREATE THIS PAGE**  
**Why:** Builds trust, especially for enterprise customers.  
**Content:** 
- Hero: "About rbee"
- Company story / origin
- Mission: Privacy-first AI infrastructure
- Values: Open source, transparency, EU-first
- Team section (if applicable)
- Open-source commitment
- Community stats

---

### 2.3 `/legal/privacy` & `/legal/terms`
**Linked from:** Footer

**Recommendation:** ✅ **CREATE THESE PAGES**  
**Why:** Legal requirement, especially with GDPR claims throughout your site.  
**Content:** 
- Standard privacy policy (GDPR-compliant)
- Terms of service
- Cookie policy (if using cookies)
- Data retention policies
- **Critical:** Your site makes GDPR/compliance claims, so these must exist

---

## 3. Medium-Priority Pages

### 3.1 `/security`
**Linked from:** PledgeCallout component (homepage)

**Recommendation:** ✅ **CREATE THIS PAGE**  
**Why:** You make security claims throughout the site. Users clicking "Security details" expect content.  
**Content:**
- Hero: "Security Architecture"
- Zero-trust authentication
- Immutable audit trails
- Bind policies
- Vulnerability reporting process
- Security best practices for self-hosting
- Link to GitHub security policy
- **Note:** This is marketing/overview, not technical docs

---

### 3.2 `/story` - Hidden Page
**Linked from:** Nowhere (hidden Easter egg)

**Recommendation:** ✅ **CREATE THIS PAGE**  
**Why:** Unique story about how rbee was built using AI-driven development. Great for tech-savvy audience.  
**Content:**
- **Hero:** "How rbee Was Built: 99% AI-Generated Code"
- **Subtitle:** "The story of Character-Driven Development and AI engineering teams"
- **Sections:**
  1. **The Problem** - AI coders drift in large codebases
  2. **The Solution** - Character-Driven Development (CDD)
  3. **The Teams** - 6 AI teams with distinct personalities:
     - Testing Team 🔍 (anti-cheating kingpins)
     - auth-min Team 🎭 (trickster guardians)
     - Performance Team ⏱️ (obsessive timekeepers)
     - Audit Logging Team 🔒 (compliance engine)
     - Narration Core Team 🎀 (observability artists)
     - Developer Experience Team 🎨 (readability minimalists)
  4. **The Process** - TEAM-XXX handoffs, BDD scenarios, debates
  5. **The Results** - 42/62 scenarios passing, production-ready code
  6. **The Innovation** - AI teams debate design decisions
  7. **Real Examples** - Code reviews, fines, optimizations
- **Tone:** Technical, transparent, fascinating behind-the-scenes
- **CTA:** "Want to contribute? Join the revolution"
- **Note:** NOT linked in navigation - Easter egg for curious developers

---

### 3.3 Industry Pages - CONSOLIDATE
**Links found:**
- `/industries/finance` (4 links)
- `/industries/healthcare` (3 links)
- `/industries/legal` (3 links)
- `/industries/government` (3 links)

**Linked from:** Enterprise use cases section

**Recommendation:** ⚠️ **CREATE ONE PAGE: `/industries`**  
**Why:** You don't have enough unique content for 4 separate pages yet.  
**Alternative Solution:**
1. Create **ONE page**: `/industries`
2. Sections for each industry: Finance, Healthcare, Legal, Government
3. Each section: compliance requirements, challenges, solutions
4. Update all links to use anchor links: `/industries#finance`, `/industries#healthcare`, etc.

**OR:** Use anchor links on existing `/enterprise` page: `/enterprise#finance`

**Don't create 4 separate pages until you have:**
- Real customer case studies for each
- Industry-specific features
- Unique compliance documentation

---

## 4. Links to Update (No New Pages)

### 4.1 External Links (Point to Docs Project)
These should link to your separate docs Next.js project when ready:
- All `/docs/*` links → `https://docs.rbee.io/*` (or subdomain)
- `/getting-started` CTA button → External docs link

### 4.2 Anchor Links (Update Components)
**Update these in your components:**
- `#compare` (AudienceSelector) → Ensure ComparisonSection has `id="compare"`
- `#contact` (AudienceSelector) → Change to `/contact`
- `#get-started` (multiple) → Change to `/getting-started`
- `#how-it-works` (Developers hero) → Ensure section has `id="how-it-works"`
- `#pricing` (Footer) → Ensure PricingSection has `id="pricing"`
## Summary: Commercial Pages to Create

**Aligned with IMPLEMENTATION_PLAN.md - See that document for detailed component strategy and timeline.**

### Tier 1 - Critical (Create First) - MUST HAVE
1. ✅ `/getting-started` - Marketing bridge to docs
2. ✅ `/contact` - Single contact form (consolidate 4+ variants)

### Tier 2 - High Priority (Create Next) - MUST HAVE
3. ✅ `/blog` - Blog listing page
4. ✅ `/about` - About page
5. ✅ `/legal/privacy` - Privacy policy
6. ✅ `/legal/terms` - Terms of service

### Tier 3 - Medium Priority (Create Later) - MUST HAVE
7. ✅ `/security` - Security overview (marketing, not docs)
8. ✅ `/industries` - Single page with all 4 industries (NOT 4 separate pages)
9. ✅ `/story` - Hidden page: "How rbee was built" (99% AI-generated, Character-Driven Development)

**Total Must-Have: 9 pages**

**Implementation Details:** See IMPLEMENTATION_PLAN.md for:
- Component reuse strategy (133 organisms, 50 molecules, 71 atoms)
- Only 5 new components needed
- Copy sources (90% already written in stakeholder docs)
- Week-by-week execution plan

### Tier 4 - Audience-Specific Landing Pages (DEFERRED - Not in Current Plan)

**Analysis of 10 audiences from VIDEO_SCRIPTS.md:**

| Audience | Existing Page Coverage | Need Dedicated Page? |
|----------|----------------------|---------------------|
| 🚀 Startup Founders | `/pricing`, `/use-cases` | ❌ No - generic works |
| 💼 Enterprise CTOs | `/enterprise` ✅ | ❌ Already covered |
| 👨‍💻 Software Developers | `/developers` ✅ | ❌ Already covered |
| 🏠 Homelab Enthusiasts | None | ⚠️ **MAYBE** - unique pain points |
| 💰 Angel Investors/VCs | None | ⚠️ **MAYBE** - pitch deck style |
| 🔬 AI Researchers | None | ❌ No - overlap with developers |
| 🌍 EU Businesses | `/enterprise` ✅ | ❌ Already covered |
| 🎓 CS Students | None | ❌ No - overlap with developers |
| 🤖 AI Dev Community | `/story` (hidden) | ✅ Already planned |
| 💻 DevOps/SREs | `/developers` ✅ | ❌ Already covered |

**Verdict:** Only **2 audiences** have truly unique pain points not covered by existing pages:
1. **Homelabbers** - "Use idle GPUs across home network" (very specific use case)
2. **Investors** - Pitch deck format (completely different content type)

**Status:** DEFERRED - Not included in IMPLEMENTATION_PLAN.md. Focus on 9 must-have pages first.

**Note:** The following audience-specific pages are NOT in the current implementation plan. Deferred until 9 must-have pages are complete.

#### 10. `/for/homelabbers` - Homelab Enthusiasts & Self-Hosters (DEFERRED)
**Content:**
- **Hero:** "Use ALL Your Home Network Hardware for AI"
- **Pain Point:** "Idle GPUs gathering dust across your homelab"
- **Solution:** SSH-based control, pick which GPU runs which model
- **Features:**
  - Multi-backend: CUDA, Metal, CPU
  - Model catalog with auto-download
  - Idle timeout frees VRAM
  - Cascading shutdown (no orphaned processes)
  - Complete control, complete privacy
- **Use Cases:** 
  - Power Zed IDE with your gaming PC
  - Build AI agents using spare workstation
  - Monetize idle capacity (marketplace)
- **CTA:** "Start Using Your Hardware Today"

#### 11. `/for/researchers` - AI Researchers & ML Engineers (DEFERRED)
**Content:**
- **Hero:** "Reproducible AI Research Infrastructure"
- **Pain Point:** "Can't reproduce results, no deterministic testing"
- **Solution:** Proof bundles, same seed → same output
- **Features:**
  - Multi-modal: LLMs, Stable Diffusion, TTS, embeddings
  - Candle-powered (Rust ML framework)
  - BDD-tested with executable specs
  - Backend auto-detection
  - Property tests for invariants
  - Determinism suite for regression testing
- **Technical Details:**
  - Research-grade quality
  - Production-ready infrastructure
  - Test reproducibility for CI/CD
- **CTA:** "Try Reproducible AI Research"

#### 12. `/for/investors` - Angel Investors & VCs (DEFERRED)
**Content:** (Pitch deck style)
- **Hero:** "The Future of AI Infrastructure"
- **Problem:** Developers fear building with AI (provider dependency)
- **Solution:** Own your AI infrastructure
- **Market Opportunity:**
  - AI coding boom + fear of provider lock-in
  - Perfect timing
- **Traction:**
  - 68% complete (42/62 BDD tests)
  - 99% AI-generated code
  - 11 shared crates already built
- **Business Model:**
  - Year 1: 35 customers, €70K revenue
  - Year 2: 100 customers, €360K revenue
  - Year 3: €1M+ revenue (marketplace)
- **Pricing:** €99-299/month SaaS + 30-40% marketplace fees
- **Moats:**
  - EU-compliance (GDPR-native)
  - Multi-modal support
  - User-scriptable routing (Rhai)
  - GPL license (copyleft protection)
- **The Ask:** "Schedule a call to discuss investment"
- **CTA:** "Download Pitch Deck" / "Schedule Call"

#### 13. `/for/students` - CS Students & Educators (DEFERRED)
**Content:**
- **Hero:** "Learn Distributed Systems from Nature"
- **Value Prop:** Study real production code, not toy examples
- **Architecture:**
  - 4 components mirror a beehive
  - Queen (orchestrator), Hive (pool manager), Workers (executors), Keeper (interface)
  - Smart/dumb architecture pattern
- **Learning Opportunities:**
  - Distributed systems
  - Rust + ML (Candle)
  - BDD testing with Gherkin
  - Multi-backend support (CUDA, Metal, CPU)
  - Production system design
- **Open Source:** GPL-3.0, study real code
- **Educational Use:**
  - Class projects
  - Thesis topics
  - Distributed systems labs
- **CTA:** "Start Learning" / "View Architecture"

---

## Pages You Should NOT Create

### ❌ Don't Create in Commercial Frontend
All `/docs/*` pages go in separate docs project:
- `/docs`
- `/docs/architecture`
- `/docs/setup`
- `/docs/errors`
- `/docs/compliance`
- `/docs/compliance-pack`
- `/docs/quickstart`

### ❌ Don't Create Separate Industry Pages
**Instead of:**
- `/industries/finance`
- `/industries/healthcare`
- `/industries/legal`
- `/industries/government`

**Do this:**
- Create ONE page: `/industries` with sections for each
- OR use anchor links: `/enterprise#finance`, etc.
- Update all 13 industry links to use anchors

### ❌ Don't Create Multiple Contact Pages
**Instead of:**
- `/enterprise/demo`
- `/contact/industry-brief`
- `/contact/sales`
- `/contact/solutions`

**Do this:**
- Create ONE page: `/contact`
- Use URL params: `/contact?type=demo`, `/contact?type=sales`
- Pre-populate form based on param

---

## Component Updates Required

### Update Links in Components
1. **WhatIsRbee.tsx** - Change `/technical-deep-dive` → External docs link
2. **All components** - Change `/docs/*` → External docs URLs
3. **Enterprise use cases** - Update industry links to `/industries#[industry]`
4. **All contact CTAs** - Update to `/contact?type=[type]`
5. **Anchor links** - Ensure target sections have matching IDs

### Add Missing IDs to Sections
Ensure these sections have proper IDs:
- `<ComparisonSection id="compare" />`
- `<PricingSection id="pricing" />`
- `<DevelopersHowItWorks>` - Add `id="how-it-works"`

---

## Implementation Priority

**See IMPLEMENTATION_PLAN.md for detailed week-by-week execution with specific tasks and checklists.**

```
Week 1: Critical Conversion Pages
├── Navigation & Footer updates (Day 1-2)
├── /getting-started (Day 3-4) - CRITICAL - primary CTA destination
└── /contact (Day 5-7) - HIGH - enterprise leads

Week 2: Trust & Legal
├── LegalPageLayout template (Day 8-9)
├── /legal/privacy (Day 10-11) - HIGH - GDPR requirement
├── /legal/terms (Day 12-13) - HIGH - legal requirement
└── /blog (Day 14) - HIGH - SEO & content marketing (placeholder)

Week 3: Supporting Pages
├── /about (Day 15-16) - HIGH - trust building
├── /security (Day 17-18) - MEDIUM - linked from homepage
├── /industries (Day 19-20) - MEDIUM - consolidate 4 industry pages
└── /story (Day 21) - Hidden page - AI development story
```

**Total Effort:** ~21 days for 9 pages + 5 new components

**New Components Needed (Only 5!):**
1. `ContactForm` (Molecule)
2. `BlogPostCard` (Molecule)
3. `IndustryCard` (Molecule)
4. `LegalPageLayout` (Template)
5. `TimelineSection` (Organism)

**Component Reuse:**
- 133 organisms already built
- 50 molecules already built
- 71 atoms already built
- Most pages = composition of existing components

---

## Final Recommendations for Commercial Frontend

### ✅ DO THIS
1. **Create 9 pages total** (2 critical + 4 high + 3 medium)
2. **Consolidate contact flows** into one `/contact` page with URL params
3. **Consolidate industries** into one `/industries` page with anchor links
4. **Update all `/docs/*` links** to point to external docs project
5. **Add section IDs** for anchor link navigation
6. **Make `/getting-started`** a marketing bridge, not technical docs
7. **Create `/story`** as hidden Easter egg (not in navigation)

### ❌ DON'T DO THIS
1. Don't create any `/docs/*` pages in commercial frontend
2. Don't create 4 separate industry pages
3. Don't create 4 separate contact pages
4. Don't create `/technical-deep-dive` (redirect to external docs)
5. Don't link `/story` in navigation (it's a hidden page)

### 🎯 Commercial Frontend Scope
**This project handles:**
- Marketing pages (homepage, features, pricing, use-cases, etc.)
- Lead generation (`/getting-started`, `/contact`)
- Trust building (`/about`, `/blog`, `/security`)
- Legal compliance (`/legal/*`)
- Industry marketing (`/industries`)

**Docs project handles:**
- Technical documentation
- API reference
- Installation guides
- Architecture deep-dives
- Troubleshooting

---

## Total Pages Count

**Existing:** 7 pages
- `/` (homepage)
- `/use-cases`
- `/pricing`
- `/gpu-providers`
- `/features`
- `/enterprise`
- `/developers`

**To Create:** 9 pages (MUST HAVE - per IMPLEMENTATION_PLAN.md)
- `/getting-started`
- `/contact`
- `/blog`
- `/about`
- `/legal/privacy`
- `/legal/terms`
- `/security`
- `/industries`
- `/story` (hidden page - not in navigation)

**Note:** `/join-waitlist` removed from plan - Navigation CTA updated to "Get Started" linking to `/getting-started`

**Optional (DEFERRED - Not in Current Plan):**
- `/for/homelabbers` - Homelab enthusiasts (unique pain point: idle GPU monetization)
- `/for/investors` - Pitch deck style (different content format)

**Final Total:** 16 commercial pages (9 must-have + 7 existing)
**With Optional:** 18 commercial pages

---

**Next Steps:**

**See IMPLEMENTATION_PLAN.md for detailed execution plan with:**
- Navigation & Footer update patterns
- Component reuse strategy (existing 133 organisms)
- Copy sources (90% already written in stakeholder docs)
- Day-by-day task breakdown
- Verification checklists

**Quick Start:**
1. Week 1: Update Navigation.tsx & Footer.tsx, then create `/getting-started` and `/contact`
2. Week 2: Create LegalPageLayout, then `/legal/privacy`, `/legal/terms`, and `/blog` placeholder
3. Week 3: Create `/about`, `/security`, `/industries`, and `/story`

**Key Decisions (from IMPLEMENTATION_PLAN.md):**
- Navigation CTA: "Join Waitlist" → "Get Started" (links to `/getting-started`)
- Contact flow: Single `/contact` page with URL params (consolidates 4+ variants)
- Industries: Single `/industries` page with anchor links (NOT 4 separate pages)
- Legal: Use privacy/terms generators, customize for rbee
- Copy: 90% already written in `.business/stakeholders/` docs

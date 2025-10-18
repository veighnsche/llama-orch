# Pages To Build - Content Checklist

**Last Updated:** October 17, 2025  
**Status:** 11 pages need full content (currently stub pages)

---

## 📋 Overview

All pages have been scaffolded with:
- ✅ SEO metadata (title, description, keywords, Open Graph, Twitter Cards)
- ✅ Responsive layout with container
- ✅ Dark mode support
- ✅ "Coming Soon" stub content
- ✅ Import structure from `@rbee/ui/pages`

**What's needed:** Replace stub content with full page designs using templates from `@rbee/ui/templates`.

---

## 🏗️ Industry Pages (6 pages)

### 1. Startups Page
**Path:** `/industries/startups`  
**Component:** `StartupsPage` in `@rbee/ui/pages`  
**Target Audience:** Startup founders and entrepreneurs

**Content Needed:**
- [ ] Hero section with value proposition
- [ ] Business opportunity overview (Year 1: 35 customers, €70K revenue)
- [ ] Marketplace model explanation
- [ ] Revenue projections and growth path
- [ ] Platform mode for scaling
- [ ] Independence from external AI providers
- [ ] Pricing calculator or estimator
- [ ] Case study or testimonial (if available)
- [ ] CTA: Join Waitlist

**Templates to Use:**
- `IndustryHero` or `HomeHero`
- `ProblemTemplate`
- `SolutionTemplate`
- `PricingTemplate` or custom pricing calculator
- `CTATemplate`

---

### 2. Homelab Page
**Path:** `/industries/homelab`  
**Component:** `HomelabPage` in `@rbee/ui/pages`  
**Target Audience:** Homelab enthusiasts, self-hosters, privacy advocates

**Content Needed:**
- [ ] Hero section emphasizing self-hosting and privacy
- [ ] SSH-based control explanation
- [ ] Multi-backend support (CUDA, Metal, CPU)
- [ ] Web UI + CLI tools overview
- [ ] Model catalog with auto-download
- [ ] Complete control and privacy benefits
- [ ] Setup guide or quick start
- [ ] Hardware requirements
- [ ] CTA: Get Started / Join Community

**Templates to Use:**
- `IndustryHero`
- `TechnicalTemplate`
- `HowItWorks`
- `FeaturesTemplate` or `FeaturesTabs`
- `CTATemplate`

---

### 3. Research Page
**Path:** `/industries/research`  
**Component:** `ResearchPage` in `@rbee/ui/pages`  
**Target Audience:** AI researchers, ML engineers

**Content Needed:**
- [ ] Hero section for research use cases
- [ ] Multi-modal support (LLMs, Stable Diffusion, TTS, embeddings)
- [ ] Proof bundles for reproducibility
- [ ] Determinism suite for regression testing
- [ ] BDD-tested with executable specs
- [ ] Candle-powered (Rust ML framework) benefits
- [ ] Research workflow examples
- [ ] Academic use cases
- [ ] CTA: Explore Docs / Join Waitlist

**Templates to Use:**
- `IndustryHero`
- `TechnicalTemplate`
- `FeaturesTabs`
- `UseCasesTemplate`
- `CTATemplate`

---

### 4. Compliance Page
**Path:** `/industries/compliance`  
**Component:** `CompliancePage` in `@rbee/ui/pages`  
**Target Audience:** Compliance officers, EU businesses, regulated industries

**Content Needed:**
- [ ] Hero section emphasizing GDPR compliance
- [ ] GDPR-compliant by design features
- [ ] Immutable audit logging (7-year retention)
- [ ] EU data residency controls
- [ ] SOC2 and ISO 27001 alignment
- [ ] Tamper-evident logs with blockchain-style hash chains
- [ ] Compliance checklist or framework
- [ ] Security architecture overview
- [ ] CTA: Request Compliance Demo

**Templates to Use:**
- `IndustryHero`
- `SecurityTemplate` (if exists) or `TechnicalTemplate`
- `FeaturesTabs`
- `ComparisonTemplate` (vs. US cloud providers)
- `CTATemplate`

---

### 5. Education Page
**Path:** `/industries/education`  
**Component:** `EducationPage` in `@rbee/ui/pages`  
**Target Audience:** CS students, educators, universities

**Content Needed:**
- [ ] Hero section for educational use
- [ ] Nature-inspired beehive architecture explanation
- [ ] Open source (GPL-3.0) benefits
- [ ] BDD-tested with Gherkin scenarios
- [ ] Rust + Candle for ML learning
- [ ] Smart/dumb architecture patterns
- [ ] Learning resources and tutorials
- [ ] Course integration ideas
- [ ] CTA: Explore Docs / Get Started

**Templates to Use:**
- `IndustryHero`
- `WhatIsRbee` (adapted for education)
- `TechnicalTemplate`
- `HowItWorks`
- `CTATemplate`

---

### 6. DevOps Page
**Path:** `/industries/devops`  
**Component:** `DevOpsPage` in `@rbee/ui/pages`  
**Target Audience:** DevOps engineers, SREs, infrastructure teams

**Content Needed:**
- [ ] Hero section for production deployment
- [ ] Cascading shutdown (prevents orphaned processes)
- [ ] Health monitoring (30-second heartbeats)
- [ ] Multi-node SSH control
- [ ] Lifecycle management (daemon, hive, worker control)
- [ ] Proof bundles for debugging
- [ ] Deployment workflow
- [ ] Monitoring and observability
- [ ] CTA: View Docs / Join Waitlist

**Templates to Use:**
- `IndustryHero`
- `TechnicalTemplate`
- `HowItWorks`
- `FeaturesTabs`
- `CTATemplate`

---

## 🌐 Resource Pages (5 pages)

### 7. Community Page
**Path:** `/community`  
**Component:** `CommunityPage` in `@rbee/ui/pages`  
**Target Audience:** Developers, contributors, users

**Content Needed:**
- [ ] Hero section for community
- [ ] GitHub Discussions link and overview
- [ ] Discord server (when available)
- [ ] GitHub Issues for bug reports
- [ ] Contributing guidelines
- [ ] Code of conduct
- [ ] Community stats (if available)
- [ ] Featured contributors
- [ ] CTA: Join Discord / Start Contributing

**Templates to Use:**
- Custom hero
- Card grid for community channels
- `CTATemplate`
- Links to external resources

---

### 8. Security Page
**Path:** `/security`  
**Component:** `SecurityPage` in `@rbee/ui/pages`  
**Target Audience:** Security teams, compliance officers, enterprises

**Content Needed:**
- [ ] Hero section for security overview
- [ ] Audit logging features (7-year retention)
- [ ] Tamper detection (blockchain-style hash chains)
- [ ] Data residency (EU-only)
- [ ] Compliance (GDPR, SOC2, ISO 27001)
- [ ] Security architecture (6 dedicated security crates)
- [ ] Vulnerability reporting process
- [ ] Security best practices
- [ ] CTA: View Security Docs / Contact Security Team

**Templates to Use:**
- Custom hero
- `TechnicalTemplate`
- `FeaturesTabs`
- Security-focused content blocks

---

### 9. Legal Hub Page
**Path:** `/legal`  
**Component:** `LegalPage` in `@rbee/ui/pages`  
**Target Audience:** All users

**Content Needed:**
- [ ] Hero section for legal information
- [ ] Links to Privacy Policy
- [ ] Links to Terms of Service
- [ ] GPL-3.0 license overview
- [ ] Contact information for legal inquiries
- [ ] Last updated dates

**Templates to Use:**
- Simple card grid (already implemented)
- May need minimal updates

**Status:** ⚠️ Partially complete (has card grid, may need minor enhancements)

---

### 10. Privacy Policy Page
**Path:** `/legal/privacy`  
**Component:** `PrivacyPage` in `@rbee/ui/pages`  
**Target Audience:** All users

**Content Needed:**
- [ ] **CRITICAL:** Full GDPR-compliant privacy policy
- [ ] Data collection and usage
- [ ] Cookie policy
- [ ] User rights (access, deletion, portability)
- [ ] Data retention policies
- [ ] Third-party services
- [ ] EU data residency
- [ ] Contact information for privacy inquiries
- [ ] Last updated date

**Templates to Use:**
- Prose/markdown content with proper typography
- Legal document formatting

**Status:** 🚨 **HIGH PRIORITY** - Required for GDPR compliance

---

### 11. Terms of Service Page
**Path:** `/legal/terms`  
**Component:** `TermsPage` in `@rbee/ui/pages`  
**Target Audience:** All users

**Content Needed:**
- [ ] **CRITICAL:** Full terms of service
- [ ] Acceptable use policy
- [ ] Service availability and support
- [ ] Intellectual property rights
- [ ] Liability disclaimers
- [ ] Termination conditions
- [ ] Dispute resolution
- [ ] GPL-3.0 license reference
- [ ] Last updated date

**Templates to Use:**
- Prose/markdown content with proper typography
- Legal document formatting

**Status:** 🚨 **HIGH PRIORITY** - Required for legal compliance

---

## 📊 Priority Matrix

### **P0 - Critical (Legal Compliance)**
1. 🚨 Privacy Policy (`/legal/privacy`)
2. 🚨 Terms of Service (`/legal/terms`)

**Action:** Consult with legal counsel before publishing

---

### **P1 - High Priority (Core Audiences)**
3. Developers (already complete via `/developers`)
4. Enterprise (already complete via `/enterprise`)
5. Providers (already complete via `/gpu-providers`)
6. Startups (`/industries/startups`)
7. Homelab (`/industries/homelab`)

---

### **P2 - Medium Priority (Secondary Audiences)**
8. Research (`/industries/research`)
9. DevOps (`/industries/devops`)
10. Compliance (`/industries/compliance`)

---

### **P3 - Lower Priority (Supporting Pages)**
11. Education (`/industries/education`)
12. Community (`/community`)
13. Security (`/security`)

---

## 🎨 Design Pattern

All pages should follow this structure:

```tsx
// Example: StartupsPage.tsx
'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  IndustryHero,
  ProblemTemplate,
  SolutionTemplate,
  CTATemplate,
} from '@rbee/ui/templates'
import {
  startupsHeroProps,
  startupsProblemProps,
  startupsSolutionProps,
  startupsCtaProps,
} from './StartupsPageProps'

export default function StartupsPage() {
  return (
    <>
      <TemplateContainer {...startupsHeroContainerProps}>
        <IndustryHero {...startupsHeroProps} />
      </TemplateContainer>
      
      <TemplateContainer {...startupsProblemContainerProps}>
        <ProblemTemplate {...startupsProblemProps} />
      </TemplateContainer>
      
      <TemplateContainer {...startupsSolutionContainerProps}>
        <SolutionTemplate {...startupsSolutionProps} />
      </TemplateContainer>
      
      <TemplateContainer {...startupsCtaContainerProps}>
        <CTATemplate {...startupsCtaProps} />
      </TemplateContainer>
    </>
  )
}
```

---

## 📝 Content Guidelines

### **Copy Style**
- **Crisp and declarative:** 7-10 words per description
- **Action-oriented:** Focus on what users can do
- **Avoid jargon:** Unless audience-specific (e.g., DevOps, Compliance)
- **Value-first:** Lead with benefits, not features

### **SEO**
- All metadata already in place (title, description, keywords)
- Ensure content matches metadata promises
- Use semantic HTML (h1, h2, h3 hierarchy)
- Include relevant keywords naturally

### **Accessibility**
- Proper heading hierarchy
- Alt text for all images
- ARIA labels where needed
- Color contrast WCAG AA minimum

---

## 🚀 Next Steps

1. **Legal Pages (P0):**
   - Draft Privacy Policy
   - Draft Terms of Service
   - Review with legal counsel
   - Publish

2. **Core Industry Pages (P1):**
   - Create content for Startups page
   - Create content for Homelab page
   - Design templates if needed

3. **Secondary Pages (P2-P3):**
   - Research, DevOps, Compliance
   - Education, Community, Security

4. **Review & Polish:**
   - Consistency check across all pages
   - SEO audit
   - Accessibility audit
   - Performance optimization

---

## 📚 Resources

- **Existing Templates:** `/frontend/packages/rbee-ui/src/templates/`
- **Existing Pages (Reference):** `/frontend/packages/rbee-ui/src/pages/HomePage/`, `FeaturesPage/`, etc.
- **Design System:** `/frontend/packages/rbee-ui/src/atoms/`, `/molecules/`
- **Navigation Plan:** `/frontend/packages/rbee-ui/NAVIGATION_REDESIGN_PLAN.md`

---

**Total Pages:** 11 stub pages  
**Completed:** 0 of 11  
**In Progress:** 0  
**Blocked:** 2 (Privacy Policy, Terms of Service - need legal review)

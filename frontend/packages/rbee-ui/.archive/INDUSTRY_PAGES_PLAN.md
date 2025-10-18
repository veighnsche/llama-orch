# Industry Pages Implementation Plan

**Date:** October 17, 2025  
**Source:** `.business/stakeholders/VIDEO_SCRIPTS.md`  
**Status:** Planning Phase

---

## Overview

Based on the 10 target audiences identified in VIDEO_SCRIPTS.md, we need to create **6 new industry-specific landing pages** plus redirects for 2 existing pages.

**Existing Pages (Redirects Only):**
- `/industries/enterprise` → `/enterprise` (already exists)
- `/industries/developers` → `/developers` (already exists)

**New Pages to Create:**
1. `/industries/startups` - Startup Founders / Entrepreneurs
2. `/industries/homelab` - Homelab Enthusiasts / Self-Hosters
3. `/industries/research` - AI Researchers / ML Engineers
4. `/industries/compliance` - EU Businesses / Compliance Officers
5. `/industries/education` - Computer Science Students / Educators
6. `/industries/devops` - DevOps Engineers / SREs

**Note:** Investors and AI Development Community audiences are not getting dedicated pages (covered in other content).

---

## Page Specifications

### 1. `/industries/startups` - Startup Founders / Entrepreneurs

**Target Audience:** Startup founders, entrepreneurs, business leaders

**Key Messages:**
- Business opportunity and marketplace model
- Platform mode for scaling
- Revenue projections: Year 1 (35 customers, €70K), Year 2 (100 customers, €360K)
- Independence from external AI providers

**Content Sections:**
1. **Hero:** "Build Your AI Infrastructure. Escape Provider Dependency."
2. **Problem:** Complex codebases with AI assistance become unmaintainable if provider changes
3. **Solution:** Build AI coders from scratch using home network hardware
4. **Business Model:** Agentic API with task-based streaming
5. **Revenue Potential:** Conservative projections with growth path
6. **Features:** OpenAI-compatible, multi-modal, GPL-3.0
7. **CTA:** "Start Building Your AI Independence"

**SEO Keywords:**
- AI startup infrastructure
- OpenAI alternative for startups
- self-hosted AI business
- AI provider independence
- startup AI platform

---

### 2. `/industries/homelab` - Homelab Enthusiasts / Self-Hosters

**Target Audience:** Homelab enthusiasts, self-hosters, privacy advocates

**Key Messages:**
- Complete control and privacy
- Use ALL your home network hardware
- SSH-based control (homelab-friendly)
- Zero external dependencies

**Content Sections:**
1. **Hero:** "Your Hardware. Your AI. Your Rules."
2. **Problem:** Dependence on cloud AI providers for personal projects
3. **Solution:** Self-hosted AI infrastructure across all your computers
4. **Features:**
   - SSH-based control
   - Multi-backend: CUDA, Metal, CPU
   - Web UI + CLI
   - Model catalog with auto-download
   - Idle timeout frees VRAM
   - Cascading shutdown
5. **Use Cases:** Personal AI coders, homelab projects, learning
6. **CTA:** "Take Control of Your AI"

**SEO Keywords:**
- homelab AI
- self-hosted LLM
- privacy-first AI
- local AI infrastructure
- SSH AI control

---

### 3. `/industries/research` - AI Researchers / ML Engineers

**Target Audience:** AI researchers, ML engineers, academics

**Key Messages:**
- Multi-modal support (LLMs, SD, TTS, embeddings)
- Reproducibility with proof bundles
- Research-grade quality with production infrastructure
- BDD-tested for reliability

**Content Sections:**
1. **Hero:** "Research-Grade AI Infrastructure. Production-Ready."
2. **Multi-Modal Platform:** Unified API for LLMs, Stable Diffusion, TTS, embeddings
3. **Reproducibility:**
   - Proof bundles: same seed → same output
   - Determinism suite for regression testing
   - Property tests for invariants
4. **Technical Stack:**
   - Candle-powered (Rust ML framework)
   - Backend auto-detection (CUDA, Metal, CPU)
   - User-scriptable routing via Rhai
5. **Research Features:**
   - Test reproducibility for CI/CD
   - BDD-tested with executable specs
   - Observable via narration events
6. **CTA:** "Build Reproducible AI Research"

**SEO Keywords:**
- AI research infrastructure
- reproducible ML
- multi-modal AI platform
- research AI tools
- ML experiment tracking

---

### 4. `/industries/compliance` - EU Businesses / Compliance Officers

**Target Audience:** EU businesses, compliance officers, legal teams, regulated industries

**Key Messages:**
- GDPR-compliant by design
- EU data residency
- 7-year audit retention
- Zero US cloud dependencies

**Content Sections:**
1. **Hero:** "EU-Native AI. GDPR-Compliant by Design."
2. **Compliance Features:**
   - Immutable audit logging (7-year retention)
   - Data residency controls
   - Consent tracking
   - Right to erasure
3. **EU-Only Infrastructure:**
   - EU-only worker filtering via geo-verification
   - No US cloud dependencies
   - Self-hosted for maximum control
4. **Standards Alignment:**
   - SOC2 ready
   - ISO 27001 aligned
   - Tamper-evident logs with blockchain-style hash chains
5. **Built-in Compliance Endpoints:**
   - Data export
   - Data deletion
   - Consent management
   - Audit trail access
6. **CTA:** "Ensure AI Compliance"

**SEO Keywords:**
- GDPR AI infrastructure
- EU data residency AI
- compliant AI platform
- SOC2 AI
- ISO 27001 AI

---

### 5. `/industries/education` - Computer Science Students / Educators

**Target Audience:** CS students, educators, professors, bootcamp instructors

**Key Messages:**
- Learn distributed systems from nature-inspired architecture
- Open source (GPL-3.0) - study real production code
- BDD-tested with Gherkin scenarios
- Real production system, not a toy

**Content Sections:**
1. **Hero:** "Learn Distributed AI from Nature-Inspired Architecture."
2. **Architecture:**
   - 4 components mirror a beehive
   - Queen (orchestrator brain)
   - Hive (pool manager)
   - Workers (inference executors)
   - Keeper (CLI interface)
3. **Learning Opportunities:**
   - Study real production code
   - BDD-tested with Gherkin scenarios
   - Rust + Candle for ML
   - Multi-backend (CUDA, Metal, CPU)
4. **Smart/Dumb Architecture:**
   - Intelligence at the edge
   - Execution at the workers
5. **Open Source:**
   - GPL-3.0 license
   - Study, contribute, learn
6. **CTA:** "Start Learning Distributed AI"

**SEO Keywords:**
- learn distributed systems
- AI architecture education
- open source AI platform
- CS education AI
- distributed AI tutorial

---

### 6. `/industries/devops` - DevOps Engineers / SREs

**Target Audience:** DevOps engineers, SREs, infrastructure teams

**Key Messages:**
- Production-ready orchestration
- Lifecycle management
- Monitoring and observability
- Multi-node deployment

**Content Sections:**
1. **Hero:** "Production-Ready AI Orchestration."
2. **Lifecycle Management:**
   - Cascading shutdown (prevents orphaned processes)
   - Health monitoring (30-second heartbeats)
   - Idle timeout (5min) automatically frees VRAM
3. **Deployment:**
   - Multi-node SSH control
   - Backend auto-detection (CUDA, Metal, CPU)
   - Model catalog with auto-download and caching
4. **Operations:**
   - Daemon start/stop
   - Hive start/stop
   - Worker start/stop
5. **Observability:**
   - Proof bundles for debugging (NDJSON logs, seeds, metadata)
   - Immutable audit logs for forensics
   - Observable via narration events
6. **CTA:** "Deploy Production AI"

**SEO Keywords:**
- AI DevOps
- production AI infrastructure
- AI orchestration platform
- AI monitoring
- distributed AI deployment

---

## Shared Template: IndustryTemplate

All 6 industry pages will use a shared `IndustryTemplate` component with customizable props.

### IndustryTemplate Props Interface

```tsx
export interface IndustryTemplateProps {
  // Hero Section
  hero: {
    title: string
    subtitle: string
    description: string
    cta: {
      primary: { label: string; href: string }
      secondary?: { label: string; href: string }
    }
    background?: ReactNode
  }

  // Problem Section (optional)
  problem?: {
    title: string
    description: string
    painPoints: string[]
  }

  // Solution Section
  solution: {
    title: string
    description: string
    features: {
      icon: ReactNode
      title: string
      description: string
    }[]
  }

  // Highlights Section (key differentiators)
  highlights: {
    title: string
    items: {
      metric?: string  // e.g., "€70K" or "7-year"
      label: string
      description: string
    }[]
  }

  // Use Cases Section (optional)
  useCases?: {
    title: string
    cases: {
      title: string
      description: string
      outcome: string
    }[]
  }

  // Technical Details Section (optional)
  technical?: {
    title: string
    features: string[]
  }

  // CTA Section
  cta: {
    title: string
    description: string
    primary: { label: string; href: string }
    secondary?: { label: string; href: string }
  }

  // Metadata
  metadata: {
    title: string
    description: string
    keywords: string[]
    canonical: string
  }
}
```

---

## Implementation Checklist

### Phase 1: Template Creation (2 days)
- [ ] Create `IndustryTemplate` component
- [ ] Create Storybook story with example props
- [ ] Test responsive behavior
- [ ] Verify accessibility

### Phase 2: Content Creation (3 days)
- [ ] Write content for `/industries/startups`
- [ ] Write content for `/industries/homelab`
- [ ] Write content for `/industries/research`
- [ ] Write content for `/industries/compliance`
- [ ] Write content for `/industries/education`
- [ ] Write content for `/industries/devops`

### Phase 3: Page Implementation (2 days)
- [ ] Create Next.js pages for all 6 industries
- [ ] Add metadata to each page
- [ ] Create props files for each industry
- [ ] Add redirects for `/industries/enterprise` and `/industries/developers`

### Phase 4: SEO Optimization (1 day)
- [ ] Add structured data (Industry schema)
- [ ] Optimize meta descriptions
- [ ] Add Open Graph images
- [ ] Test with Google Rich Results

---

## SEO Impact Estimate

**New Pages:** 6 industry-specific landing pages  
**Target Keywords:** 30+ industry-specific long-tail keywords  
**Estimated Traffic Boost:** +500-800 monthly visits from industry-specific searches

**Keyword Examples:**
- "AI infrastructure for startups"
- "homelab AI setup"
- "GDPR compliant AI platform"
- "AI research infrastructure"
- "DevOps AI orchestration"
- "AI education platform"

---

## Next Steps

1. ✅ Review and approve this plan
2. Create `IndustryTemplate` component
3. Write content for each industry page (based on VIDEO_SCRIPTS.md)
4. Implement pages in Next.js
5. Add SEO metadata and structured data
6. Test and deploy

---

**Total Effort:** 5-7 days  
**Priority:** HIGH (Phase 3 of SEO audit)

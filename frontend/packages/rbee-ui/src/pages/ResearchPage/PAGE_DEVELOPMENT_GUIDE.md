# ResearchPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/research` (Research & Academia)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Research page showcasing rbee for academic research, reproducible experiments, and scientific computing.

**Target Audience:** Researchers, academics, data scientists, PhD students

**Key Message:** Reproducible runs with deterministic seeds, experiment tracking, and multi-modal support.

---

## 📋 Before You Start

### 1. Read Required Documentation

- ✅ **TEMPLATE_CATALOG.md** - Complete template inventory with reusability analysis
- ✅ **CONSOLIDATION_OPPORTUNITIES.md** - Template consolidation guidelines
- ✅ **CHECKLIST.md** (this directory) - Content requirements

### 2. Understand the Reusability Philosophy

**🚨 CRITICAL:** Template names are **marketing labels**, not technical constraints.

**Your job:**
1. Review TEMPLATE_CATALOG.md with a **reusability mindset**
2. Try adapting existing templates before creating new ones
3. Only propose new templates if existing ones truly cannot be adapted

**Examples of creative reuse for Research:**
- `ProvidersEarnings` → Experiment cost calculator (compute hours × cost)
- `EnterpriseCompliance` → Research methodologies (Quantitative/Qualitative/Mixed)
- `EnterpriseSecurity` → Research tools showcase (6 tool cards with bullets)
- `HomeHero` → Experiment runner demo (terminal with live output)
- `EnterpriseHero` → Experiment logs visualization (console with events)

---

## 🔄 Template Reusability Matrix for Research

### Hero Section
**Need:** Hero with reproducibility message

**Options:**
1. ✅ **HeroTemplate** (base) - Most flexible, accepts custom aside
2. ✅ **HomeHero** - Adapt terminal demo → experiment runner
3. ✅ **EnterpriseHero** - Adapt audit console → experiment logs
4. ❌ Create new template - Only if none of above work

**Recommendation:** Try `HeroTemplate` with custom aside showing experiment reproducibility visualization.

### Reproducibility Section
**Need:** Show proof bundles, deterministic seeds, experiment tracking

**Options:**
1. ✅ **SolutionTemplate** - Feature cards + optional aside
2. ✅ **EnterpriseSecurity** - Grid of detailed feature cards
3. ✅ **AdditionalFeaturesGrid** - Simple feature grid

**Recommendation:** `SolutionTemplate` with 4 feature cards (proof bundles, seeds, tracking, version control) + aside with topology diagram.

### Multi-Modal Support
**Need:** Show LLMs, Stable Diffusion, TTS, Embeddings

**Options:**
1. ✅ **FeaturesTabs** - Tabbed showcase (perfect for modalities)
2. ✅ **AdditionalFeaturesGrid** - Grid of modality cards
3. ✅ **CardGridTemplate** - Generic grid

**Recommendation:** `FeaturesTabs` with tabs for each modality (Text, Image, Audio, Embeddings).

### Technical Architecture
**Need:** BDD-tested, Candle-powered, performance benefits

**Options:**
1. ✅ **TechnicalTemplate** - Architecture + tech stack (PERFECT FIT)
2. ✅ **EnterpriseSecurity** - Grid of technical features

**Recommendation:** `TechnicalTemplate` - already has architecture highlights, coverage progress, diagram, and tech stack.

### Research Workflow
**Need:** Step-by-step experiment workflow

**Options:**
1. ✅ **HowItWorks** - Step-by-step guide with code/terminal blocks (PERFECT FIT)
2. ✅ **EnterpriseHowItWorks** - Process with timeline

**Recommendation:** `HowItWorks` - shows numbered steps with code examples.

### Determinism Suite
**Need:** Regression testing, seed management, result verification

**Options:**
1. ✅ **SolutionTemplate** - Feature cards
2. ✅ **AdditionalFeaturesGrid** - Feature grid
3. ✅ **EnterpriseSecurity** - Detailed feature cards

**Recommendation:** `AdditionalFeaturesGrid` with 4 cards (regression, seeds, verification, debugging).

### Academic Use Cases
**Need:** Research papers, thesis work, collaborative projects, teaching

**Options:**
1. ✅ **UseCasesTemplate** - Scenario/solution/outcome cards (PERFECT FIT)
2. ✅ **EnterpriseUseCases** - Industry cards with challenges/solutions

**Recommendation:** `UseCasesTemplate` with 4-6 academic scenarios.

### Cost Calculator
**Need:** Experiment cost estimator

**Options:**
1. ✅ **ProvidersEarnings** - Interactive calculator (PERFECT FIT - just change the labels!)
2. ❌ Create new calculator

**Recommendation:** `ProvidersEarnings` adapted:
- GPU selection → Model selection
- Availability slider → Experiment duration slider
- Earnings projection → Cost projection
- Monthly/yearly toggle → Per-experiment/per-month toggle

### CTA Section
**Need:** Access documentation, join research community

**Options:**
1. ✅ **CTATemplate** - Universal CTA (use on every page)
2. ✅ **EnterpriseCTA** - Multi-option CTA (Demo/Docs/Contact)
3. ✅ **EmailCapture** - Lead capture form

**Recommendation:** Use both `EmailCapture` (mid-page) and `CTATemplate` (end of page).

---

## 📐 Proposed Page Structure

Based on reusability analysis, here's the recommended structure:

```tsx
<ResearchPage>
  {/* Hero */}
  <TemplateContainer {...heroContainerProps}>
    <HeroTemplate {...heroProps} />
  </TemplateContainer>

  {/* Email Capture */}
  <TemplateContainer {...emailCaptureContainerProps}>
    <EmailCapture {...emailCaptureProps} />
  </TemplateContainer>

  {/* Problem: Research Challenges */}
  <TemplateContainer {...problemContainerProps}>
    <ProblemTemplate {...problemProps} />
  </TemplateContainer>

  {/* Solution: Reproducibility Features */}
  <TemplateContainer {...solutionContainerProps}>
    <SolutionTemplate {...solutionProps} />
  </TemplateContainer>

  {/* Multi-Modal Support */}
  <TemplateContainer {...multiModalContainerProps}>
    <FeaturesTabs {...multiModalProps} />
  </TemplateContainer>

  {/* Research Workflow */}
  <TemplateContainer {...workflowContainerProps}>
    <HowItWorks {...workflowProps} />
  </TemplateContainer>

  {/* Determinism Suite */}
  <TemplateContainer {...determinismContainerProps}>
    <AdditionalFeaturesGrid {...determinismProps} />
  </TemplateContainer>

  {/* Cost Calculator */}
  <TemplateContainer {...calculatorContainerProps}>
    <ProvidersEarnings {...calculatorProps} />
  </TemplateContainer>

  {/* Academic Use Cases */}
  <TemplateContainer {...useCasesContainerProps}>
    <UseCasesTemplate {...useCasesProps} />
  </TemplateContainer>

  {/* Technical Deep-Dive */}
  <TemplateContainer {...technicalContainerProps}>
    <TechnicalTemplate {...technicalProps} />
  </TemplateContainer>

  {/* FAQ */}
  <TemplateContainer {...faqContainerProps}>
    <FAQTemplate {...faqProps} />
  </TemplateContainer>

  {/* Final CTA */}
  <CTATemplate {...ctaProps} />
</ResearchPage>
```

**Total templates used:** 10 (all existing, zero new templates needed!)

---

## ✅ Implementation Checklist

### Phase 1: Setup (30 min)
- [ ] Read TEMPLATE_CATALOG.md completely
- [ ] Read CONSOLIDATION_OPPORTUNITIES.md
- [ ] Review existing page props files (HomePage, EnterprisePage, ProvidersPage)
- [ ] Create `ResearchPageProps.tsx` file

### Phase 2: Props Definition (2-3 hours)
- [ ] Define `heroContainerProps` and `heroProps`
- [ ] Define `emailCaptureContainerProps` and `emailCaptureProps`
- [ ] Define `problemContainerProps` and `problemProps`
- [ ] Define `solutionContainerProps` and `solutionProps`
- [ ] Define `multiModalContainerProps` and `multiModalProps`
- [ ] Define `workflowContainerProps` and `workflowProps`
- [ ] Define `determinismContainerProps` and `determinismProps`
- [ ] Define `calculatorContainerProps` and `calculatorProps` (adapt ProvidersEarnings)
- [ ] Define `useCasesContainerProps` and `useCasesProps`
- [ ] Define `technicalContainerProps` and `technicalProps`
- [ ] Define `faqContainerProps` and `faqProps`
- [ ] Define `ctaProps`

### Phase 3: Page Component (1 hour)
- [ ] Create `ResearchPage.tsx` component
- [ ] Import all templates
- [ ] Import all props from `ResearchPageProps.tsx`
- [ ] Compose page with TemplateContainer wrappers
- [ ] Add proper TypeScript types

### Phase 4: Content Writing (2-3 hours)
- [ ] Write hero headline and subcopy
- [ ] Write problem cards (reproducibility issues, collaboration challenges, etc.)
- [ ] Write solution feature cards
- [ ] Write multi-modal tabs content
- [ ] Write workflow steps
- [ ] Write determinism suite cards
- [ ] Adapt calculator labels for research context
- [ ] Write academic use case scenarios
- [ ] Write technical architecture content
- [ ] Write FAQ questions and answers
- [ ] Write final CTA copy

### Phase 5: Testing (1 hour)
- [ ] Test in Storybook
- [ ] Test responsive layout (mobile, tablet, desktop)
- [ ] Test dark mode
- [ ] Test all interactive elements (tabs, accordion, calculator)
- [ ] Verify accessibility (ARIA labels, keyboard navigation)

### Phase 6: Documentation (30 min)
- [ ] Update CHECKLIST.md with completion status
- [ ] Document any template adaptations made
- [ ] Note any issues or improvements needed

---

## 🚫 When to Propose a New Template

**Only propose a new template if:**

1. ✅ You've tried adapting at least 3 existing templates
2. ✅ You can explain why each existing template won't work
3. ✅ The new template would be reusable for other pages
4. ✅ You've written a proposal document with:
   - Problem statement
   - Why existing templates don't work
   - Proposed template API
   - Reusability analysis for other pages

**Template proposal location:** `src/templates/[TemplateName]/PROPOSAL.md`

---

## 📚 Reference Examples

### Good Examples (Reuse Existing)
- ✅ HomePage uses `ProvidersEarnings` → Research can too (just change labels)
- ✅ EnterprisePage uses `EnterpriseCompliance` → Research can adapt for methodologies
- ✅ ProvidersPage uses `SolutionTemplate` → Research can use same pattern

### Bad Examples (Don't Do This)
- ❌ Creating `ResearchHero` when `HeroTemplate` works
- ❌ Creating `ResearchCalculator` when `ProvidersEarnings` works
- ❌ Creating `ResearchFeatures` when `SolutionTemplate` works

---

## 🎨 Design Consistency

Follow these patterns from existing pages:

1. **Background decorations:** Use SVG backgrounds from `src/atoms` (NetworkMesh, OrchestrationFlow, etc.)
2. **Spacing:** Use TemplateContainer `paddingY` prop (xl, 2xl)
3. **Max-width:** Use TemplateContainer `maxWidth` prop (5xl, 6xl, 7xl)
4. **Icons:** Use lucide-react icons consistently
5. **Colors:** Use design tokens, support light/dark modes

---

## 🤝 Need Help?

1. **Template confusion?** Re-read TEMPLATE_CATALOG.md reusability sections
2. **Props unclear?** Look at existing page props files (HomePage, EnterprisePage)
3. **Still stuck?** Write a proposal document and ask for review

---

## 📊 Success Criteria

- ✅ Page uses 100% existing templates (no new templates created)
- ✅ All content requirements from CHECKLIST.md met
- ✅ Props file follows existing patterns
- ✅ Page component is clean and readable
- ✅ Responsive and accessible
- ✅ Works in light and dark modes

---

**Remember:** Speed comes from reuse, not from creating new components. Think creatively about how to adapt existing templates!

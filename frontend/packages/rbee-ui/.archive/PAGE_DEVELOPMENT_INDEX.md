# Page Development Index

**Version:** 1.0  
**Last Updated:** Oct 17, 2025  
**Total Pages:** 10

This document tracks all page development assignments and status.

---

## 📚 Required Reading for ALL Developers

Before starting ANY page, read these documents in order:

1. ✅ **TEMPLATE_CATALOG.md** - Complete template inventory with reusability analysis
2. ✅ **CONSOLIDATION_OPPORTUNITIES.md** - Template consolidation guidelines
3. ✅ **NEW_BACKGROUNDS_PLAN.md** - Background decoration system
4. ✅ **NAVIGATION_REDESIGN_PLAN.md** - Navigation structure

---

## 🎯 Core Philosophy

**REUSE, DON'T CREATE**

- Template names are **marketing labels**, not technical constraints
- Every template can be adapted for different contexts
- Only create new templates if existing ones truly cannot be adapted
- Write a proposal document before creating any new template

---

## 📄 Page Assignments

### Industry Pages (High Priority)

| Page | Route | Developer | Status | Guide |
|------|-------|-----------|--------|-------|
| **Research** | `/research` | TBD | 🔴 Not Started | `src/pages/ResearchPage/PAGE_DEVELOPMENT_GUIDE.md` |
| **Homelab** | `/homelab` | Developer 2 | ✅ Complete | `src/pages/HomelabPage/PAGE_DEVELOPMENT_GUIDE.md` |
| **Startups** | `/startups` | TBD | 🔴 Not Started | `src/pages/StartupsPage/PAGE_DEVELOPMENT_GUIDE.md` |
| **Education** | `/education` | TBD | 🔴 Not Started | `src/pages/EducationPage/PAGE_DEVELOPMENT_GUIDE.md` |
| **DevOps** | `/devops` | TBD | 🔴 Not Started | `src/pages/DevOpsPage/PAGE_DEVELOPMENT_GUIDE.md` |
| **Compliance** | `/compliance` | TBD | 🔴 Not Started | `src/pages/CompliancePage/PAGE_DEVELOPMENT_GUIDE.md` |

### Support Pages (Medium Priority)

| Page | Route | Developer | Status | Guide |
|------|-------|-----------|--------|-------|
| **Community** | `/community` | TBD | 🔴 Not Started | `src/pages/CommunityPage/PAGE_DEVELOPMENT_GUIDE.md` |
| **Security** | `/security` | TBD | 🔴 Not Started | `src/pages/SecurityPage/PAGE_DEVELOPMENT_GUIDE.md` |

### Legal Pages (Low Priority)

| Page | Route | Developer | Status | Guide |
|------|-------|-----------|--------|-------|
| **Privacy** | `/legal/privacy` | TBD | 🔴 Not Started | `src/pages/PrivacyPage/PAGE_DEVELOPMENT_GUIDE.md` |
| **Terms** | `/legal/terms` | TBD | 🔴 Not Started | `src/pages/TermsPage/PAGE_DEVELOPMENT_GUIDE.md` |

---

## 📊 Development Workflow

### For Each Page

1. **Claim Assignment**
   - Add your name to the "Developer" column above
   - Update status to 🟡 In Progress

2. **Read Documentation**
   - Read TEMPLATE_CATALOG.md completely
   - Read your page's PAGE_DEVELOPMENT_GUIDE.md
   - Review existing page props files (HomePage, EnterprisePage, ProvidersPage)

3. **Plan Template Reuse**
   - Identify which existing templates can be reused
   - Document any adaptations needed
   - Only propose new templates if absolutely necessary

4. **Implement**
   - Create `[PageName]PageProps.tsx` file
   - Define all container and template props
   - Create `[PageName]Page.tsx` component
   - Write content for all sections

5. **Test**
   - Test in Storybook
   - Test responsive layout (mobile, tablet, desktop)
   - Test dark mode
   - Test accessibility (ARIA labels, keyboard navigation)

6. **Document**
   - Update CHECKLIST.md with completion status
   - Document any template adaptations
   - Update status to ✅ Complete

---

## 🔄 Template Reusability Quick Reference

### Most Versatile Templates (Use These First)

1. **HeroTemplate** - Universal hero, accepts any aside content
2. **ProblemTemplate** - Any pain points (domain-agnostic)
3. **SolutionTemplate** - Any solution showcase (domain-agnostic)
4. **HowItWorks** - Any step-by-step guide
5. **UseCasesTemplate** - Any scenario-based cards
6. **FAQTemplate** - Any Q&A content
7. **CTATemplate** - Universal CTA (use on every page)
8. **EmailCapture** - Any lead capture

### Specialized But Adaptable

9. **ProvidersEarnings** - ANY calculator/estimator (earnings, costs, savings, time, etc.)
10. **EnterpriseCompliance** - ANY three-pillar showcase
11. **EnterpriseSecurity** - ANY grid of detailed features
12. **EnterpriseHowItWorks** - ANY process with timeline
13. **EnterpriseUseCases** - ANY industry/segment breakdown
14. **FeaturesTabs** - ANY tabbed content
15. **ComparisonTemplate** - ANY feature comparison matrix
16. **PricingTemplate** - ANY tiered options
17. **TestimonialsTemplate** - ANY social proof with stats

---

## 🚫 When to Propose a New Template

**Only if:**
1. ✅ You've tried adapting at least 3 existing templates
2. ✅ You can explain why each won't work
3. ✅ The new template would be reusable for other pages
4. ✅ You've written a proposal document

**Proposal format:**
```markdown
# [TemplateName] Proposal

## Problem Statement
What problem does this solve that existing templates can't?

## Why Existing Templates Don't Work
- Template A: Doesn't work because...
- Template B: Doesn't work because...
- Template C: Doesn't work because...

## Proposed API
```tsx
export type [TemplateName]Props = {
  // Props definition
}
```

## Reusability Analysis
How can other pages use this template?
- Page X: Use case...
- Page Y: Use case...
- Page Z: Use case...
```

**Save proposal as:** `src/templates/[TemplateName]/PROPOSAL.md`

---

## 📈 Progress Tracking

### Overall Progress

- **Total Pages:** 10
- **Completed:** 1 (10%)
- **In Progress:** 0 (0%)
- **Not Started:** 9 (90%)

### By Priority

**High Priority (Industry Pages):**
- Completed: 1/6 (17%)
  - ✅ HomelabPage (Developer 2)

**Medium Priority (Support Pages):**
- Completed: 0/2 (0%)

**Low Priority (Legal Pages):**
- Completed: 0/2 (0%)

---

## 🎨 Design Consistency Guidelines

### Background Decorations
- Use SVG backgrounds from `src/atoms` (NetworkMesh, OrchestrationFlow, etc.)
- Apply via TemplateContainer `backgroundDecoration` prop
- Use `opacity-25` and `blur-[0.5px]`
- Wrap in `<div className="absolute inset-0 opacity-25">` (NOT -z-10 directly)

### Spacing
- Use TemplateContainer `paddingY` prop (sm, md, lg, xl, 2xl)
- Don't mix manual spacing with component spacing

### Max-Width
- Use TemplateContainer `maxWidth` prop (3xl, 5xl, 6xl, 7xl)
- Consistent with existing pages

### Icons
- Use lucide-react icons consistently
- Same icon for same concept across pages

### Colors
- Use design tokens from `tokens.css`
- Support light and dark modes
- Test both themes

---

## 🤝 Getting Help

### Questions About Templates?
1. Re-read TEMPLATE_CATALOG.md reusability sections
2. Look at existing page props files (HomePage, EnterprisePage, ProvidersPage)
3. Check the template's Storybook stories

### Questions About Content?
1. Check the page's CHECKLIST.md
2. Look at similar sections on existing pages
3. Review the page's PAGE_DEVELOPMENT_GUIDE.md

### Still Stuck?
1. Write a clear question with:
   - What you're trying to achieve
   - What you've tried
   - Why it didn't work
2. Ask in team chat or create a discussion

---

## 📝 Notes

- **Estimated time per page:** 6-8 hours (setup, props, component, content, testing, docs)
- **Fastest pages:** Legal pages (2-3 hours each)
- **Slowest pages:** Industry pages with calculators (8-10 hours)
- **Reuse is faster than creation:** Adapting existing templates is 3-5x faster than creating new ones

---

**Remember:** The goal is speed through reuse. Think creatively about adapting existing templates. Only create new templates as a last resort.

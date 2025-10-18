# CommunityPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/community` (Community & Support)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Community page showcasing rbee's open-source community, contribution opportunities, and support channels.

**Target Audience:** Contributors, community members, open-source enthusiasts

**Key Message:** Join the community. Contribute. Get support. Build together.

---

## 🔄 Key Template Adaptations for Community

### Hero
- ✅ `HeroTemplate` with community stats (contributors, stars, PRs)

### Community Stats
- ✅ `TestimonialsTemplate` adapted - Community stats instead of testimonials
  - Stats: GitHub stars, contributors, PRs merged, Discord members

### Contribution Opportunities
- ✅ `UseCasesTemplate` adapted - Different contribution types (code, docs, testing, design)

### Getting Started
- ✅ `HowItWorks` - How to contribute (fork → code → PR → review)

### Support Channels
- ✅ `AdditionalFeaturesGrid` - Support options (GitHub Discussions, Discord, Docs, Email)

### Community Guidelines
- ✅ `EnterpriseCompliance` adapted - Community guidelines (Code of Conduct, Contributing, License)

### Featured Contributors
- ✅ `TestimonialsTemplate` - Contributor spotlights

### Roadmap
- ✅ `EnterpriseHowItWorks` adapted - Project roadmap (M0 → M1 → M2 → M3)

---

## 📐 Proposed Structure

```tsx
<CommunityPage>
  <HeroTemplate /> {/* Community stats */}
  <EmailCapture /> {/* "Join Community" */}
  <TestimonialsTemplate /> {/* Community stats */}
  <UseCasesTemplate /> {/* Contribution types */}
  <HowItWorks /> {/* How to contribute */}
  <AdditionalFeaturesGrid /> {/* Support channels */}
  <EnterpriseCompliance /> {/* Guidelines */}
  <TestimonialsTemplate /> {/* Contributors */}
  <EnterpriseHowItWorks /> {/* Roadmap */}
  <FAQTemplate /> {/* Community FAQs */}
  <CTATemplate /> {/* "Start Contributing" */}
</CommunityPage>
```

---

## ✅ Implementation Checklist

- [ ] Read TEMPLATE_CATALOG.md
- [ ] Create `CommunityPageProps.tsx`
- [ ] Adapt templates for community context
- [ ] Write community-focused copy
- [ ] Create `CommunityPage.tsx`
- [ ] Test and document

---

**Key Message:** Open-source first. Show the community, the contribution process, and the support channels.

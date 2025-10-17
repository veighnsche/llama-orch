# CompliancePage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/compliance` (Compliance & Regulatory)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Compliance page showcasing rbee for regulated industries requiring GDPR, SOC2, ISO 27001 compliance.

**Target Audience:** Compliance officers, DPOs, legal teams, regulated industries

**Key Message:** Compliance by design. EU-native data paths. Tamper-evident logs. Audit-ready.

---

## 🔄 Key Template Adaptations for Compliance

### Hero
- ✅ `EnterpriseHero` - PERFECT FIT (audit console, compliance chips, floating badges)

### Problem: Compliance Risks
- ✅ `ProblemTemplate` - Data sovereignty violations, missing audit trails, regulatory fines

### Solution: Compliance by Design
- ✅ `SolutionTemplate` - EU-native, immutable logs, audit endpoints

### Compliance Standards
- ✅ `EnterpriseCompliance` - PERFECT FIT (GDPR, SOC2, ISO 27001 pillars)

### Security Features
- ✅ `EnterpriseSecurity` - PERFECT FIT (6 security crates)

### Audit Process
- ✅ `EnterpriseHowItWorks` - Audit preparation process

### Industry Use Cases
- ✅ `EnterpriseUseCases` - PERFECT FIT (Finance, Healthcare, Legal, Government)

### Compliance Comparison
- ✅ `ComparisonTemplate` - rbee vs cloud providers (compliance features)

### Audit Cost Estimator
- ✅ `ProvidersEarnings` adapted - Calculate audit costs (events × retention × storage)

---

## 📐 Proposed Structure

```tsx
<CompliancePage>
  <EnterpriseHero /> {/* Audit console */}
  <EmailCapture /> {/* "Download Compliance Pack" */}
  <ProblemTemplate /> {/* Compliance risks */}
  <SolutionTemplate /> {/* Compliance by design */}
  <EnterpriseCompliance /> {/* Standards */}
  <EnterpriseSecurity /> {/* Security features */}
  <EnterpriseHowItWorks /> {/* Audit process */}
  <EnterpriseUseCases /> {/* Industry cases */}
  <ComparisonTemplate /> {/* vs cloud providers */}
  <ProvidersEarnings /> {/* Audit cost estimator */}
  <FAQTemplate /> {/* Compliance FAQs */}
  <EnterpriseCTA /> {/* Multi-option CTA */}
</CompliancePage>
```

**Note:** This page can reuse almost ALL Enterprise templates directly!

---

## ✅ Implementation Checklist

- [ ] Read TEMPLATE_CATALOG.md
- [ ] Create `CompliancePageProps.tsx`
- [ ] Reuse Enterprise templates with compliance-focused copy
- [ ] Adapt `ProvidersEarnings` for audit cost calculation
- [ ] Create `CompliancePage.tsx`
- [ ] Test and document

---

**Key Message:** This is the easiest page—Enterprise templates ARE compliance templates. Just adjust the copy!

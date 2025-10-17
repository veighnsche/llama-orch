# SecurityPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/security` (Security & Privacy)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Security page showcasing rbee's security architecture, threat model, and security practices.

**Target Audience:** Security engineers, CISOs, security-conscious developers

**Key Message:** Defense-in-depth. Zero-trust. Process isolation. Security by design.

---

## 🔄 Key Template Adaptations for Security

### Hero
- ✅ `EnterpriseHero` adapted - Security console showing threat detection

### Security Architecture
- ✅ `EnterpriseSecurity` - PERFECT FIT (6 security crates with detailed bullets)

### Threat Model
- ✅ `ProblemTemplate` adapted - Security threats (injection, exhaustion, leaks, etc.)

### Security Layers
- ✅ `SolutionTemplate` - Defense-in-depth layers

### Security Features
- ✅ `SecurityIsolation` - PERFECT FIT (process isolation, sandboxing)

### Security Process
- ✅ `EnterpriseHowItWorks` adapted - Security development lifecycle

### Security Guarantees
- ✅ `EnterpriseCompliance` adapted - Security guarantees (timing-safe, zeroization, fingerprinting)

### Vulnerability Disclosure
- ✅ `HowItWorks` - How to report vulnerabilities

---

## 📐 Proposed Structure

```tsx
<SecurityPage>
  <EnterpriseHero /> {/* Security console */}
  <EmailCapture /> {/* "Get Security Whitepaper" */}
  <ProblemTemplate /> {/* Threat model */}
  <SolutionTemplate /> {/* Defense-in-depth */}
  <EnterpriseSecurity /> {/* Security crates */}
  <SecurityIsolation /> {/* Isolation features */}
  <EnterpriseCompliance /> {/* Guarantees */}
  <HowItWorks /> {/* Vulnerability disclosure */}
  <TechnicalTemplate /> {/* Security architecture */}
  <FAQTemplate /> {/* Security FAQs */}
  <CTATemplate /> {/* "Review Security Docs" */}
</SecurityPage>
```

---

## ✅ Implementation Checklist

- [ ] Read TEMPLATE_CATALOG.md
- [ ] Create `SecurityPageProps.tsx`
- [ ] Reuse Enterprise security templates
- [ ] Write security-focused copy (emphasize zero-trust, isolation, guarantees)
- [ ] Create `SecurityPage.tsx`
- [ ] Test and document

---

**Key Message:** Show the security architecture. Security engineers want details—give them the crates, the guarantees, the threat model.

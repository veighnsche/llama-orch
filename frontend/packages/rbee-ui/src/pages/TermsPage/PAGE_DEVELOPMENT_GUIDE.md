# TermsPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/legal/terms` (Terms of Service)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Terms of Service page with legal content and user-friendly navigation.

**Target Audience:** Users, legal teams

**Key Message:** Clear terms. Open-source license. Fair use policies.

---

## 🔄 Template Recommendations

### Simple Legal Page Structure

**Option 1: FAQ-style (Recommended)**
```tsx
<TermsPage>
  <HeroTemplate /> {/* "Terms of Service" title */}
  <FAQTemplate /> {/* Terms sections as Q&A */}
  <CTATemplate /> {/* "Contact Legal Team" */}
</TermsPage>
```

**Option 2: Accordion-style**
```tsx
<TermsPage>
  <HeroTemplate /> {/* Title + last updated */}
  <CardGridTemplate /> {/* Terms sections as cards */}
  <CTATemplate /> {/* Contact */}
</TermsPage>
```

---

## 📋 Content Structure

### Required Sections
1. Acceptance of terms
2. License (GPL-3.0-or-later)
3. User accounts
4. Acceptable use
5. Prohibited activities
6. Intellectual property
7. Disclaimers
8. Limitation of liability
9. Indemnification
10. Termination
11. Governing law
12. Dispute resolution
13. Changes to terms
14. Contact information

---

## ✅ Implementation Checklist

- [ ] Read TEMPLATE_CATALOG.md
- [ ] Create `TermsPageProps.tsx`
- [ ] Use `FAQTemplate` for terms sections
- [ ] Write clear, fair terms
- [ ] Reference GPL-3.0-or-later license
- [ ] Add last updated date
- [ ] Create `TermsPage.tsx`
- [ ] Legal review required before publish

---

**Note:** This is a legal page. Keep it simple. Use `FAQTemplate` for easy navigation.

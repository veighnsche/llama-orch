# PrivacyPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/legal/privacy` (Privacy Policy)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Privacy Policy page with legal content and user-friendly navigation.

**Target Audience:** Users, legal teams, compliance officers

**Key Message:** Transparent privacy practices. GDPR-compliant. User rights respected.

---

## 🔄 Template Recommendations

### Simple Legal Page Structure

**Option 1: FAQ-style (Recommended)**
```tsx
<PrivacyPage>
  <HeroTemplate /> {/* "Privacy Policy" title */}
  <FAQTemplate /> {/* Privacy topics as Q&A */}
  <CTATemplate /> {/* "Contact Privacy Team" */}
</PrivacyPage>
```

**Option 2: Accordion-style**
```tsx
<PrivacyPage>
  <HeroTemplate /> {/* Title + last updated */}
  <CardGridTemplate /> {/* Privacy sections as cards */}
  <CTATemplate /> {/* Contact */}
</PrivacyPage>
```

---

## 📋 Content Structure

### Required Sections (GDPR)
1. What data we collect
2. How we use data
3. Data retention
4. Your rights (access, deletion, portability)
5. Cookies and tracking
6. Third-party services
7. Data security
8. International transfers
9. Children's privacy
10. Policy updates
11. Contact information

---

## ✅ Implementation Checklist

- [ ] Read TEMPLATE_CATALOG.md
- [ ] Create `PrivacyPageProps.tsx`
- [ ] Use `FAQTemplate` for privacy sections
- [ ] Write GDPR-compliant content
- [ ] Add last updated date
- [ ] Create `PrivacyPage.tsx`
- [ ] Legal review required before publish

---

**Note:** This is a legal page. Keep it simple. Use `FAQTemplate` for easy navigation.

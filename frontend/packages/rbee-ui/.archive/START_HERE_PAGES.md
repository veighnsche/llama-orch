# 🚀 START HERE: Page Development Assignments

**Project:** rbee-ui Pages  
**Total Developers Needed:** 10  
**Estimated Time per Page:** 6-8 hours  
**Last Updated:** Oct 17, 2025

---

## 📖 Before You Start

### Required Reading (30 minutes)

Read these documents **in order** before touching any code:

1. ✅ **TEMPLATE_CATALOG.md** - Complete template inventory with reusability analysis
2. ✅ **PAGE_DEVELOPMENT_INDEX.md** - Master tracking and workflow
3. ✅ **Your assigned folder's PAGE_DEVELOPMENT_GUIDE.md** - Specific instructions for your page

### Core Philosophy

**🎯 REUSE, DON'T CREATE**

- Template names are **marketing labels**, not technical constraints
- Try adapting 3+ existing templates before creating new ones
- Only propose new templates if absolutely necessary
- Speed comes from reuse, not creation

---

## 👥 Developer Assignments

### Developer 1: ResearchPage
**Folder:** `src/pages/ResearchPage/`  
**Route:** `/research`  
**Priority:** 🔴 High  
**Estimated Time:** 8 hours

**Your Mission:** Build the Research & Academia page showcasing reproducible experiments and deterministic seeds.

**Key Templates to Reuse:**
- `HeroTemplate` with experiment visualization
- `ProvidersEarnings` → Experiment cost calculator
- `SolutionTemplate` for reproducibility features
- `FeaturesTabs` for multi-modal support
- `HowItWorks` for research workflow

**Start Here:**
1. Read `src/pages/ResearchPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/ResearchPage/CHECKLIST.md`
3. Create `src/pages/ResearchPage/ResearchPageProps.tsx`
4. Create `src/pages/ResearchPage/ResearchPage.tsx`

---

### Developer 2: HomelabPage
**Folder:** `src/pages/HomelabPage/`  
**Route:** `/homelab`  
**Priority:** 🔴 High  
**Estimated Time:** 7 hours

**Your Mission:** Build the Homelab & Self-Hosting page for hardware enthusiasts.

**Key Templates to Reuse:**
- `HeroTemplate` with network topology
- `ProvidersEarnings` → Power cost calculator
- `CrossNodeOrchestration` for multi-machine setup
- `MultiBackendGpuTemplate` for hardware support
- `SecurityIsolation` for privacy features

**Start Here:**
1. Read `src/pages/HomelabPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/HomelabPage/CHECKLIST.md`
3. Create `src/pages/HomelabPage/HomelabPageProps.tsx`
4. Create `src/pages/HomelabPage/HomelabPage.tsx`

---

### Developer 3: StartupsPage
**Folder:** `src/pages/StartupsPage/`  
**Route:** `/startups`  
**Priority:** 🔴 High  
**Estimated Time:** 8 hours

**Your Mission:** Build the Startups & Small Teams page emphasizing cost savings.

**Key Templates to Reuse:**
- `HeroTemplate` with cost savings focus
- `ProvidersEarnings` → ROI calculator (API costs vs self-hosted)
- `EnterpriseHowItWorks` → Growth roadmap
- `ComparisonTemplate` for rbee vs API providers
- `PricingTemplate` → Growth tiers

**Start Here:**
1. Read `src/pages/StartupsPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/StartupsPage/CHECKLIST.md`
3. Create `src/pages/StartupsPage/StartupsPageProps.tsx`
4. Create `src/pages/StartupsPage/StartupsPage.tsx`

---

### Developer 4: EducationPage
**Folder:** `src/pages/EducationPage/`  
**Route:** `/education`  
**Priority:** 🔴 High  
**Estimated Time:** 7 hours

**Your Mission:** Build the Education & Learning page for teaching distributed AI.

**Key Templates to Reuse:**
- `HeroTemplate` with learning path visualization
- `ProvidersEarnings` → Learning time estimator
- `PricingTemplate` → Course levels (Beginner/Intermediate/Advanced)
- `EnterpriseSecurity` → Curriculum modules (6 cards)
- `TestimonialsTemplate` → Student outcomes

**Start Here:**
1. Read `src/pages/EducationPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/EducationPage/CHECKLIST.md`
3. Create `src/pages/EducationPage/EducationPageProps.tsx`
4. Create `src/pages/EducationPage/EducationPage.tsx`

---

### Developer 5: DevOpsPage
**Folder:** `src/pages/DevOpsPage/`  
**Route:** `/devops`  
**Priority:** 🔴 High  
**Estimated Time:** 7 hours

**Your Mission:** Build the DevOps & Production page for infrastructure teams.

**Key Templates to Reuse:**
- `EnterpriseHero` → Deployment console
- `EnterpriseSecurity` → Ops features (6 cards)
- `EnterpriseHowItWorks` → Deployment process
- `ErrorHandlingTemplate` for resilience
- `RealTimeProgress` for monitoring

**Start Here:**
1. Read `src/pages/DevOpsPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/DevOpsPage/CHECKLIST.md`
3. Create `src/pages/DevOpsPage/DevOpsPageProps.tsx`
4. Create `src/pages/DevOpsPage/DevOpsPage.tsx`

---

### Developer 6: CompliancePage
**Folder:** `src/pages/CompliancePage/`  
**Route:** `/compliance`  
**Priority:** 🔴 High  
**Estimated Time:** 6 hours (EASIEST - reuses all Enterprise templates!)

**Your Mission:** Build the Compliance & Regulatory page for regulated industries.

**Key Templates to Reuse:**
- `EnterpriseHero` - PERFECT FIT (audit console)
- `EnterpriseCompliance` - PERFECT FIT (GDPR/SOC2/ISO)
- `EnterpriseSecurity` - PERFECT FIT (security crates)
- `EnterpriseHowItWorks` - Audit process
- `EnterpriseUseCases` - PERFECT FIT (Finance/Healthcare/Legal/Gov)
- `ProvidersEarnings` → Audit cost estimator

**Start Here:**
1. Read `src/pages/CompliancePage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/CompliancePage/CHECKLIST.md`
3. Create `src/pages/CompliancePage/CompliancePageProps.tsx`
4. Create `src/pages/CompliancePage/CompliancePage.tsx`

**Note:** This is the easiest page—just reuse Enterprise templates with compliance-focused copy!

---

### Developer 7: CommunityPage
**Folder:** `src/pages/CommunityPage/`  
**Route:** `/community`  
**Priority:** 🟡 Medium  
**Estimated Time:** 6 hours

**Your Mission:** Build the Community & Support page for open-source contributors.

**Key Templates to Reuse:**
- `HeroTemplate` with community stats
- `TestimonialsTemplate` → Community stats (GitHub stars, contributors, etc.)
- `UseCasesTemplate` → Contribution types (code, docs, testing, design)
- `HowItWorks` → How to contribute
- `EnterpriseHowItWorks` → Project roadmap

**Start Here:**
1. Read `src/pages/CommunityPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/CommunityPage/CHECKLIST.md`
3. Create `src/pages/CommunityPage/CommunityPageProps.tsx`
4. Create `src/pages/CommunityPage/CommunityPage.tsx`

---

### Developer 8: SecurityPage
**Folder:** `src/pages/SecurityPage/`  
**Route:** `/security`  
**Priority:** 🟡 Medium  
**Estimated Time:** 6 hours

**Your Mission:** Build the Security & Privacy page showcasing security architecture.

**Key Templates to Reuse:**
- `EnterpriseHero` → Security console
- `EnterpriseSecurity` - PERFECT FIT (6 security crates)
- `SecurityIsolation` - PERFECT FIT (process isolation)
- `EnterpriseCompliance` → Security guarantees
- `TechnicalTemplate` → Security architecture

**Start Here:**
1. Read `src/pages/SecurityPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/SecurityPage/CHECKLIST.md`
3. Create `src/pages/SecurityPage/SecurityPageProps.tsx`
4. Create `src/pages/SecurityPage/SecurityPage.tsx`

---

### Developer 9: PrivacyPage
**Folder:** `src/pages/PrivacyPage/`  
**Route:** `/legal/privacy`  
**Priority:** 🟢 Low  
**Estimated Time:** 3 hours (SIMPLE - legal page)

**Your Mission:** Build the Privacy Policy page with GDPR-compliant content.

**Key Templates to Reuse:**
- `HeroTemplate` (simple title)
- `FAQTemplate` → Privacy sections as Q&A
- `CTATemplate` → Contact privacy team

**Start Here:**
1. Read `src/pages/PrivacyPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/PrivacyPage/CHECKLIST.md`
3. Create `src/pages/PrivacyPage/PrivacyPageProps.tsx`
4. Create `src/pages/PrivacyPage/PrivacyPage.tsx`

**Note:** Keep it simple. Use `FAQTemplate` for easy navigation. Legal review required before publish.

---

### Developer 10: TermsPage
**Folder:** `src/pages/TermsPage/`  
**Route:** `/legal/terms`  
**Priority:** 🟢 Low  
**Estimated Time:** 3 hours (SIMPLE - legal page)

**Your Mission:** Build the Terms of Service page with clear, fair terms.

**Key Templates to Reuse:**
- `HeroTemplate` (simple title)
- `FAQTemplate` → Terms sections as Q&A
- `CTATemplate` → Contact legal team

**Start Here:**
1. Read `src/pages/TermsPage/PAGE_DEVELOPMENT_GUIDE.md`
2. Review `src/pages/TermsPage/CHECKLIST.md`
3. Create `src/pages/TermsPage/TermsPageProps.tsx`
4. Create `src/pages/TermsPage/TermsPage.tsx`

**Note:** Keep it simple. Reference GPL-3.0-or-later license. Legal review required before publish.

---

## 📊 Quick Stats

### By Priority

**🔴 High Priority (Industry Pages):** 6 developers
- Developer 1: ResearchPage
- Developer 2: HomelabPage
- Developer 3: StartupsPage
- Developer 4: EducationPage
- Developer 5: DevOpsPage
- Developer 6: CompliancePage

**🟡 Medium Priority (Support Pages):** 2 developers
- Developer 7: CommunityPage
- Developer 8: SecurityPage

**🟢 Low Priority (Legal Pages):** 2 developers
- Developer 9: PrivacyPage
- Developer 10: TermsPage

### By Difficulty

**Easy (3-6 hours):**
- Developer 6: CompliancePage (6h - reuses all Enterprise templates)
- Developer 7: CommunityPage (6h)
- Developer 8: SecurityPage (6h)
- Developer 9: PrivacyPage (3h - simple legal page)
- Developer 10: TermsPage (3h - simple legal page)

**Medium (7-8 hours):**
- Developer 2: HomelabPage (7h)
- Developer 4: EducationPage (7h)
- Developer 5: DevOpsPage (7h)

**Complex (8+ hours):**
- Developer 1: ResearchPage (8h - calculator adaptation)
- Developer 3: StartupsPage (8h - ROI calculator)

---

## 🔄 Workflow for Each Developer

### Phase 1: Setup (30 min)
1. ✅ Read TEMPLATE_CATALOG.md
2. ✅ Read PAGE_DEVELOPMENT_INDEX.md
3. ✅ Read your folder's PAGE_DEVELOPMENT_GUIDE.md
4. ✅ Read your folder's CHECKLIST.md
5. ✅ Review existing page props files (HomePage, EnterprisePage, ProvidersPage)

### Phase 2: Props Definition (2-3 hours)
1. Create `[PageName]PageProps.tsx` in your folder
2. Define all container props (`[section]ContainerProps`)
3. Define all template props (`[section]Props`)
4. Adapt existing templates (e.g., ProvidersEarnings → calculator)
5. Write all content (headlines, descriptions, bullets, etc.)

### Phase 3: Page Component (1 hour)
1. Create `[PageName]Page.tsx` in your folder
2. Import all templates
3. Import all props from `[PageName]PageProps.tsx`
4. Compose page with TemplateContainer wrappers
5. Add proper TypeScript types

### Phase 4: Testing (1 hour)
1. Test in Storybook
2. Test responsive layout (mobile, tablet, desktop)
3. Test dark mode
4. Test interactive elements (tabs, accordion, calculator)
5. Verify accessibility (ARIA labels, keyboard navigation)

### Phase 5: Documentation (30 min)
1. Update your folder's CHECKLIST.md with completion status
2. Document any template adaptations made
3. Update PAGE_DEVELOPMENT_INDEX.md status
4. Note any issues or improvements needed

---

## 🚫 When to Propose a New Template

**Only if:**
1. ✅ You've tried adapting at least 3 existing templates
2. ✅ You can explain why each won't work
3. ✅ The new template would be reusable for other pages
4. ✅ You've written a proposal document

**Proposal location:** `src/templates/[TemplateName]/PROPOSAL.md`

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
[TypeScript interface]

## Reusability Analysis
How can other pages use this template?
```

---

## 📁 Folder Structure

Each developer works in their assigned folder:

```
src/pages/
├── ResearchPage/          ← Developer 1
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── ResearchPageProps.tsx     (YOU CREATE THIS)
│   └── ResearchPage.tsx          (YOU CREATE THIS)
├── HomelabPage/           ← Developer 2
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── HomelabPageProps.tsx      (YOU CREATE THIS)
│   └── HomelabPage.tsx           (YOU CREATE THIS)
├── StartupsPage/          ← Developer 3
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── StartupsPageProps.tsx     (YOU CREATE THIS)
│   └── StartupsPage.tsx          (YOU CREATE THIS)
├── EducationPage/         ← Developer 4
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── EducationPageProps.tsx    (YOU CREATE THIS)
│   └── EducationPage.tsx         (YOU CREATE THIS)
├── DevOpsPage/            ← Developer 5
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── DevOpsPageProps.tsx       (YOU CREATE THIS)
│   └── DevOpsPage.tsx            (YOU CREATE THIS)
├── CompliancePage/        ← Developer 6
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── CompliancePageProps.tsx   (YOU CREATE THIS)
│   └── CompliancePage.tsx        (YOU CREATE THIS)
├── CommunityPage/         ← Developer 7
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── CommunityPageProps.tsx    (YOU CREATE THIS)
│   └── CommunityPage.tsx         (YOU CREATE THIS)
├── SecurityPage/          ← Developer 8
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── SecurityPageProps.tsx     (YOU CREATE THIS)
│   └── SecurityPage.tsx          (YOU CREATE THIS)
├── PrivacyPage/           ← Developer 9
│   ├── CHECKLIST.md
│   ├── PAGE_DEVELOPMENT_GUIDE.md
│   ├── PrivacyPageProps.tsx      (YOU CREATE THIS)
│   └── PrivacyPage.tsx           (YOU CREATE THIS)
└── TermsPage/             ← Developer 10
    ├── CHECKLIST.md
    ├── PAGE_DEVELOPMENT_GUIDE.md
    ├── TermsPageProps.tsx        (YOU CREATE THIS)
    └── TermsPage.tsx             (YOU CREATE THIS)
```

---

## 🎨 Design Consistency Rules

### Background Decorations
```tsx
// ✅ CORRECT
background: {
  decoration: (
    <div className="absolute inset-0 opacity-25">
      <NetworkMesh className="blur-[0.5px]" />
    </div>
  ),
}

// ❌ WRONG (causes invisible backgrounds)
background: {
  decoration: <NetworkMesh className="absolute inset-0 -z-10 opacity-25" />
}
```

### Spacing
- Use TemplateContainer `paddingY` prop: `'sm' | 'md' | 'lg' | 'xl' | '2xl'`
- Don't mix manual spacing (mb-4, mb-6) with component spacing

### Max-Width
- Use TemplateContainer `maxWidth` prop: `'3xl' | '5xl' | '6xl' | '7xl'`
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
1. Check your folder's CHECKLIST.md
2. Look at similar sections on existing pages
3. Review your folder's PAGE_DEVELOPMENT_GUIDE.md

### Still Stuck?
Ask in team chat with:
- What you're trying to achieve
- What you've tried (which templates)
- Why it didn't work

---

## ✅ Success Criteria

Your page is complete when:

- ✅ Uses 100% existing templates (no new templates created)
- ✅ All content requirements from CHECKLIST.md met
- ✅ Props file follows existing patterns (see HomePage, EnterprisePage)
- ✅ Page component is clean and readable
- ✅ Responsive (mobile, tablet, desktop)
- ✅ Accessible (ARIA labels, keyboard navigation)
- ✅ Works in light and dark modes
- ✅ All interactive elements tested
- ✅ CHECKLIST.md updated with completion status

---

## 🚀 Ready to Start?

1. **Find your developer number** (1-10)
2. **Go to your assigned folder** (`src/pages/[PageName]Page/`)
3. **Read your PAGE_DEVELOPMENT_GUIDE.md**
4. **Start building!**

**Remember:** Speed comes from reuse. Think creatively about adapting existing templates. Only create new templates as a last resort.

---

**Good luck! 🐝**

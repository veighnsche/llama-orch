# rbee Commercial Frontend v2 - Department Workflow

**Project Manager:** TEAM-FE-000  
**Date:** 2025-10-11  
**Status:** Workflow Defined - Ready to Execute

---

## 🎯 Project Goal

Create a world-class commercial website for rbee that:
1. Clearly communicates the value proposition to developers
2. Addresses the fear of AI provider dependency
3. Converts visitors into users/customers
4. Establishes rbee as the solution for AI independence

---

## 📋 Optimal Department Sequence

Based on professional agency workflows and the stakeholder documentation, here's the optimal sequence:

### **Phase 1: Foundation (Strategy & Content)**

#### 1️⃣ TEAM-FE-001-CONTENT-STRATEGY (Content Strategist)
**Duration:** 2-3 days  
**Input:** Stakeholder docs  
**Output:** Content strategy document

**Responsibilities:**
- Analyze stakeholder documentation
- Define target audience personas (Developers, DevOps, Startups)
- Map user journeys and conversion funnels
- Define information architecture (sitemap, page hierarchy)
- Identify key messages and value propositions
- Define CTAs (Call-to-Actions) for each page
- Create content requirements document

**Deliverables:**
- `CONTENT_STRATEGY.md` - Complete content strategy
- `SITEMAP.md` - Site structure and navigation
- `USER_JOURNEYS.md` - User flow diagrams
- `MESSAGING_HIERARCHY.md` - Key messages per page

**Handoff Checklist:**
- [ ] Target audiences clearly defined
- [ ] User journeys mapped for each persona
- [ ] Sitemap with all pages and sections
- [ ] CTAs defined for each page
- [ ] Messaging hierarchy established
- [ ] Content requirements documented

---

#### 2️⃣ TEAM-FE-002-COPYWRITING (Copywriter)
**Duration:** 3-5 days  
**Input:** Content strategy document  
**Output:** All website copy

**Responsibilities:**
- Write compelling headlines and subheadlines
- Craft body copy for all pages
- Write CTAs that convert
- Create microcopy (buttons, labels, tooltips)
- Write meta descriptions and SEO copy
- Ensure consistent tone and voice
- Address pain points and objections

**Deliverables:**
- `COPY_HOME.md` - Home page copy
- `COPY_PRODUCT.md` - Product/features pages
- `COPY_PRICING.md` - Pricing page copy
- `COPY_ABOUT.md` - About/story page copy
- `COPY_USE_CASES.md` - Use case pages
- `COPY_FAQ.md` - FAQ content
- `COPY_MICROCOPY.md` - All UI microcopy

**Handoff Checklist:**
- [ ] All headlines written and approved
- [ ] Body copy complete for all pages
- [ ] CTAs written and tested for clarity
- [ ] Microcopy documented
- [ ] SEO meta descriptions (≤155 chars)
- [ ] Tone and voice consistent
- [ ] Pain points addressed
- [ ] No lorem ipsum or placeholders

---

#### 3️⃣ TEAM-FE-003-SEO (SEO Specialist)
**Duration:** 2 days  
**Input:** Copy and content strategy  
**Output:** SEO optimization plan

**Responsibilities:**
- Keyword research and mapping
- Optimize meta titles and descriptions
- Define structured data (JSON-LD schemas)
- Plan internal linking strategy
- Define canonical URLs and hreflang tags
- Create SEO checklist for implementation
- Plan sitemap.xml and robots.txt

**Deliverables:**
- `SEO_STRATEGY.md` - Complete SEO plan
- `KEYWORDS.md` - Keyword mapping per page
- `STRUCTURED_DATA.md` - JSON-LD schemas
- `SEO_CHECKLIST.md` - Implementation checklist

**Handoff Checklist:**
- [ ] Keywords researched and mapped
- [ ] Meta titles optimized (≤60 chars)
- [ ] Meta descriptions optimized (≤155 chars)
- [ ] JSON-LD schemas defined
- [ ] Internal linking strategy documented
- [ ] Technical SEO requirements listed

---

### **Phase 2: Design (Visual Identity)**

#### 4️⃣ TEAM-FE-004-BRAND-DESIGN (Brand Designer)
**Duration:** 3-4 days  
**Input:** Content strategy, copy  
**Output:** Brand identity system

**Responsibilities:**
- Define brand identity (logo, colors, typography)
- Create design tokens (CSS variables)
- Define spacing, sizing, and grid system
- Create brand guidelines document
- Design icon set or select icon library
- Define motion/animation principles

**Deliverables:**
- `BRAND_GUIDELINES.md` - Complete brand guide
- `design-tokens.css` - CSS custom properties
- Logo files (SVG, PNG)
- Color palette swatches
- Typography specimens
- Icon library

**Handoff Checklist:**
- [ ] Logo finalized (all formats)
- [ ] Color palette defined (primary, secondary, neutrals)
- [ ] Typography system (headings, body, code)
- [ ] Design tokens documented
- [ ] Spacing/sizing scale defined
- [ ] Icon library selected/created
- [ ] Brand guidelines complete

---

#### 5️⃣ TEAM-FE-005-UI-UX-DESIGN (UI/UX Designer)
**Duration:** 5-7 days  
**Input:** Brand identity, copy, content strategy  
**Output:** High-fidelity mockups

**Responsibilities:**
- Create wireframes for all pages
- Design high-fidelity mockups (desktop, tablet, mobile)
- Define component library (buttons, cards, forms, etc.)
- Design interactive states (hover, active, disabled)
- Create responsive breakpoints
- Design loading states and error states
- Prototype key user flows

**Deliverables:**
- Wireframes (low-fidelity)
- High-fidelity mockups (Figma/Sketch)
- Component library designs
- Responsive design specs
- Interactive prototype
- `DESIGN_SPECS.md` - Design specifications

**Handoff Checklist:**
- [ ] Wireframes approved for all pages
- [ ] High-fidelity mockups complete (desktop, tablet, mobile)
- [ ] Component library designed
- [ ] Interactive states defined
- [ ] Responsive breakpoints specified
- [ ] Design specs documented
- [ ] Prototype demonstrates key flows

---

#### 6️⃣ TEAM-FE-006-VISUAL-ASSETS (Illustrator/Visual Designer)
**Duration:** 3-4 days  
**Input:** Brand guidelines, mockups  
**Output:** All visual assets

**Responsibilities:**
- Create hero images/illustrations
- Design diagrams (architecture, workflows)
- Create icons (if custom)
- Optimize images for web (WebP, compression)
- Create social media preview images
- Design loading animations
- Create favicon and app icons

**Deliverables:**
- Hero images (all pages)
- Diagrams and infographics
- Icon set (if custom)
- Optimized image assets
- Social preview images (OG images)
- Favicon and app icons
- `ASSETS_INVENTORY.md` - Asset catalog

**Handoff Checklist:**
- [ ] Hero images created and optimized
- [ ] Diagrams/infographics complete
- [ ] Icons finalized
- [ ] All images optimized (WebP, compressed)
- [ ] Social preview images (1200x630)
- [ ] Favicon set (multiple sizes)
- [ ] Asset inventory documented

---

### **Phase 3: Development (Implementation)**

#### 7️⃣ TEAM-FE-007-DESIGN-SYSTEM (Frontend Developer - Design System)
**Duration:** 3-4 days  
**Input:** Brand guidelines, component designs  
**Output:** Component library

**Responsibilities:**
- Implement design tokens in CSS
- Build reusable Vue components
- Integrate with orchyra-storybook
- Document component APIs
- Create component stories (Histoire)
- Ensure accessibility (ARIA, keyboard nav)
- Write component tests

**Deliverables:**
- Design tokens implemented
- Component library (Button, Card, Input, etc.)
- Component documentation
- Histoire stories for all components
- Accessibility audit results
- `COMPONENT_LIBRARY.md` - Component docs

**Handoff Checklist:**
- [ ] Design tokens in CSS variables
- [ ] All components built and tested
- [ ] Components integrated with storybook
- [ ] Histoire stories created
- [ ] Accessibility tested (WCAG AA)
- [ ] Component APIs documented
- [ ] No relative imports (workspace packages only)

---

#### 8️⃣ TEAM-FE-008-PAGE-IMPLEMENTATION (Frontend Developer - Pages)
**Duration:** 5-7 days  
**Input:** Mockups, copy, components  
**Output:** All pages implemented

**Responsibilities:**
- Build all page views
- Implement routing (Vue Router)
- Integrate copy from markdown files
- Implement responsive layouts
- Add animations and transitions
- Implement SEO meta tags
- Add JSON-LD structured data
- Ensure mobile-first responsive design

**Deliverables:**
- All pages implemented
- Routing configured
- Responsive layouts working
- Animations/transitions added
- SEO meta tags implemented
- JSON-LD schemas added
- `IMPLEMENTATION_NOTES.md` - Technical notes

**Handoff Checklist:**
- [ ] All pages built and responsive
- [ ] Routing works correctly
- [ ] Copy integrated (no hardcoded text)
- [ ] Animations smooth (60fps)
- [ ] SEO meta tags on all pages
- [ ] JSON-LD schemas implemented
- [ ] Mobile-first responsive
- [ ] No console errors

---

#### 9️⃣ TEAM-FE-009-INTERACTIONS (Frontend Developer - Interactions)
**Duration:** 2-3 days  
**Input:** Implemented pages  
**Output:** Polished interactions

**Responsibilities:**
- Add micro-interactions
- Implement scroll animations
- Add loading states
- Implement form validation
- Add error handling
- Optimize animations (performance)
- Add keyboard navigation
- Implement focus management

**Deliverables:**
- Micro-interactions implemented
- Scroll animations added
- Loading states for all async actions
- Form validation working
- Error handling complete
- `INTERACTIONS_GUIDE.md` - Interaction patterns

**Handoff Checklist:**
- [ ] Micro-interactions polished
- [ ] Scroll animations smooth
- [ ] Loading states implemented
- [ ] Forms validate correctly
- [ ] Error messages clear and helpful
- [ ] Keyboard navigation works
- [ ] Focus management correct
- [ ] Performance optimized (no jank)

---

### **Phase 4: Quality Assurance**

#### 🔟 TEAM-FE-010-QA-TESTING (QA Engineer)
**Duration:** 3-4 days  
**Input:** Complete implementation  
**Output:** Bug-free website

**Responsibilities:**
- Cross-browser testing (Chrome, Firefox, Safari, Edge)
- Mobile device testing (iOS, Android)
- Accessibility testing (screen readers, keyboard)
- Performance testing (Lighthouse, Core Web Vitals)
- Functional testing (links, forms, navigation)
- Visual regression testing
- Create bug reports and track fixes

**Deliverables:**
- `QA_TEST_PLAN.md` - Test plan
- `QA_REPORT.md` - Test results
- Bug reports (with screenshots)
- Browser compatibility matrix
- Performance audit results
- Accessibility audit results

**Handoff Checklist:**
- [ ] All browsers tested (Chrome, Firefox, Safari, Edge)
- [ ] Mobile devices tested (iOS, Android)
- [ ] Accessibility audit passed (WCAG AA)
- [ ] Performance audit passed (Lighthouse >90)
- [ ] All links working
- [ ] All forms working
- [ ] No console errors
- [ ] All bugs fixed or documented

---

#### 1️⃣1️⃣ TEAM-FE-011-PERFORMANCE (Performance Engineer)
**Duration:** 2-3 days  
**Input:** QA-tested website  
**Output:** Optimized website

**Responsibilities:**
- Optimize images (lazy loading, WebP)
- Implement code splitting
- Optimize bundle size
- Add caching strategies
- Optimize fonts (preload, subset)
- Minimize render-blocking resources
- Optimize Core Web Vitals (LCP, FID, CLS)
- Add performance monitoring

**Deliverables:**
- Optimized build
- Performance report
- `PERFORMANCE_OPTIMIZATIONS.md` - Optimization log
- Lighthouse score >90
- Core Web Vitals passing

**Handoff Checklist:**
- [ ] Images optimized and lazy-loaded
- [ ] Code splitting implemented
- [ ] Bundle size optimized (<200KB gzipped)
- [ ] Fonts optimized (preload, subset)
- [ ] Lighthouse score >90
- [ ] Core Web Vitals passing (LCP <2.5s, FID <100ms, CLS <0.1)
- [ ] Performance monitoring added

---

#### 1️⃣2️⃣ TEAM-FE-012-ACCESSIBILITY (Accessibility Specialist)
**Duration:** 2 days  
**Input:** Optimized website  
**Output:** WCAG AA compliant website

**Responsibilities:**
- Screen reader testing (NVDA, JAWS, VoiceOver)
- Keyboard navigation testing
- Color contrast verification
- ARIA label audit
- Focus management audit
- Semantic HTML audit
- Form accessibility audit
- Create accessibility statement

**Deliverables:**
- `ACCESSIBILITY_AUDIT.md` - Audit results
- `ACCESSIBILITY_STATEMENT.md` - Public statement
- Remediation recommendations
- WCAG AA compliance report

**Handoff Checklist:**
- [ ] Screen reader tested (NVDA, JAWS, VoiceOver)
- [ ] Keyboard navigation works (Tab, Enter, Esc)
- [ ] Color contrast passes (4.5:1 for text)
- [ ] ARIA labels correct
- [ ] Focus management correct
- [ ] Semantic HTML used
- [ ] Forms accessible
- [ ] WCAG AA compliant

---

### **Phase 5: Launch Preparation**

#### 1️⃣3️⃣ TEAM-FE-013-FINAL-REVIEW (Project Manager + Stakeholder)
**Duration:** 1-2 days  
**Input:** Complete, tested, optimized website  
**Output:** Launch-ready website

**Responsibilities:**
- Final stakeholder review
- Content accuracy verification
- Brand consistency check
- Legal compliance check
- Analytics setup verification
- Error tracking setup
- Deployment checklist
- Launch plan

**Deliverables:**
- `LAUNCH_CHECKLIST.md` - Pre-launch checklist
- `DEPLOYMENT_PLAN.md` - Deployment steps
- Stakeholder sign-off
- Launch plan

**Handoff Checklist:**
- [ ] Stakeholder approval received
- [ ] Content accuracy verified
- [ ] Brand consistency confirmed
- [ ] Legal compliance checked
- [ ] Analytics configured
- [ ] Error tracking configured
- [ ] Deployment plan ready
- [ ] Rollback plan ready

---

## 📊 Timeline Estimate

| Phase | Duration | Teams |
|-------|----------|-------|
| **Phase 1: Foundation** | 7-10 days | Content Strategy, Copywriting, SEO |
| **Phase 2: Design** | 11-15 days | Brand Design, UI/UX, Visual Assets |
| **Phase 3: Development** | 10-14 days | Design System, Pages, Interactions |
| **Phase 4: QA** | 7-9 days | QA Testing, Performance, Accessibility |
| **Phase 5: Launch Prep** | 1-2 days | Final Review |
| **TOTAL** | **36-50 days** | 13 teams |

---

## 🔄 Handoff Process

Each team must:

1. **Review previous team's deliverables** - Read all handoff documents
2. **Complete their work** - Follow their responsibilities
3. **Create deliverables** - Document all outputs
4. **Complete handoff checklist** - Verify all items checked
5. **Create handoff document** - Max 2 pages with:
   - What was completed
   - Key decisions made
   - Files created/modified
   - Next team's inputs ready
   - Blockers or concerns (if any)

**Handoff Document Template:**
```markdown
# TEAM-FE-XXX-DEPARTMENT Handoff

**Team:** TEAM-FE-XXX-DEPARTMENT  
**Date:** YYYY-MM-DD  
**Duration:** X days  
**Status:** Complete ✅

## Completed Work
- [List what was done]

## Deliverables
- [List files created]

## Key Decisions
- [Important decisions made]

## Handoff Checklist
- [x] All checklist items from WORKFLOW.md

## Next Team
**TEAM-FE-YYY-NEXT-DEPARTMENT** is ready to start.
Inputs provided: [list files/docs]

## Signatures
// Created by: TEAM-FE-XXX-DEPARTMENT
```

---

## 🚨 Critical Rules

1. **No skipping departments** - Each team builds on previous work
2. **No lorem ipsum** - Wait for real copy from Copywriting team
3. **No placeholder images** - Wait for real assets from Visual Assets team
4. **Follow handoff checklist** - All items must be checked
5. **Document everything** - Future teams depend on your docs
6. **Add team signatures** - `// Created by: TEAM-FE-XXX` on all files
7. **Keep old signatures** - Never remove previous team comments
8. **Max 2 pages per handoff** - Be concise
9. **Use workspace packages** - No relative imports for shared code
10. **Test before handoff** - Verify your work builds and runs

---

## 📁 Documentation Structure

```
commercial-frontend-v2/
├── .handoffs/                    # All team handoff documents
│   ├── TEAM-FE-001-CONTENT-STRATEGY.md
│   ├── TEAM-FE-002-COPYWRITING.md
│   ├── TEAM-FE-003-SEO.md
│   └── ...
├── .content/                     # Content deliverables
│   ├── CONTENT_STRATEGY.md
│   ├── COPY_HOME.md
│   ├── SEO_STRATEGY.md
│   └── ...
├── .design/                      # Design deliverables
│   ├── BRAND_GUIDELINES.md
│   ├── DESIGN_SPECS.md
│   └── assets/
└── .qa/                          # QA deliverables
    ├── QA_REPORT.md
    ├── PERFORMANCE_OPTIMIZATIONS.md
    └── ACCESSIBILITY_AUDIT.md
```

---

## 🎯 Success Criteria

The website is ready to launch when:

- ✅ All 13 teams have completed their work
- ✅ All handoff checklists are complete
- ✅ Lighthouse score >90
- ✅ WCAG AA compliant
- ✅ Core Web Vitals passing
- ✅ Cross-browser tested
- ✅ Mobile responsive
- ✅ No console errors
- ✅ Stakeholder approved
- ✅ Deployment plan ready

---

## 🚀 Ready to Start

**Current Status:** Scaffold complete ✅  
**Next Team:** TEAM-FE-001-CONTENT-STRATEGY  
**Next Step:** Content strategist analyzes stakeholder docs and creates content strategy

---

**Project Manager:** TEAM-FE-000  
**Scaffold Complete:** 2025-10-11  
**Ready for:** TEAM-FE-001-CONTENT-STRATEGY

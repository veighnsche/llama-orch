# React to Vue Port Plan - Exhaustive TODO List

**Created by:** TEAM-FE-000 (Project Manager)  
**Date:** 2025-10-11  
**Source:** `/home/vince/Projects/llama-orch/frontend/reference/v0` ← **REACT REFERENCE**  
**Target:** `/home/vince/Projects/llama-orch/frontend/bin/commercial-frontend` ← **VUE PORT**  
**Strategy:** Component-first - Build in Storybook, then import into site

---

## 🚨 CRITICAL: THIS IS A PORT, NOT A NEW PROJECT

**YOU ARE PORTING EXISTING REACT COMPONENTS TO VUE. NOT BUILDING FROM SCRATCH.**

### What This Means:

✅ **Every component has a React reference to port from**  
✅ **Visual design is already done**  
✅ **Behavior is already defined**  
✅ **Props are already documented**  
✅ **You just need to convert React → Vue**

❌ **DO NOT:**
- Design new components
- Guess how things should work
- Change behavior without approval
- Skip reading the React reference

### React Reference:

**Location:** `/frontend/reference/v0/`  
**Components:** `/frontend/reference/v0/components/ui/`  
**Run:** `pnpm --filter frontend/reference/v0 dev`  
**URL:** http://localhost:3000

**ALWAYS compare your Vue component with the React reference side-by-side.**

---

## 📊 Analysis Summary

**Source Project:**
- Framework: Next.js 15 + React 19 + shadcn/ui
- Total Files: 140 files
- UI Components: 60+ (shadcn/ui primitives)
- Section Components: 50+ (page sections)
- Pages: 7 (Home, Features, Use Cases, Pricing, Developers, GPU Providers, Enterprise)

**Target Project:**
- Framework: Vue 3 + TypeScript + Vite
- Component Library: orchyra-storybook
- Styling: CSS with design tokens

---

## 🎯 Port Strategy

```
Phase 1: UI Primitives (60 atoms) → orchyra-storybook
Phase 2: Composite Components (15 molecules) → orchyra-storybook  
Phase 3: Section Components (50+ organisms) → orchyra-storybook
Phase 4: Page Assembly (7 pages) → commercial-frontend-v2
Phase 5: Integration & Polish → commercial-frontend-v2
```

**Key Rule:** Build in **orchyra-storybook FIRST**, then import into site.

---

## PHASE 1: UI PRIMITIVES (60 Atoms)

Location: `/frontend/libs/storybook/stories/atoms/`

### Priority 1 - Core UI (20 components)

- [ ] Button (variants, sizes, states)
- [ ] Input (types, validation states)
- [ ] Textarea (auto-resize)
- [ ] Label (required indicator)
- [ ] Checkbox (states)
- [ ] Radio Group (layouts)
- [ ] Switch (toggle)
- [ ] Slider (single/range)
- [ ] Badge ✅ (review existing)
- [ ] Avatar (sizes, fallback)
- [ ] Separator (h/v)
- [ ] Spinner (sizes)
- [ ] Skeleton (variants)
- [ ] Progress (determinate/indeterminate)
- [ ] Kbd (keyboard shortcuts)
- [ ] Card (with subcomponents)
- [ ] Alert (variants)
- [ ] Toast (with composable)
- [ ] Dialog (sizes)
- [ ] Tooltip (positioning)

### Priority 2 - Advanced UI (25 components)

- [ ] Dropdown Menu
- [ ] Context Menu
- [ ] Menubar
- [ ] Navigation Menu
- [ ] Select (single/multi)
- [ ] Command (Cmd+K palette)
- [ ] Tabs (h/v)
- [ ] Breadcrumb
- [ ] Pagination
- [ ] Drawer ✅ (review existing)
- [ ] Sheet (sides)
- [ ] Popover
- [ ] Hover Card
- [ ] Alert Dialog
- [ ] Accordion
- [ ] Collapsible
- [ ] Toggle
- [ ] Toggle Group
- [ ] Aspect Ratio
- [ ] Scroll Area
- [ ] Resizable
- [ ] Table
- [ ] Calendar
- [ ] Chart
- [ ] Carousel ✅ (review existing)

### Priority 3 - Specialized (15 components)

- [ ] Form (validation)
- [ ] Field (wrapper)
- [ ] Input Group (prefix/suffix)
- [ ] Input OTP
- [ ] Sidebar
- [ ] Empty (empty state)
- [ ] Item (list item)
- [ ] Button Group
- [ ] useMobile (composable)
- [ ] useToast (composable)
- [ ] (5 more specialized components)

---

## PHASE 2: MOLECULES (15 Components)

Location: `/frontend/libs/storybook/stories/molecules/`

- [ ] FormField (Label + Input + Error)
- [ ] SearchBar (Input + Button)
- [ ] PasswordInput (Input + Toggle)
- [ ] NavItem (Link + Icon + Active state)
- [ ] BreadcrumbItem (Link + Separator)
- [ ] StatCard (Card + Number + Label)
- [ ] FeatureCard (Card + Icon + Title + Desc)
- [ ] TestimonialCard (Card + Avatar + Quote)
- [ ] PricingCard (Card + Badge + Price + Button)
- [ ] MediaCard ✅ (review existing)
- [ ] ImageWithCaption
- [ ] ConfirmDialog
- [ ] DropdownAction
- [ ] TabPanel
- [ ] AccordionItem

---

## PHASE 3: ORGANISMS (50+ Components)

Location: `/frontend/libs/storybook/stories/organisms/`

### Navigation (2 components)

- [ ] Navigation (main nav bar)
- [ ] Footer (site footer)

### Home Page (14 sections)

- [ ] HeroSection
- [ ] WhatIsRbee
- [ ] AudienceSelector
- [ ] EmailCapture
- [ ] ProblemSection
- [ ] SolutionSection
- [ ] HowItWorksSection
- [ ] FeaturesSection
- [ ] UseCasesSection
- [ ] ComparisonSection
- [ ] PricingSection
- [ ] SocialProofSection
- [ ] TechnicalSection
- [ ] FAQSection
- [ ] CTASection

### Developers Page (10 sections)

- [ ] DevelopersHero
- [ ] DevelopersProblem
- [ ] DevelopersSolution
- [ ] DevelopersHowItWorks
- [ ] DevelopersFeatures
- [ ] DevelopersCodeExamples
- [ ] DevelopersUseCases
- [ ] DevelopersPricing
- [ ] DevelopersTestimonials
- [ ] DevelopersCTA

### Enterprise Page (11 sections)

- [ ] EnterpriseHero
- [ ] EnterpriseProblem
- [ ] EnterpriseSolution
- [ ] EnterpriseHowItWorks
- [ ] EnterpriseFeatures
- [ ] EnterpriseSecurity
- [ ] EnterpriseCompliance
- [ ] EnterpriseComparison
- [ ] EnterpriseUseCases
- [ ] EnterpriseTestimonials
- [ ] EnterpriseCTA

### GPU Providers Page (11 sections)

- [ ] ProvidersHero
- [ ] ProvidersProblem
- [ ] ProvidersSolution
- [ ] ProvidersHowItWorks
- [ ] ProvidersFeatures
- [ ] ProvidersMarketplace
- [ ] ProvidersEarnings
- [ ] ProvidersSecurity
- [ ] ProvidersUseCases
- [ ] ProvidersTestimonials
- [ ] ProvidersCTA

### Features Page (9 sections)

- [ ] FeaturesHero
- [ ] CoreFeaturesTabs
- [ ] MultiBackendGPU
- [ ] CrossNodeOrchestration
- [ ] IntelligentModelManagement
- [ ] RealTimeProgress
- [ ] ErrorHandling
- [ ] SecurityIsolation
- [ ] AdditionalFeaturesGrid

---

## PHASE 4: PAGE ASSEMBLY (7 Pages)

Location: `/frontend/bin/commercial-frontend-v2/src/views/`

- [ ] HomeView.vue (import 14 organisms)
- [ ] FeaturesView.vue (import 9 organisms)
- [ ] UseCasesView.vue (import organisms)
- [ ] PricingView.vue (import organisms)
- [ ] DevelopersView.vue (import 10 organisms)
- [ ] GpuProvidersView.vue (import 11 organisms)
- [ ] EnterpriseView.vue (import 11 organisms)

---

## PHASE 5: INTEGRATION & POLISH

### Router Setup

- [ ] Configure Vue Router with all 7 routes
- [ ] Add route transitions
- [ ] Add scroll behavior

### Layout

- [ ] Create DefaultLayout.vue (Navigation + RouterView + Footer)
- [ ] Add mobile responsiveness
- [ ] Test navigation flow

### Styling

- [ ] Port Tailwind classes to CSS with design tokens
- [ ] Ensure dark mode support (if needed)
- [ ] Responsive breakpoints

### Content

- [ ] Extract copy from React components
- [ ] Create content files (.md or .json)
- [ ] Integrate with i18n (if needed)

### Assets

- [ ] Copy images from /public
- [ ] Optimize images (WebP)
- [ ] Add favicon

### Testing

- [ ] Test all components in Histoire
- [ ] Test all pages in browser
- [ ] Cross-browser testing
- [ ] Mobile testing

### Performance

- [ ] Code splitting
- [ ] Lazy loading
- [ ] Image optimization
- [ ] Bundle size analysis

### Accessibility

- [ ] ARIA labels
- [ ] Keyboard navigation
- [ ] Screen reader testing
- [ ] Color contrast

### Final QA

- [ ] All links working
- [ ] All buttons functional
- [ ] Forms validate
- [ ] No console errors
- [ ] Lighthouse score >90

---

## 📈 Progress Tracking

**Total Components:** 140+

- [ ] Phase 1: 0/60 atoms complete
- [ ] Phase 2: 0/15 molecules complete
- [ ] Phase 3: 0/50+ organisms complete
- [ ] Phase 4: 0/7 pages complete
- [ ] Phase 5: 0/10 integration tasks complete

---

## 🚀 Getting Started

### Step 1: Set up workspace

```bash
cd /home/vince/Projects/llama-orch
pnpm install
```

### Step 2: Start Histoire (for component development)

```bash
cd frontend/libs/storybook
pnpm story:dev
```

### Step 3: Start building atoms

Pick a component from Phase 1, Priority 1 and:

1. Create component file in `storybook/stories/atoms/ComponentName/`
2. Create story file `ComponentName.story.ts`
3. Test in Histoire
4. Check off in this TODO list
5. Commit with signature: `// Created by: TEAM-FE-XXX`

### Step 4: Move to molecules, then organisms

Follow the same pattern for each phase.

### Step 5: Assemble pages

Import completed organisms into page views.

---

## 📝 Component Template

```vue
<!-- Created by: TEAM-FE-XXX -->
<script setup lang="ts">
interface Props {
  // Define props
}

const props = defineProps<Props>()
</script>

<template>
  <div class="component-name">
    <!-- Component markup -->
  </div>
</template>

<style scoped>
.component-name {
  /* Use design tokens */
  color: var(--color-text);
}
</style>
```

---

## 📚 Reference Files

**Source:** `/home/vince/Projects/llama-orch/frontend/reference/v0/`
**Target:** `/home/vince/Projects/llama-orch/frontend/bin/commercial-frontend-v2/`
**Storybook:** `/home/vince/Projects/llama-orch/frontend/libs/storybook/`

---

**Ready to start porting! Begin with Phase 1, Priority 1 atoms.** 🚀

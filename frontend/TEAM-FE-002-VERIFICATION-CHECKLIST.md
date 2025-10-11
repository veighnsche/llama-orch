# TEAM-FE-002 Verification Checklist

**Before marking complete, verify all items below:**

---

## 🧪 Testing Verification

### Histoire (Storybook)
```bash
cd /home/vince/Projects/llama-orch/frontend/libs/storybook
pnpm story:dev
# Open: http://localhost:6006
```

**Check:**
- [ ] Badge atom shows 5 variants (Default, Secondary, Destructive, Outline, Most Popular)
- [ ] PricingCard molecule shows 3 variants (Home/Lab, Team, Enterprise)
- [ ] PricingHero organism displays correctly
- [ ] PricingTiers organism shows 3 cards in grid
- [ ] PricingComparisonTable organism shows table with styling
- [ ] PricingFAQ organism shows 6 FAQ items
- [ ] All components have proper Tailwind styling (colors, spacing, borders)
- [ ] Team tier card has amber background/border
- [ ] "Most Popular" badge is visible on Team card

### Vue App
```bash
cd /home/vince/Projects/llama-orch/frontend/bin/commercial-frontend
pnpm dev
# Open: http://localhost:5173/pricing
```

**Check:**
- [ ] /pricing route loads without errors
- [ ] Hero section has gradient background (slate-950 to slate-900)
- [ ] "Scale When Ready" text is amber-500
- [ ] Three pricing cards display in grid
- [ ] Middle card (Team) is highlighted with amber styling
- [ ] "Most Popular" badge appears on Team card
- [ ] Feature comparison table renders correctly
- [ ] Check/X icons display properly
- [ ] FAQ section shows all 6 questions
- [ ] Page is responsive (test mobile, tablet, desktop)

### React Reference (for comparison)
```bash
cd /home/vince/Projects/llama-orch/frontend/reference/v0
pnpm dev
# Open: http://localhost:3000/pricing
```

**Check:**
- [ ] Vue version matches React version visually
- [ ] All text content is identical
- [ ] All colors match
- [ ] All spacing matches
- [ ] Layout is identical

---

## 📁 File Structure Verification

### Component Files
- [ ] `/frontend/libs/storybook/stories/atoms/Badge/Badge.vue` exists
- [ ] `/frontend/libs/storybook/stories/atoms/Badge/Badge.story.vue` exists
- [ ] `/frontend/libs/storybook/stories/molecules/PricingCard/PricingCard.vue` exists
- [ ] `/frontend/libs/storybook/stories/molecules/PricingCard/PricingCard.story.vue` exists
- [ ] `/frontend/libs/storybook/stories/organisms/PricingHero/PricingHero.vue` exists
- [ ] `/frontend/libs/storybook/stories/organisms/PricingHero/PricingHero.story.vue` exists
- [ ] `/frontend/libs/storybook/stories/organisms/PricingTiers/PricingTiers.vue` exists
- [ ] `/frontend/libs/storybook/stories/organisms/PricingTiers/PricingTiers.story.vue` exists
- [ ] `/frontend/libs/storybook/stories/organisms/PricingComparisonTable/PricingComparisonTable.vue` exists
- [ ] `/frontend/libs/storybook/stories/organisms/PricingComparisonTable/PricingComparisonTable.story.vue` exists
- [ ] `/frontend/libs/storybook/stories/organisms/PricingFAQ/PricingFAQ.vue` exists
- [ ] `/frontend/libs/storybook/stories/organisms/PricingFAQ/PricingFAQ.story.vue` exists
- [ ] `/frontend/bin/commercial-frontend/src/views/PricingView.vue` exists

### Configuration Files
- [ ] `/frontend/libs/storybook/styles/tokens.css` has `@import "tailwindcss";`
- [ ] `/frontend/libs/storybook/histoire.config.ts` has PostCSS config with tailwindcss plugin
- [ ] `/frontend/libs/storybook/stories/index.ts` exports Badge
- [ ] `/frontend/bin/commercial-frontend/src/router/index.ts` has /pricing route

### Documentation Files
- [ ] `/frontend/TEAM-FE-002-COMPLETE.md` exists
- [ ] `/frontend/TEAM-FE-002-FINAL-SUMMARY.md` exists
- [ ] `/frontend/libs/storybook/.handoffs/TEAM-FE-003-HANDOFF.md` exists
- [ ] `/frontend/TEAM-FE-002-VERIFICATION-CHECKLIST.md` exists (this file)

---

## 🎨 Code Quality Verification

### Team Signatures
- [ ] All new `.vue` files have `<!-- Created by: TEAM-FE-002 -->`
- [ ] Modified files have `// TEAM-FE-002: description` comments

### No TODOs
- [ ] No TODO comments in Badge.vue
- [ ] No TODO comments in PricingCard.vue
- [ ] No TODO comments in PricingHero.vue
- [ ] No TODO comments in PricingTiers.vue
- [ ] No TODO comments in PricingComparisonTable.vue
- [ ] No TODO comments in PricingFAQ.vue
- [ ] No TODO comments in PricingView.vue

### Exports
- [ ] Badge exported in `stories/index.ts`
- [ ] PricingCard already exported (was scaffolded)
- [ ] All organisms already exported (were scaffolded)

### Story Format
- [ ] All stories use `.story.vue` format (NOT `.story.ts`)
- [ ] All stories use `<Story>` and `<Variant>` components
- [ ] No `.story.ts` files remain

---

## 🔧 Infrastructure Verification

### Tailwind CSS v4
- [ ] `@import "tailwindcss";` in tokens.css
- [ ] PostCSS plugin configured in histoire.config.ts
- [ ] Tailwind classes work in Histoire
- [ ] Tailwind classes work in Vue app

### Histoire Configuration
- [ ] histoire.config.ts imports `@tailwindcss/postcss`
- [ ] histoire.setup.ts imports `./styles/tokens.css`
- [ ] Histoire dev server starts without errors
- [ ] Stories appear in Histoire sidebar

---

## 📊 Content Verification

### Pricing Tiers
- [ ] Home/Lab: $0 forever
- [ ] Team: €99 /month, "5-10 developers"
- [ ] Enterprise: Custom, "Contact sales"

### Feature Lists
- [ ] Home/Lab has 7 features
- [ ] Team has 7 features (starts with "Everything in Home/Lab")
- [ ] Enterprise has 7 features (starts with "Everything in Team")

### Comparison Table
- [ ] 11 feature rows
- [ ] Home/Lab column
- [ ] Team column (highlighted with bg-amber-50)
- [ ] Enterprise column
- [ ] Check icons for available features
- [ ] X icons for unavailable features

### FAQ Section
- [ ] 6 FAQ items
- [ ] "Is the free tier really free forever?"
- [ ] "What's the difference between tiers?"
- [ ] "Can I upgrade or downgrade anytime?"
- [ ] "Do you offer discounts for non-profits?"
- [ ] "What payment methods do you accept?"
- [ ] "Is there a trial period?"

---

## 🎯 Visual Verification

### Colors
- [ ] Hero gradient: slate-950 to slate-900
- [ ] Accent color: amber-500
- [ ] Team card background: amber-50
- [ ] Team card border: amber-500
- [ ] Check icons: green-600
- [ ] X icons: slate-300
- [ ] Text: slate-900 (headings), slate-600 (body)

### Spacing
- [ ] Hero section: py-24
- [ ] Pricing tiers: py-24
- [ ] Comparison table: py-24
- [ ] FAQ section: py-24
- [ ] Card padding: p-8
- [ ] Grid gap: gap-8

### Responsive
- [ ] Mobile: Single column
- [ ] Tablet: 3-column grid (md:grid-cols-3)
- [ ] Desktop: Max-width containers

---

## 📝 Documentation Verification

### TEAM-FE-003-HANDOFF.md
- [ ] Has critical lessons learned section
- [ ] Has Histoire story format examples
- [ ] Has Tailwind v4 setup instructions
- [ ] Has component development workflow
- [ ] Has list of remaining pages to implement
- [ ] Has code examples
- [ ] Has common mistakes section
- [ ] Has testing commands
- [ ] Has success criteria

### TEAM-FE-002-COMPLETE.md
- [ ] Lists all deliverables
- [ ] Documents issues found and fixed
- [ ] Has component architecture diagram
- [ ] Has comparison with React reference
- [ ] Has testing instructions

---

## ✅ Final Checklist

- [ ] All components render correctly in Histoire
- [ ] All components render correctly in Vue app
- [ ] Visual parity with React reference
- [ ] No console errors in browser
- [ ] No TypeScript errors
- [ ] No linting errors
- [ ] All files have team signatures
- [ ] No TODO comments
- [ ] All documentation complete
- [ ] Handoff document created
- [ ] Infrastructure fixes documented

---

## 🚀 Ready for Handoff?

If all items above are checked, TEAM-FE-002 work is complete and ready for TEAM-FE-003!

**Final Steps:**
1. Commit all changes
2. Push to repository
3. Notify TEAM-FE-003
4. Point them to `/frontend/libs/storybook/.handoffs/TEAM-FE-003-HANDOFF.md`

---

**Verification completed by:** _________________  
**Date:** _________________  
**Status:** _________________

```
// Created by: TEAM-FE-002
// Purpose: Verification checklist
// Date: 2025-10-11
```

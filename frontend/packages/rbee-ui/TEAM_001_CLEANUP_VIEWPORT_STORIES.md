# TEAM-001: CLEANUP VIEWPORT STORIES

**Mission:** Remove ALL nonsensical viewport-only stories (MobileView, TabletView)  
**Components:** 10 organisms  
**Estimated Time:** 3-4 hours  
**Priority:** P0 (MUST BE DONE FIRST)

---

## ⚠️ THE PROBLEM

Multiple story files have "MobileView" and "TabletView" stories that DO NOTHING except change the viewport parameter. These are **NOT VARIANTS**—they show the exact same content, just in a smaller frame.

**WHY THIS IS WRONG:**
- Storybook has a viewport toolbar that lets users switch viewports
- These stories waste space and clutter the sidebar
- They provide ZERO additional value
- They inflate story counts artificially

**Example of GARBAGE stories:**

```typescript
export const MobileView: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
  },
}

export const TabletView: Story = {
  parameters: {
    viewport: {
      defaultViewport: 'tablet',
    },
  },
}
```

❌ **DELETE THESE!** They're useless. Users can click the viewport button in Storybook.

---

## 📋 COMPONENTS TO CLEAN

### 1. AudienceSelector
**File:** `src/organisms/AudienceSelector/AudienceSelector.stories.tsx`
- ❌ Remove: `MobileView` story
- ❌ Remove: `TabletView` story
- ✅ Keep: Default, any ACTUAL variants (different content/props)

### 2. CtaSection
**File:** `src/organisms/CtaSection/CtaSection.stories.tsx`
- ❌ Remove: `MobileView` story
- ✅ Keep: Default, SingleButton, WithGradient, LeftAligned, MinimalWithEyebrow, AllVariants

### 3. EmailCapture
**File:** `src/organisms/EmailCapture/EmailCapture.stories.tsx`
- ❌ Remove: `MobileView` story (lines 95-106)
- ❌ Remove: `TabletView` story (lines 108-119)
- ✅ Keep: Default, InteractiveDemo, FormStates, WithPageContext

### 4. FaqSection
**File:** `src/organisms/FaqSection/FaqSection.stories.tsx`
- ❌ Remove: `MobileView` story (lines 188-199)
- ❌ Remove: `TabletView` story (lines 201-212)
- ✅ Keep: Default, WithoutSupportCard, CustomContent, InteractiveSearch, CategoryFiltering, SupportCardHighlight, SEOFeatures

### 5. Footer
**File:** `src/organisms/Footer/Footer.stories.tsx`
- ❌ Remove: `MobileView` story
- ❌ Remove: `TabletView` story
- ✅ Keep: Default, WithPageContent, NewsletterFormFocus, SocialLinksHighlight, LinkOrganization

### 6. HeroSection
**File:** `src/organisms/HeroSection/HeroSection.stories.tsx`
- ❌ Remove: `MobileView` story
- ❌ Remove: `TabletView` story
- ✅ Keep: Default, WithScrollIndicator

### 7. Navigation
**File:** `src/organisms/Navigation/Navigation.stories.tsx`
- ❌ Remove: `MobileView` story
- ❌ Remove: `TabletView` story
- ✅ Keep: Default, WithScrolledPage, FocusStates

### 8. PricingSection
**File:** `src/organisms/PricingSection/PricingSection.stories.tsx`
- ❌ Remove: `MobileView` story
- ❌ Remove: `TabletView` story
- ✅ Keep: Default, PricingPage, MinimalVariant, WithoutImage, CustomContent, InteractiveBillingToggle, PricingFeatures

### 9. ProblemSection
**File:** `src/organisms/ProblemSection/ProblemSection.stories.tsx`
- ❌ Remove: `MobileView` story (lines 204-215)
- ❌ Remove: `TabletView` story (lines 217-228)
- ✅ Keep: Default, WithoutCTA, CustomProblems, ToneVariations

### 10. WhatIsRbee
**File:** `src/organisms/WhatIsRbee/WhatIsRbee.stories.tsx`
- ❌ Remove: `MobileView` story
- ❌ Remove: `TabletView` story
- ✅ Keep: Default, any ACTUAL variants

---

## 🎯 EXECUTION PLAN

### Step 1: Read Each Story File (5 min per file)
1. Open the story file
2. Find all `MobileView` and `TabletView` exports
3. Verify they ONLY change viewport parameter
4. Mark them for deletion

### Step 2: Delete Viewport Stories (2 min per file)
1. Delete the entire story export (including JSDoc if present)
2. **DO NOT** delete stories that:
   - Change actual props/content
   - Demonstrate different behaviors
   - Show specific mobile-only features (like hamburger menu)

### Step 3: Verify (5 min per file)
1. Run Storybook: `pnpm storybook`
2. Navigate to the component
3. Verify remaining stories still work
4. Use viewport toolbar to test responsive behavior
5. Check no TypeScript errors

### Step 4: Commit Each File (1 min per file)
```bash
git add src/organisms/ComponentName/ComponentName.stories.tsx
git commit -m "cleanup(storybook): remove viewport-only stories from ComponentName

- Removed MobileView and TabletView stories
- These provided no value beyond Storybook's viewport toolbar
- Kept all legitimate variant stories"
```

---

## ✅ QUALITY CHECKLIST

For EACH component:
- [ ] Read story file completely
- [ ] Identified all viewport-only stories
- [ ] Deleted MobileView story (if viewport-only)
- [ ] Deleted TabletView story (if viewport-only)
- [ ] Kept all legitimate variant stories
- [ ] Verified in Storybook (no errors)
- [ ] Tested responsive behavior with viewport toolbar
- [ ] Committed with descriptive message

**Total checklist items: 80 (8 per component × 10 components)**

---

## 🚨 CRITICAL RULES

### ✅ DELETE IF:
- Story ONLY changes `parameters.viewport`
- Story has SAME args as Default
- Story shows SAME content, just smaller
- Description says "Mobile view" or "Tablet view" without elaboration

### ❌ KEEP IF:
- Story changes props/content
- Story demonstrates mobile-specific behavior (hamburger menu, collapsed nav)
- Story shows different layout that's NOT just responsive
- Story has different content for mobile users

### 🤔 WHEN IN DOUBT:
Ask: "Does this story show something I CAN'T see by clicking the viewport button?"
- **NO** → DELETE
- **YES** → KEEP

---

## 📊 PROGRESS TRACKER

Update as you complete each component:

- [ ] AudienceSelector ✅ Cleaned
- [ ] CtaSection ✅ Cleaned
- [ ] EmailCapture ✅ Cleaned
- [ ] FaqSection ✅ Cleaned
- [ ] Footer ✅ Cleaned
- [ ] HeroSection ✅ Cleaned
- [ ] Navigation ✅ Cleaned
- [ ] PricingSection ✅ Cleaned
- [ ] ProblemSection ✅ Cleaned
- [ ] WhatIsRbee ✅ Cleaned

**Completion: 0/10 (0%)**

---

## 🎯 EXPECTED RESULTS

### Before Cleanup:
- EmailCapture: 6 stories (2 are viewport-only)
- FaqSection: 9 stories (2 are viewport-only)
- ProblemSection: 5 stories (2 are viewport-only)
- **Total: ~60 stories** (20 are garbage)

### After Cleanup:
- EmailCapture: 4 stories (all legitimate)
- FaqSection: 7 stories (all legitimate)
- ProblemSection: 3 stories (all legitimate)
- **Total: ~40 stories** (all valuable)

**33% reduction in clutter! 🎉**

---

## 🚀 HANDOFF TO NEXT TEAMS

After this cleanup, the remaining teams will:
1. Create stories for components WITHOUT stories
2. Add **MARKETING/COPY documentation** to organisms
3. Document how components are used across different pages
4. Show REAL variants (different copy, different CTAs, different audiences)

**Your cleanup makes their work possible. Do it right!**

---

**START TIME:** [Fill in]  
**END TIME:** [Fill in]  
**TEAM MEMBERS:** [Fill in]  
**STATUS:** 🔴 NOT STARTED

---

## 📞 QUESTIONS?

**Q: What if a component has a story called "MobileLayout" that changes props?**  
A: KEEP IT. It's a legitimate variant, not just a viewport change.

**Q: What if MobileView shows a hamburger menu?**  
A: KEEP IT. That's a real behavior difference, not just responsive CSS.

**Q: What if I'm unsure?**  
A: Keep it and add a comment. Better to keep a borderline story than delete a good one.

**Q: Should I update documentation after cleanup?**  
A: Yes! Remove references to deleted stories in component descriptions.

---

**LET'S CLEAN THIS UP! 🧹**

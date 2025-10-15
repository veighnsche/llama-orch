# TEAM-006: ATOMS & MOLECULES

**Mission:** Create stories for atoms and molecules used in commercial site  
**Components:** 8 components (2 atoms, 6 molecules/misc)  
**Estimated Time:** 8-12 hours  
**Priority:** P3 (Low Priority - Foundational)

---

## 🎯 MISSION BRIEFING

You're documenting the **foundational components** - atoms and molecules that organisms are built from. These are smaller, reusable pieces.

### KEY CHARACTERISTICS:
- **Atoms:** Single-purpose, minimal components (icons, buttons, badges)
- **Molecules:** Small compositions (brand logo, cards, simple layouts)
- **Usage:** Often INDIRECT (used by organisms, not imported directly by pages)

### CRITICAL REQUIREMENTS:
1. ✅ **Show ALL variants** - Atoms especially have many size/color/state variants
2. ✅ **Document composition** - For molecules, show what atoms they're built from
3. ✅ **NO viewport stories**
4. ✅ **Keep it simple** - These are NOT complex organisms
5. ✅ **Focus on props/API** - Document every prop clearly

---

## 📋 YOUR COMPONENTS

## SECTION A: ATOMS (2 components)

### 1. GitHubIcon ✅
**File:** `src/atoms/GitHubIcon/GitHubIcon.stories.tsx`  
**Status:** ✅ STORY EXISTS  
**Used in:** Developers page CTA (as iconLeft for secondary button)

**YOUR TASKS:**
- ✅ REVIEW EXISTING STORY - No changes needed unless quality issues
- ✅ Verify it follows documentation standard
- ✅ Check: Are all size variants shown?
- ✅ Check: Are color variants shown?
- ✅ Check: Is usage documented (where it's used in commercial site)?

**If issues found:**
- Add missing variants
- Add usage documentation
- Remove viewport stories if present

---

### 2. DiscordIcon ✅
**File:** `src/atoms/DiscordIcon/DiscordIcon.stories.tsx`  
**Status:** ✅ STORY EXISTS  
**Used in:** Footer, possibly social links

**YOUR TASKS:**
- ✅ REVIEW EXISTING STORY - No changes needed unless quality issues
- ✅ Verify it follows documentation standard
- ✅ Check: Are all size variants shown?
- ✅ Check: Brand color variant shown? (Discord's #5865F2)
- ✅ Check: Is usage documented?

**If issues found:**
- Add missing variants
- Add usage documentation
- Remove viewport stories if present

---

## SECTION B: MOLECULES (6 components)

### 3. BrandLogo
**File:** `src/molecules/BrandLogo/BrandLogo.stories.tsx`  
**Status:** ✅ STORY EXISTS  
**Used in:** Navigation, Footer

**YOUR TASKS:**
- ✅ REVIEW AND ENHANCE existing story
- ✅ Add composition docs:
  - "Combines BrandMark + BrandWordmark (optional)"
  - "Used in Navigation and Footer"
- ✅ Verify variants:
  - Logo with wordmark
  - Logo only (icon-only mode)
  - Different sizes (if applicable)
  - Light/dark mode compatibility
- ✅ Add story: `NavigationContext` - show how it appears in nav
- ✅ Add story: `FooterContext` - show how it appears in footer
- ✅ Document props thoroughly

---

### 4. BrandMark ✅
**File:** `src/atoms/BrandMark/BrandMark.stories.tsx`  
**Status:** ✅ STORY EXISTS  
**Used in:** Part of BrandLogo

**YOUR TASKS:**
- ✅ REVIEW EXISTING STORY
- ✅ Document composition: "SVG icon of rbee bee logo"
- ✅ Verify size variants shown
- ✅ Document usage: "Used in BrandLogo component"
- ✅ Check if it needs enhancement, otherwise leave as-is

---

### 5. Card
**File:** `src/molecules/Card.stories.tsx`  
**Status:** ✅ STORY EXISTS  
**Used in:** Various organisms (features, use cases, problem cards, etc.)

**YOUR TASKS:**
- ✅ REVIEW AND ENHANCE existing story
- ✅ Add composition docs:
  - "Container component for card-based layouts"
  - "Used in ProblemSection, FeaturesSection, etc."
- ✅ Verify variants:
  - Default card
  - Card with header
  - Card with footer
  - Interactive card (hover states)
  - Different padding/spacing options
- ✅ Add story: `ProblemCardExample` - show problem section usage
- ✅ Add story: `FeatureCardExample` - show feature section usage
- ✅ Document all props and slots

---

### 6. TestimonialsSection (or TestimonialsRail)
**File:** `src/organisms/TestimonialsRail/TestimonialsRail.stories.tsx` OR similar (create if needed)  
**Status:** ❓ UNCLEAR - Used in Developers page line 14, check actual component
**Used in:** Developers page

**YOUR TASKS:**
- ✅ Find the actual component (grep for "TestimonialsSection")
- ✅ Create story if none exists
- ✅ Document the developers page usage (lines 30-60):
  - Title, testimonials array, stats array
  - Quote cards with avatar, author, role, quote
  - Stats with value, label, optional tone
- ✅ Create story: `DevelopersPageDefault` - exact developers page usage
- ✅ Create story: `WithoutStats` - testimonials only
- ✅ Create story: `StatsOnly` - stats rail only
- ✅ Document composition: What atoms/molecules are used?

---

### 7. HomeSolutionSection (SolutionSection)
**File:** Check if this is in `src/organisms/SolutionSection/` or a variant
**Status:** ❓ NEEDS INVESTIGATION  
**Used in:** Home page (lines 39-87)

**YOUR TASKS:**
- ✅ Find the actual component
- ✅ Check if story exists
- ✅ If no story: CREATE IT (this might belong in TEAM-002's work)
- ✅ Document the home page usage with topology diagram
- ✅ Create story showing multi-host GPU pool visualization

**NOTE:** If this is an organism (not molecule), coordinate with TEAM-002. It might already be in their scope.

---

### 8. CodeExamplesSection
**File:** `src/organisms/CodeExamplesSection/CodeExamplesSection.stories.tsx` (create if needed)  
**Status:** ❓ NEEDS INVESTIGATION  
**Used in:** Possibly used indirectly (check if it's part of DevelopersCodeExamples)

**YOUR TASKS:**
- ✅ Find the actual component
- ✅ Check if it's standalone or part of DevelopersCodeExamples
- ✅ If standalone: Create story showing code example presentation
- ✅ If part of Developers: Note that it's handled by TEAM-003

**NOTE:** This might be in TEAM-003's scope. Check before duplicating work.

---

## 🎯 STORY REQUIREMENTS (MANDATORY)

For EACH component, include:

### 1. Component Documentation
```markdown
## Overview
[What is this component? Single sentence.]

## Composition (for molecules)
This component is composed of:
- **[Atom/Component 1]**: [Purpose]
- **[Atom/Component 2]**: [Purpose]

## When to Use
- [Use case 1]
- [Use case 2]

## Variants
- **[Variant 1]**: [Description]
- **[Variant 2]**: [Description]

## Used In
- [Organism 1] (e.g., Navigation)
- [Organism 2] (e.g., Footer)
- [Page context] (e.g., Used in header across all pages)
```

### 2. Props Documentation (argTypes)
Every prop MUST be documented:
```typescript
argTypes: {
  propName: {
    control: 'text' | 'select' | 'boolean',
    description: 'Clear description of what this prop does',
    table: {
      type: { summary: 'string' },
      defaultValue: { summary: 'default' },
      category: 'Appearance' | 'Content' | 'Behavior',
    },
  },
}
```

### 3. Minimum Stories
- ✅ `Default` - Standard configuration
- ✅ `AllVariants` - Showcase all size/color/state variants (for atoms)
- ✅ `UsageContext` - Show how it's used in actual organisms (for molecules)

**Atoms should have MORE variants than organisms!** Show every size, color, state.

---

## ✅ QUALITY CHECKLIST

For EACH component:
- [ ] Story file exists (created or verified)
- [ ] Component overview documented
- [ ] Composition documented (for molecules)
- [ ] All props in argTypes with descriptions
- [ ] All variants shown (especially for atoms)
- [ ] Usage context documented
- [ ] Minimum 2-3 stories
- [ ] NO viewport stories
- [ ] Tested in Storybook
- [ ] Committed

**Total: 80 checklist items (10 per component × 8 components)**

---

## 📊 PROGRESS TRACKER

### Atoms (2 components - REVIEW ONLY)
- [ ] GitHubIcon ✅ Reviewed
- [ ] DiscordIcon ✅ Reviewed

### Molecules (6 components - ENHANCE/CREATE)
- [ ] BrandLogo ✅ Enhanced
- [ ] BrandMark ✅ Reviewed
- [ ] Card ✅ Enhanced
- [ ] TestimonialsSection ✅ Created/Enhanced
- [ ] HomeSolutionSection ✅ Investigated (may be organism)
- [ ] CodeExamplesSection ✅ Investigated (may be handled by TEAM-003)

**Completion: 0/8 (0%)**

---

## 🚨 COORDINATION WITH OTHER TEAMS

**IMPORTANT:** Some components in your list might actually be organisms:
1. **HomeSolutionSection** - If it's complex, TEAM-002 handles it
2. **CodeExamplesSection** - If part of Developers page, TEAM-003 handles it
3. **TestimonialsSection** - Check if it's organism or molecule

**Before starting work on these:**
1. Check if another team is handling it
2. If unsure, create the story but coordinate
3. Don't duplicate work!

---

## 🚀 COMMIT MESSAGES

```bash
# For reviewed atoms
git add src/atoms/ComponentName/ComponentName.stories.tsx
git commit -m "docs(storybook): review and enhance ComponentName atom

- Verified all variants present
- Added usage documentation
- Removed viewport stories (if any)
- Added missing size/color variants"

# For created/enhanced molecules
git add src/molecules/ComponentName/ComponentName.stories.tsx
git commit -m "docs(storybook): create/enhance ComponentName molecule

- Added complete story with composition docs
- Documented usage in organisms
- Created context stories showing real usage
- Added all variants"
```

---

## 📞 NOTES

**Atoms are SIMPLE:**
- Few props, many variants
- Show EVERY size, color, state
- Document thoroughly but keep it concise

**Molecules are SMALL COMPOSITIONS:**
- Document what atoms they're made of
- Show usage context (how organisms use them)
- More complex than atoms, simpler than organisms

**Don't overcomplicate:**
- These are NOT complex organisms
- Stories should be straightforward
- Focus on variants and usage

**If a component seems too complex:**
- It might be an organism, not a molecule
- Check with other teams
- Coordinate to avoid duplication

---

**START TIME:** [Fill in]  
**END TIME:** [Fill in]  
**TEAM MEMBERS:** [Fill in]  
**STATUS:** 🔴 NOT STARTED

---

**DOCUMENT THE BUILDING BLOCKS! 🧱**

# Storybook Stories Implementation Plan

**Version:** 1.0  
**Date:** 2025-10-14  
**Status:** READY FOR EXECUTION

---

## Executive Summary

This document provides a **complete, actionable plan** to create Storybook stories for all components currently used in the `@rbee/commercial` application.

**Key Principles:**
- ✅ **Only document USED components** (not all components in rbee-ui)
- ✅ **Dark/Light mode via Storybook toolbar** (not separate variants)
- ✅ **Comprehensive documentation** for each story
- ✅ **Multiphase execution** to manage workload

**Total Components: ~40 stories**  
**Estimated Time: 35-40 hours**  
**Timeline: 2-3 weeks**

---

## Phase 0: Discovery & Inventory

### Components Requiring Stories

**Atoms (2):**
- GitHubIcon
- DiscordIcon

**Organisms - Core (2):**
- Navigation (P0)
- Footer (P0)

**Organisms - Marketing (13):**
- HeroSection (P1)
- EmailCapture (P1)
- CTASection (P1)
- PricingSection (P1)
- FAQSection (P1)
- WhatIsRbee (P2)
- AudienceSelector (P2)
- ProblemSection (P2)
- SolutionSection (P2)
- HowItWorksSection (P2)
- FeaturesSection (P2)
- UseCasesSection (P2)
- ComparisonSection (P2)
- SocialProofSection (P2)
- TechnicalSection (P2)

**Organisms - Page Groups (23):**
- Enterprise/* (4 components - P3)
- Developers/* (4 components - P3)
- Features/* (4 components - P3)
- Pricing/* (2 components - P3)
- Providers/* (5 components - P3)
- UseCases/* (3 components - P3)

---

## Phase 1: Foundation Setup

### Task 1.1: Update Dark Mode Configuration

**File:** `.storybook/preview.ts`

Add theme decorator:

```typescript
import type { Preview, Decorator } from "@storybook/react";
import { useEffect } from "react";
import "../src/tokens/globals.css";

const withTheme: Decorator = (Story, context) => {
  const theme = context.globals.theme || 'light';
  
  useEffect(() => {
    document.documentElement.classList.remove('light', 'dark');
    document.documentElement.classList.add(theme);
  }, [theme]);

  return <Story />;
};

const preview: Preview = {
  decorators: [withTheme],
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
  globalTypes: {
    theme: {
      description: 'Global theme for components',
      defaultValue: 'light',
      toolbar: {
        title: 'Theme',
        icon: 'circlehollow',
        items: [
          { value: 'light', icon: 'sun', title: 'Light' },
          { value: 'dark', icon: 'moon', title: 'Dark' },
        ],
        dynamicTitle: true,
      },
    },
  },
};

export default preview;
```

### Task 1.2: Create Story Templates

**File:** `.storybook/templates/story-template.md`

```markdown
# Story Template Guide

## Atom Story Template

\`\`\`typescript
import type { Meta, StoryObj } from '@storybook/react';
import { ComponentName } from './ComponentName';

const meta = {
  title: 'Atoms/ComponentName',
  component: ComponentName,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'Component description here.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    // Configure props
  },
} satisfies Meta<typeof ComponentName>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};
\`\`\`

## Organism Story Template

\`\`\`typescript
import type { Meta, StoryObj } from '@storybook/react';
import { ComponentName } from './ComponentName';

const meta = {
  title: 'Organisms/ComponentName',
  component: ComponentName,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: \`
## Overview
[Description]

## When to Use
- Use case 1
- Use case 2

## Props
[Key props explanation]
        \`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ComponentName>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};
\`\`\`
```

---

## Phase 2: P0 Components (Critical)

### GitHubIcon Story

**File:** `src/atoms/GitHubIcon/GitHubIcon.stories.tsx`

**Stories to create:**
- Default
- Large
- Small
- Colored variants

### DiscordIcon Story

**File:** `src/atoms/DiscordIcon/DiscordIcon.stories.tsx`

Same structure as GitHubIcon.

### Navigation Story

**File:** `src/organisms/Navigation/Navigation.stories.tsx`

**Stories to create:**
- Default
- Mobile view
- Tablet view

**Key aspects:**
- Test theme toggle
- Test mobile menu
- Test responsive behavior

### Footer Story

**File:** `src/organisms/Footer/Footer.stories.tsx`

**Stories to create:**
- Default
- Mobile view

---

## Phase 3: P1 Components (High Priority)

### Components:
1. HeroSection
2. EmailCapture
3. CTASection
4. PricingSection
5. FAQSection

### Story Requirements for Each:

**Minimum stories:**
- Default (with realistic data)
- 1-2 variants
- Mobile view (if layout changes)

**Documentation must include:**
- Overview
- When to use
- Props explanation
- Usage examples

---

## Phase 4: P2 Components (Medium Priority)

### Components:
1. WhatIsRbee
2. AudienceSelector
3. ProblemSection
4. SolutionSection
5. HowItWorksSection
6. FeaturesSection
7. UseCasesSection
8. ComparisonSection
9. SocialProofSection
10. TechnicalSection

### Execution:
- 2-3 stories per component
- Focus on realistic data
- Test dark mode thoroughly

---

## Phase 5: P3 Components (Page-Specific)

### Component Groups:
- Enterprise/* (4)
- Developers/* (4)
- Features/* (4)
- Pricing/* (2)
- Providers/* (5)
- UseCases/* (3)

### Strategy:
- Link to general component stories
- Focus on unique aspects
- Minimal documentation (reference parent)

---

## Phase 6: Quality Assurance

### QA Checklist (per story):

**Visual:**
- [ ] Renders in light mode
- [ ] Renders in dark mode
- [ ] No layout breaks
- [ ] Good contrast

**Functional:**
- [ ] Interactive elements work
- [ ] Theme toggle works
- [ ] Responsive works
- [ ] No console errors

**Documentation:**
- [ ] Clear description
- [ ] Props documented
- [ ] Examples provided

---

## Phase 7: Documentation

### Deliverables:

1. **STORYBOOK_INDEX.md** - Complete list of all stories
2. **STORYBOOK_USAGE_GUIDE.md** - How to use Storybook
3. **STORYBOOK_CONTRIBUTION_GUIDE.md** - How to add stories
4. Update **STORYBOOK.md** with new info

---

## Execution Timeline

### Week 1
- Day 1: Phase 0 + Phase 1 (4 hours)
- Day 2: Phase 2 (3 hours)
- Day 3: Phase 3 (6 hours)

### Week 2
- Day 4-5: Phase 4 (8 hours)
- Day 6: Phase 5 Part 1 (5 hours)

### Week 3
- Day 7: Phase 5 Part 2 (5 hours)
- Day 8: Phase 6 QA (3 hours)
- Day 9: Phase 7 Docs (2 hours)

**Total: 36 hours over 9 days**

---

## Success Criteria

- [ ] All 40 components have stories
- [ ] Dark/light mode via toolbar only
- [ ] All stories documented
- [ ] QA passed for all stories
- [ ] Documentation complete
- [ ] No console errors
- [ ] Storybook loads <3 seconds

---

## Quick Start for Engineers

### To Start Working:

1. **Pick a component** from the phase you're assigned
2. **Read the component source** to understand props
3. **Copy the template** from `.storybook/templates/`
4. **Create the story file** next to the component
5. **Write 2-4 stories** with realistic data
6. **Test in Storybook** (light/dark, responsive)
7. **Check QA checklist**
8. **Commit**: `docs(storybook): add [ComponentName] story`

### Commands:

```bash
# Start Storybook
cd /home/vince/Projects/llama-orch/frontend/libs/rbee-ui
pnpm storybook

# Build Storybook
pnpm build-storybook
```

### Need Help?

- Check existing stories: `Button.stories.tsx`, `Badge.stories.tsx`
- Read templates in `.storybook/templates/`
- Reference this plan for requirements

---

## Notes

- **NO separate dark/light variants** - use toolbar only
- **Realistic data** - extract from actual page usage
- **Comprehensive docs** - future engineers will thank you
- **Test thoroughly** - both themes, responsive, interactions

**This is a lot of work, but it's structured and manageable. Follow the phases, check off the boxes, and we'll have world-class component documentation.**

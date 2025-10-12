# Atomic Design Structure

**Status:** ✅ Migrated  
**Date:** 2025-10-12

This project follows the **Atomic Design** methodology by Brad Frost for organizing UI components.

---

## Directory Structure

```
components/
├── atoms/           # 57 components - Basic building blocks
├── molecules/       # 26 components - Simple combinations
├── organisms/       # 23 components - Complex sections
├── templates/       # 0 components - Page layouts (to be added)
└── providers/       # 1 component - Context providers
```

---

## Atomic Hierarchy

### 🔬 Atoms (57 components)

**Basic UI elements that cannot be broken down further.**

Located in: `components/atoms/`

Examples:
- `Button` - Clickable button element
- `Input` - Text input field
- `Label` - Form label
- `Badge` - Status badge
- `Card` - Container card
- `Avatar` - User avatar
- `Separator` - Visual divider

**All shadcn/ui components** are atoms.

**Import:**
```typescript
import { Button, Input, Label } from '@/components/atoms';
// or
import { Button } from '@/components/atoms/Button/Button';
```

---

### 🧪 Molecules (26 components)

**Simple combinations of 2-3 atoms functioning together.**

Located in: `components/molecules/`

Examples:
- `ThemeToggle` - Button + Icon for theme switching
- `FeatureCard` - Card + Icon + Text for features
- `TestimonialCard` - Card + Avatar + Text for testimonials
- `CodeBlock` - Pre + Code + Syntax highlighting
- `TerminalWindow` - Container + Code + Header
- `ProgressBar` - Container + Bar + Label
- `NavLink` - Link + Icon + Active state

**Import:**
```typescript
import { ThemeToggle, FeatureCard } from '@/components/molecules';
// or
import { ThemeToggle } from '@/components/molecules/ThemeToggle/ThemeToggle';
```

---

### 🦠 Organisms (23 components)

**Complex UI sections composed of molecules and atoms.**

Located in: `components/organisms/`

#### Top-Level Organisms (17 components)
- `Navigation` - Site header with menu
- `Footer` - Site footer with links
- `HeroSection` - Homepage hero
- `EmailCapture` - Email signup form
- `WhatIsRbee` - Product explanation
- `AudienceSelector` - Target audience cards
- `ProblemSection` - Problem statement
- `SolutionSection` - Solution overview
- `HowItWorksSection` - Process explanation
- `FeaturesSection` - Feature showcase
- `UseCasesSection` - Use case examples
- `ComparisonSection` - Comparison table
- `PricingSection` - Pricing tiers
- `SocialProofSection` - Testimonials/stats
- `TechnicalSection` - Technical details
- `FAQSection` - Frequently asked questions
- `CTASection` - Call-to-action

#### Feature-Specific Organisms (6 directories)
- `Developers/` - Developer-focused components
- `Enterprise/` - Enterprise-focused components
- `Features/` - Feature detail components
- `Pricing/` - Pricing detail components
- `Providers/` - GPU provider components
- `UseCases/` - Use case detail components

**Import:**
```typescript
import { Navigation, Footer, HeroSection } from '@/components/organisms';
// or
import { Navigation } from '@/components/organisms/Navigation/Navigation';
// or (feature-specific)
import { DevelopersHero } from '@/components/organisms/Developers/developers-hero';
```

---

### 📄 Templates (0 components)

**Page-level layouts that place organisms into structure.**

Located in: `components/templates/`

**To be created:**
- `MarketingLayout` - Standard marketing page layout
- `PageLayout` - Generic page wrapper
- `SectionLayout` - Reusable section wrapper

**Import (future):**
```typescript
import { MarketingLayout } from '@/components/templates';
```

---

### 🔌 Providers (1 component)

**Context providers and utilities (not part of atomic hierarchy).**

Located in: `components/providers/`

- `ThemeProvider` - Dark/light theme context

**Import:**
```typescript
import { ThemeProvider } from '@/components/providers';
```

---

## Component Classification Rules

### How to Classify a Component

#### Is it an ATOM?
- ✅ Cannot be broken down further
- ✅ Single responsibility
- ✅ No internal state (or minimal)
- ✅ Highly reusable
- **Examples:** Button, Input, Badge, Icon

#### Is it a MOLECULE?
- ✅ Combines 2-3 atoms
- ✅ Simple functionality
- ✅ Reusable across contexts
- ✅ Single purpose
- **Examples:** SearchForm (Label + Input + Button), FeatureCard (Icon + Title + Description)

#### Is it an ORGANISM?
- ✅ Complex component
- ✅ Combines multiple molecules/atoms
- ✅ Forms a distinct section
- ✅ May have significant state
- **Examples:** Navigation (Logo + Menu + Search + Theme), Footer (Columns + Links + Social)

#### Is it a TEMPLATE?
- ✅ Page-level layout
- ✅ Defines content structure
- ✅ No final content
- ✅ Reusable across pages
- **Examples:** MarketingLayout (Nav + Content + Footer)

---

## Import Patterns

### Barrel Exports (Recommended)

```typescript
// Import multiple atoms
import { Button, Input, Label, Card } from '@/components/atoms';

// Import multiple molecules
import { ThemeToggle, FeatureCard } from '@/components/molecules';

// Import multiple organisms
import { Navigation, Footer, HeroSection } from '@/components/organisms';
```

### Direct Imports (When needed)

```typescript
// Import specific component
import { Button } from '@/components/atoms/Button/Button';
import { ThemeToggle } from '@/components/molecules/ThemeToggle/ThemeToggle';
import { Navigation } from '@/components/organisms/Navigation/Navigation';
```

### Feature-Specific Imports

```typescript
// Import from feature directory
import { DevelopersHero } from '@/components/organisms/Developers/developers-hero';
import { EnterpriseCTA } from '@/components/organisms/Enterprise/enterprise-cta';
```

---

## File Naming Conventions

### Directory Structure
```
ComponentName/
├── ComponentName.tsx    # Main component file
└── index.ts            # Optional re-export
```

### Examples
```
atoms/Button/Button.tsx
molecules/ThemeToggle/ThemeToggle.tsx
organisms/Navigation/Navigation.tsx
```

### Feature Directories
```
organisms/Developers/
├── developers-hero.tsx
├── developers-features.tsx
└── developers-cta.tsx
```

---

## Benefits of Atomic Design

### 1. Clear Hierarchy
- Instantly understand component complexity
- Easy to find components
- Logical organization

### 2. Reusability
- Atoms are maximally reusable
- Molecules combine atoms in standard ways
- Organisms can be dropped into any template

### 3. Maintainability
- Changes to atoms cascade properly
- Easy to test at each level
- Clear dependencies

### 4. Scalability
- New features follow established patterns
- Consistent structure as team grows
- Easy onboarding

### 5. Collaboration
- Designers and developers speak same language
- Clear component boundaries
- Easy to parallelize work

---

## Adding New Components

### 1. Classify the Component
Ask: "Is this an atom, molecule, or organism?"

### 2. Create Directory
```bash
mkdir -p components/atoms/NewComponent
# or
mkdir -p components/molecules/NewComponent
# or
mkdir -p components/organisms/NewComponent
```

### 3. Create Component File
```bash
touch components/atoms/NewComponent/NewComponent.tsx
```

### 4. Add to Barrel Export
```typescript
// components/atoms/index.ts
export * from './NewComponent/NewComponent';
```

### 5. Use the Component
```typescript
import { NewComponent } from '@/components/atoms';
```

---

## Migration Summary

### What Was Migrated

- ✅ **57 atoms** - All UI components from `components/ui/`
- ✅ **26 molecules** - From `components/primitives/` and `theme-toggle.tsx`
- ✅ **23 organisms** - All section components and feature directories
- ✅ **1 provider** - ThemeProvider
- ✅ **63 files updated** - All import paths fixed
- ✅ **Build verified** - Zero errors

### Before
```
components/
├── ui/                  # Flat structure
├── primitives/          # Mixed complexity
├── navigation.tsx       # No hierarchy
├── footer.tsx
└── *-section.tsx
```

### After
```
components/
├── atoms/               # Clear hierarchy
├── molecules/
├── organisms/
├── templates/
└── providers/
```

---

## Resources

- [Atomic Design by Brad Frost](https://atomicdesign.bradfrost.com/)
- [Pattern Lab](https://patternlab.io/)
- [Storybook Atomic Design](https://storybook.js.org/blog/atomic-design-with-storybook/)

---

## Maintenance

### Keep the Hierarchy Clean
- ❌ Don't import molecules in atoms
- ❌ Don't import organisms in molecules
- ✅ Only import down the hierarchy (atoms → molecules → organisms)

### Review Classifications
- Periodically review if components are in the right category
- Move components if they grow in complexity
- Split organisms if they become too large

### Update Documentation
- Keep this file updated as structure evolves
- Document new patterns
- Share learnings with team

---

**The atomic design structure is now in place and ready for development!** 🎉

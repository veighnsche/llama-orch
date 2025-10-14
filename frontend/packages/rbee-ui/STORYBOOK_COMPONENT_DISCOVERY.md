# Storybook Component Discovery Report

**Date:** 2025-10-14  
**Purpose:** Identify all components from `@rbee/ui` used in commercial frontend

---

## Discovery Method

```bash
# Scan all TypeScript files in commercial app for @rbee/ui imports
grep -r "from '@rbee/ui" /home/vince/Projects/llama-orch/frontend/bin/commercial/app/ \
  --include="*.tsx" --include="*.ts" | \
  grep -oP "from '@rbee/ui/[^']*'" | \
  sort -u
```

---

## Results: Components Used in Commercial App

### Atoms (2 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| GitHubIcon | `@rbee/ui/atoms/GitHubIcon` | developers/page.tsx |
| DiscordIcon | `@rbee/ui/atoms/DiscordIcon` | (likely footer/navigation) |

### Organisms - Core Layout (2 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| Navigation | `@rbee/ui/organisms/Navigation` | layout.tsx |
| Footer | `@rbee/ui/organisms/Footer` | layout.tsx, multiple pages |

### Organisms - Marketing Sections (15 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| HeroSection | `@rbee/ui/organisms/HeroSection` | page.tsx (home) |
| WhatIsRbee | `@rbee/ui/organisms/WhatIsRbee` | page.tsx (home) |
| AudienceSelector | `@rbee/ui/organisms/AudienceSelector` | page.tsx (home) |
| ProblemSection | `@rbee/ui/organisms/ProblemSection` | page.tsx (home) |
| SolutionSection | `@rbee/ui/organisms/SolutionSection` | page.tsx (home) |
| HowItWorksSection | `@rbee/ui/organisms/HowItWorksSection` | page.tsx (home) |
| FeaturesSection | `@rbee/ui/organisms/FeaturesSection` | page.tsx (home) |
| UseCasesSection | `@rbee/ui/organisms/UseCasesSection` | page.tsx (home) |
| ComparisonSection | `@rbee/ui/organisms/ComparisonSection` | page.tsx (home) |
| PricingSection | `@rbee/ui/organisms/PricingSection` | page.tsx (home), pricing/page.tsx, developers/page.tsx |
| SocialProofSection | `@rbee/ui/organisms/SocialProofSection` | page.tsx (home) |
| TestimonialsSection | `@rbee/ui/organisms/SocialProofSection` | developers/page.tsx |
| TechnicalSection | `@rbee/ui/organisms/TechnicalSection` | page.tsx (home) |
| FAQSection | `@rbee/ui/organisms/FaqSection` | page.tsx (home), pricing/page.tsx |
| CTASection | `@rbee/ui/organisms/CtaSection` | page.tsx (home), developers/page.tsx |
| EmailCapture | `@rbee/ui/organisms/EmailCapture` | Multiple pages |

### Organisms - Enterprise Group (4 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| EnterpriseHero | `@rbee/ui/organisms/Enterprise` | enterprise/page.tsx |
| EnterpriseFeatures | `@rbee/ui/organisms/Enterprise` | enterprise/page.tsx |
| EnterpriseTestimonials | `@rbee/ui/organisms/Enterprise` | enterprise/page.tsx |
| EnterpriseCTA | `@rbee/ui/organisms/Enterprise` | enterprise/page.tsx |

### Organisms - Developers Group (4 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| DevelopersHero | `@rbee/ui/organisms/Developers` | developers/page.tsx |
| DevelopersFeatures | `@rbee/ui/organisms/Developers` | developers/page.tsx |
| DevelopersUseCases | `@rbee/ui/organisms/Developers` | developers/page.tsx |
| DevelopersCodeExamples | `@rbee/ui/organisms/Developers` | developers/page.tsx |

### Organisms - Features Group (4 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| FeaturesHero | `@rbee/ui/organisms/Features` | features/page.tsx |
| RealTimeProgress | `@rbee/ui/organisms/Features` | features/page.tsx |
| SecurityIsolation | `@rbee/ui/organisms/Features` | features/page.tsx |
| AdditionalFeaturesGrid | `@rbee/ui/organisms/Features` | features/page.tsx |

### Organisms - Pricing Group (2 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| PricingHero | `@rbee/ui/organisms/Pricing` | pricing/page.tsx |
| PricingComparison | `@rbee/ui/organisms/Pricing` | pricing/page.tsx |

### Organisms - Providers Group (5 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| ProvidersHero | `@rbee/ui/organisms/Providers` | gpu-providers/page.tsx |
| ProvidersFeatures | `@rbee/ui/organisms/Providers` | gpu-providers/page.tsx |
| ProvidersSecurity | `@rbee/ui/organisms/Providers` | gpu-providers/page.tsx |
| ProvidersTestimonials | `@rbee/ui/organisms/Providers` | gpu-providers/page.tsx |
| ProvidersCTA | `@rbee/ui/organisms/Providers` | gpu-providers/page.tsx |

### Organisms - Use Cases Group (3 components)

| Component | Import Path | Used In |
|-----------|-------------|---------|
| UseCasesHero | `@rbee/ui/organisms/UseCases` | use-cases/page.tsx |
| UseCasesPrimary | `@rbee/ui/organisms/UseCases` | use-cases/page.tsx |
| UseCasesIndustry | `@rbee/ui/organisms/UseCases` | use-cases/page.tsx |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Atoms** | 2 |
| **Organisms - Core** | 2 |
| **Organisms - Marketing** | 15 |
| **Organisms - Enterprise** | 4 |
| **Organisms - Developers** | 4 |
| **Organisms - Features** | 4 |
| **Organisms - Pricing** | 2 |
| **Organisms - Providers** | 5 |
| **Organisms - Use Cases** | 3 |
| **TOTAL** | **41** |

---

## Components NOT Used (No Stories Needed)

The following components exist in `@rbee/ui` but are NOT imported by commercial:

### Atoms (63 unused)
- Accordion, Alert, AlertDialog, AspectRatio, Avatar, Badge, BrandMark, BrandWordmark, Breadcrumb, Button, ButtonGroup, Calendar, Card, Carousel, Chart, CheckItem, Checkbox, CodeSnippet, Collapsible, Command, ConsoleOutput, ContextMenu, Dialog, Drawer, DropdownMenu, Empty, Field, Form, HoverCard, IconButton, Input, InputGroup, InputOtp, Item, Kbd, Label, Legend, Menubar, NavigationMenu, Pagination, Popover, Progress, RadioGroup, RatingStars, Resizable, ScrollArea, Select, Separator, Sheet, Sidebar, Skeleton, Slider, Sonner, Spinner, Switch, Table, Tabs, Textarea, Toast, Toaster, Toggle, ToggleGroup, Tooltip, UseMobile, UseToast

**Note:** Some of these (Button, Card, Badge) may be used indirectly by organisms. We only create stories for directly imported components.

### Molecules (40 unused)
All molecules are used indirectly through organisms, so no direct stories needed unless specifically imported.

---

## Priority Classification

### P0 - Critical (Must Have First)
- Navigation
- Footer
- GitHubIcon
- DiscordIcon

**Rationale:** Core layout components used on every page.

### P1 - High Priority (Core Marketing)
- HeroSection
- EmailCapture
- CTASection
- PricingSection
- FAQSection

**Rationale:** Most important marketing components, used on multiple pages.

### P2 - Medium Priority (Content Sections)
- WhatIsRbee
- AudienceSelector
- ProblemSection
- SolutionSection
- HowItWorksSection
- FeaturesSection
- UseCasesSection
- ComparisonSection
- SocialProofSection
- TestimonialsSection
- TechnicalSection

**Rationale:** Important content sections, mostly used on home page.

### P3 - Low Priority (Page-Specific)
- All Enterprise/* components (4)
- All Developers/* components (4)
- All Features/* components (4)
- All Pricing/* components (2)
- All Providers/* components (5)
- All UseCases/* components (3)

**Rationale:** Page-specific variants, can be done last.

---

## Verification Commands

### Verify component usage:
```bash
# Check if a component is used
grep -r "ComponentName" /home/vince/Projects/llama-orch/frontend/bin/commercial/app/ \
  --include="*.tsx" --include="*.ts"
```

### List all unique imports:
```bash
grep -rh "from '@rbee/ui" /home/vince/Projects/llama-orch/frontend/bin/commercial/app/ \
  --include="*.tsx" --include="*.ts" | \
  sed "s/.*from '\(@rbee\/ui[^']*\)'.*/\1/" | \
  sort -u
```

### Count components per category:
```bash
# Atoms
grep -r "@rbee/ui/atoms" /home/vince/Projects/llama-orch/frontend/bin/commercial/app/ \
  --include="*.tsx" | wc -l

# Organisms
grep -r "@rbee/ui/organisms" /home/vince/Projects/llama-orch/frontend/bin/commercial/app/ \
  --include="*.tsx" | wc -l
```

---

## Next Steps

1. ✅ Discovery complete
2. ⏳ Create foundation (Phase 1)
3. ⏳ Implement P0 stories (Phase 2)
4. ⏳ Implement P1 stories (Phase 3)
5. ⏳ Implement P2 stories (Phase 4)
6. ⏳ Implement P3 stories (Phase 5)
7. ⏳ QA all stories (Phase 6)
8. ⏳ Documentation (Phase 7)

**See STORYBOOK_STORIES_PLAN.md for detailed execution instructions.**

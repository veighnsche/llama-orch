# Shared Components Migration Plan

**Version:** 2.0  
**Date:** 2025-10-12  
**Status:** READY FOR IMPLEMENTATION  
**Estimated Impact:** ~50% code reduction, improved maintainability

---

## Executive Summary

Analysis of `/frontend/bin/commercial/components` reveals **25 duplicated component patterns** used across 60+ files. Extracting these into shared components will:

- **Reduce code duplication by ~50%** (~4,800 lines)
- **Improve maintainability** (single source of truth)
- **Ensure design consistency** across all pages
- **Speed up development** (reusable building blocks)

### Extended Analysis Findings

**Additional files analyzed:**
- Navigation & Footer (`navigation.tsx`, `footer.tsx`)
- Pricing pages (`pricing/*.tsx` - 4 files)
- Provider pages (`providers/*.tsx` - 11 files)
- Additional patterns in existing components

**New components identified:** 7 additional patterns (components #19-25)  
**Additional usage instances:** 50+  
**Additional code reduction:** ~1,400 lines

---

## Identified Shared Components

### 1. **SectionContainer** ⭐ HIGH PRIORITY
**Usage:** 15+ files  
**Pattern:**
```tsx
<section className="py-24 bg-{variant}">
  <div className="container mx-auto px-4">
    <div className="max-w-4xl mx-auto text-center mb-16">
      <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
        {title}
      </h2>
      {subtitle && <p className="text-xl text-muted-foreground">{subtitle}</p>}
    </div>
    {children}
  </div>
</section>
```

**Props:**
- `title: string`
- `subtitle?: string`
- `bgVariant?: 'background' | 'secondary' | 'card'`
- `centered?: boolean`
- `maxWidth?: 'xl' | '2xl' | '3xl' | '4xl' | '5xl' | '6xl' | '7xl'`
- `children: ReactNode`

**Found in:**
- `technical-section.tsx`
- `hero-section.tsx`
- `pricing-section.tsx`
- `use-cases-section.tsx`
- `cta-section.tsx`
- `problem-section.tsx`
- `how-it-works-section.tsx`
- `faq-section.tsx`
- `social-proof-section.tsx`
- `comparison-section.tsx`
- `solution-section.tsx`
- `features-section.tsx`
- All developer/enterprise/features variants

---

### 2. **PulseBadge** ⭐ HIGH PRIORITY
**Usage:** 6+ files  
**Pattern:**
```tsx
<div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full text-primary text-sm">
  <span className="relative flex h-2 w-2">
    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
    <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
  </span>
  {text}
</div>
```

**Props:**
- `text: string`
- `variant?: 'primary' | 'success' | 'warning' | 'info'`
- `size?: 'sm' | 'md' | 'lg'`
- `animated?: boolean`

**Found in:**
- `hero-section.tsx`
- `developers-hero.tsx`
- `enterprise-hero.tsx`
- `solution-section.tsx`
- `email-capture.tsx`

---

### 3. **TerminalWindow** ⭐ HIGH PRIORITY
**Usage:** 5+ files  
**Pattern:**
```tsx
<div className="bg-card border border-border rounded-lg overflow-hidden shadow-2xl">
  <div className="flex items-center gap-2 px-4 py-3 bg-muted border-b border-border">
    <div className="flex gap-2">
      <div className="h-3 w-3 rounded-full bg-red-500"></div>
      <div className="h-3 w-3 rounded-full bg-amber-500"></div>
      <div className="h-3 w-3 rounded-full bg-green-500"></div>
    </div>
    <span className="text-muted-foreground text-sm ml-2 font-mono">{title}</span>
  </div>
  <div className="p-6 font-mono text-sm">{children}</div>
</div>
```

**Props:**
- `title?: string`
- `children: ReactNode`
- `variant?: 'terminal' | 'code' | 'output'`

**Found in:**
- `hero-section.tsx`
- `developers-hero.tsx`
- `how-it-works-section.tsx`
- `features-section.tsx`
- `core-features-tabs.tsx`

---

### 4. **ProgressBar** (GPU Utilization Bar)
**Usage:** 4+ files  
**Pattern:**
```tsx
<div className="flex items-center gap-2">
  <span className="text-muted-foreground text-xs w-24">{label}</span>
  <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
    <div className="h-full bg-primary" style={{ width: `${percentage}%` }}></div>
  </div>
  <span className="text-muted-foreground text-xs">{percentage}%</span>
</div>
```

**Props:**
- `label: string`
- `percentage: number`
- `color?: string` (Tailwind color class)
- `size?: 'sm' | 'md' | 'lg'`
- `showLabel?: boolean`
- `showPercentage?: boolean`

**Found in:**
- `hero-section.tsx`
- `features-section.tsx`
- `core-features-tabs.tsx`

---

### 5. **FeatureCard** ⭐ HIGH PRIORITY
**Usage:** 12+ files  
**Pattern:**
```tsx
<div className="bg-card border border-border rounded-lg p-6 space-y-3">
  <div className="h-10 w-10 rounded-lg bg-{color}/10 flex items-center justify-center">
    <Icon className="h-5 w-5 text-{color}" />
  </div>
  <h3 className="text-lg font-bold text-card-foreground">{title}</h3>
  <p className="text-muted-foreground text-sm leading-relaxed">{description}</p>
</div>
```

**Props:**
- `icon: LucideIcon`
- `title: string`
- `description: string`
- `iconColor?: string`
- `hover?: boolean`
- `size?: 'sm' | 'md' | 'lg'`

**Found in:**
- `solution-section.tsx`
- `problem-section.tsx`
- `use-cases-section.tsx`
- `developers-solution.tsx`
- `enterprise-features.tsx`
- `enterprise-security.tsx`

---

### 6. **BulletListItem**
**Usage:** 5+ files  
**Pattern:**
```tsx
<li className="flex items-start gap-3">
  <div className="h-6 w-6 rounded-full bg-chart-3/20 flex items-center justify-center flex-shrink-0 mt-0.5">
    <div className="h-2 w-2 rounded-full bg-chart-3"></div>
  </div>
  <div>
    <div className="font-medium text-foreground">{title}</div>
    <div className="text-sm text-muted-foreground">{description}</div>
  </div>
</li>
```

**Props:**
- `title: string`
- `description?: string`
- `color?: string`
- `variant?: 'dot' | 'check' | 'arrow'`

**Found in:**
- `technical-section.tsx`
- `pricing-section.tsx`

---

### 7. **CheckListItem** ⭐ HIGH PRIORITY
**Usage:** 8+ files  
**Pattern:**
```tsx
<li className="flex items-start gap-2">
  <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
  <span className="text-muted-foreground">{text}</span>
</li>
```

**Props:**
- `text: string`
- `variant?: 'success' | 'primary' | 'muted'`
- `size?: 'sm' | 'md' | 'lg'`

**Found in:**
- `pricing-section.tsx`
- `comparison-section.tsx`
- `enterprise-features.tsx`
- `enterprise-security.tsx`

---

### 8. **StepNumber**
**Usage:** 4 instances in 1 file (reusable pattern)  
**Pattern:**
```tsx
<div className="inline-flex items-center justify-center h-12 w-12 rounded-full bg-primary text-primary-foreground font-bold text-xl">
  {number}
</div>
```

**Props:**
- `number: number`
- `size?: 'sm' | 'md' | 'lg' | 'xl'`
- `variant?: 'primary' | 'secondary' | 'outline'`

**Found in:**
- `how-it-works-section.tsx` (4 times)

---

### 9. **TrustIndicator**
**Usage:** 5+ files  
**Pattern:**
```tsx
<div className="flex items-center gap-2 text-muted-foreground">
  <Icon className="h-5 w-5" />
  <span className="text-sm">{text}</span>
</div>
```

**Props:**
- `icon: LucideIcon`
- `text: string`
- `variant?: 'default' | 'primary' | 'success'`

**Found in:**
- `hero-section.tsx`
- `developers-hero.tsx`
- `enterprise-hero.tsx`

---

### 10. **TestimonialCard**
**Usage:** 3+ files  
**Pattern:**
```tsx
<div className="bg-card border border-border rounded-lg p-6 space-y-4">
  <div className="flex items-center gap-3">
    <div className="h-12 w-12 rounded-full bg-gradient-to-br from-{color1} to-{color2}"></div>
    <div>
      <div className="font-bold text-card-foreground">{name}</div>
      <div className="text-sm text-muted-foreground">{role}</div>
    </div>
  </div>
  <p className="text-muted-foreground leading-relaxed">{quote}</p>
</div>
```

**Props:**
- `name: string`
- `role: string`
- `quote: string`
- `avatar?: string | { from: string; to: string }` (gradient or image)

**Found in:**
- `social-proof-section.tsx`

---

### 11. **IconBox**
**Usage:** 10+ files  
**Pattern:**
```tsx
<div className="h-12 w-12 rounded-lg bg-{color}/10 flex items-center justify-center">
  <Icon className="h-6 w-6 text-{color}" />
</div>
```

**Props:**
- `icon: LucideIcon`
- `color?: string`
- `size?: 'sm' | 'md' | 'lg' | 'xl'`
- `variant?: 'rounded' | 'circle' | 'square'`

**Found in:**
- `solution-section.tsx`
- `problem-section.tsx`
- `use-cases-section.tsx`
- `developers-solution.tsx`
- `developers-features.tsx`
- `enterprise-features.tsx`
- `enterprise-security.tsx`
- `audience-selector.tsx`

---

### 12. **StatCard** / **MetricDisplay**
**Usage:** 4+ files  
**Pattern:**
```tsx
<div className="text-center">
  <div className="text-4xl font-bold text-primary mb-2">{value}</div>
  <div className="text-sm text-muted-foreground">{label}</div>
</div>
```

**Props:**
- `value: string | number`
- `label: string`
- `variant?: 'primary' | 'success' | 'warning'`
- `size?: 'sm' | 'md' | 'lg'`

**Found in:**
- `social-proof-section.tsx`
- `enterprise-hero.tsx`
- `enterprise-security.tsx`

---

### 13. **CodeBlock**
**Usage:** 6+ files  
**Pattern:**
```tsx
<div className="bg-card border border-border rounded-lg p-6 font-mono text-sm">
  <pre className="overflow-x-auto">
    <code>{code}</code>
  </pre>
</div>
```

**Props:**
- `code: string`
- `language?: string`
- `showLineNumbers?: boolean`
- `highlight?: number[]` (line numbers to highlight)

**Found in:**
- `features-section.tsx`
- `core-features-tabs.tsx`
- `developers-features.tsx`
- `how-it-works-section.tsx`

---

### 14. **BenefitCallout** / **HighlightBox**
**Usage:** 6+ files  
**Pattern:**
```tsx
<div className="bg-{color}/10 border border-{color}/20 rounded-lg p-4">
  <p className="text-{color} font-medium">✓ {text}</p>
</div>
```

**Props:**
- `text: string`
- `variant?: 'success' | 'primary' | 'info' | 'warning'`
- `icon?: ReactNode`

**Found in:**
- `features-section.tsx`
- `core-features-tabs.tsx`
- `developers-features.tsx`

---

### 15. **TabButton** / **FeatureTab**
**Usage:** 3+ files  
**Pattern:**
```tsx
<button
  onClick={() => setActiveTab(id)}
  className={`flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-all ${
    activeTab === id
      ? "border-primary bg-primary/10 text-primary"
      : "border-border bg-card text-muted-foreground hover:border-border hover:text-foreground"
  }`}
>
  <Icon className="h-4 w-4" />
  {label}
</button>
```

**Props:**
- `id: string`
- `label: string`
- `icon: LucideIcon`
- `active: boolean`
- `onClick: () => void`

**Found in:**
- `developers-features.tsx`

---

### 16. **AudienceCard** / **PathCard**
**Usage:** 3 instances in 1 file  
**Pattern:**
```tsx
<Card className="group relative overflow-hidden border-border bg-card p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-{color}/50">
  <div className="absolute inset-0 -z-10 bg-gradient-to-br from-{color}/0 via-{color}/0 to-{color}/0 opacity-0 transition-all duration-500 group-hover:from-{color}/5 group-hover:via-{color}/10 group-hover:to-transparent group-hover:opacity-100" />
  
  <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-{color} to-{color} shadow-lg">
    <Icon className="h-7 w-7 text-primary-foreground" />
  </div>
  
  <div className="mb-2 text-sm font-medium uppercase tracking-wider text-{color}">{category}</div>
  <h3 className="mb-3 text-2xl font-semibold text-card-foreground">{title}</h3>
  <p className="mb-6 leading-relaxed text-muted-foreground">{description}</p>
  
  <ul className="mb-8 space-y-3 text-sm text-muted-foreground">
    {features.map(feature => (
      <li className="flex items-start gap-2">
        <span className="mt-1 text-{color}">→</span>
        <span>{feature}</span>
      </li>
    ))}
  </ul>
  
  <Link href={href}>
    <Button className="w-full bg-{color}">
      {ctaText}
      <ArrowRight className="ml-2 h-4 w-4" />
    </Button>
  </Link>
</Card>
```

**Props:**
- `icon: LucideIcon`
- `category: string`
- `title: string`
- `description: string`
- `features: string[]`
- `href: string`
- `ctaText: string`
- `color: string`

**Found in:**
- `audience-selector.tsx` (3 times)

---

### 17. **ArchitectureDiagram** / **BeeArchitecture**
**Usage:** 2+ files  
**Pattern:**
```tsx
<div className="rounded-lg border border-border bg-card p-8">
  <h3 className="mb-6 text-center text-xl font-semibold">The Bee Architecture</h3>
  <div className="flex flex-col items-center gap-6">
    {/* Queen */}
    <div className="flex items-center gap-3 rounded-lg border border-primary/30 bg-primary/10 px-6 py-3">
      <span className="text-2xl">👑</span>
      <div>
        <div className="font-semibold">queen-rbee</div>
        <div className="text-sm text-muted-foreground">Orchestrator</div>
      </div>
    </div>
    
    <div className="h-8 w-px bg-border" />
    
    {/* Hive */}
    {/* Workers */}
  </div>
</div>
```

**Props:**
- `variant?: 'simple' | 'detailed'`
- `showLabels?: boolean`

**Found in:**
- `solution-section.tsx`
- `developers-solution.tsx`

---

### 18. **SecurityCrateCard**
**Usage:** 5 instances in 1 file  
**Pattern:**
```tsx
<div className="rounded-lg border border-border bg-card p-8">
  <div className="mb-4 flex items-center gap-3">
    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
      <Icon className="h-6 w-6 text-primary" />
    </div>
    <div>
      <h3 className="text-xl font-bold text-foreground">{title}</h3>
      <p className="text-sm text-muted-foreground">{subtitle}</p>
    </div>
  </div>
  
  <p className="mb-4 leading-relaxed text-muted-foreground">{description}</p>
  
  <div className="space-y-2">
    {features.map(feature => (
      <div className="flex items-start gap-2 text-sm text-muted-foreground">
        <span className="text-chart-3">✓</span>
        <span>{feature}</span>
      </div>
    ))}
  </div>
</div>
```

**Props:**
- `icon: LucideIcon`
- `title: string`
- `subtitle: string`
- `description: string`
- `features: string[]`

**Found in:**
- `enterprise-security.tsx` (5 times)

---

## Migration Strategy

### Phase 1: Foundation (Week 1)
**Priority:** Create most-used components first

1. ✅ Create `/components/shared/` directory
2. ✅ Implement **SectionContainer** (15+ usages)
3. ✅ Implement **FeatureCard** (12+ usages)
4. ✅ Implement **IconBox** (10+ usages)
5. ✅ Implement **CheckListItem** (8+ usages)
6. ✅ Write Storybook stories for each
7. ✅ Write unit tests (Vitest)

### Phase 2: Visual Components (Week 2)
**Priority:** Terminal, badges, progress bars

1. ✅ Implement **PulseBadge**
2. ✅ Implement **TerminalWindow**
3. ✅ Implement **ProgressBar**
4. ✅ Implement **CodeBlock**
5. ✅ Implement **BenefitCallout**
6. ✅ Write Storybook stories
7. ✅ Write unit tests

### Phase 3: Specialized Components (Week 3)
**Priority:** Domain-specific patterns

1. ✅ Implement **TestimonialCard**
2. ✅ Implement **StatCard**
3. ✅ Implement **StepNumber**
4. ✅ Implement **TrustIndicator**
5. ✅ Implement **BulletListItem**
6. ✅ Implement **TabButton**
7. ✅ Write Storybook stories
8. ✅ Write unit tests

### Phase 4: Complex Components (Week 4)
**Priority:** Composite patterns

1. ✅ Implement **AudienceCard**
2. ✅ Implement **SecurityCrateCard**
3. ✅ Implement **ArchitectureDiagram**
4. ✅ Write Storybook stories
5. ✅ Write unit tests

### Phase 5: New Primitives from Extended Analysis (Week 5) ⭐⭐
**Priority:** Navigation, pricing, and provider-specific components

1. ✅ Implement **NavLink**
2. ✅ Implement **FooterColumn**
3. ✅ Implement **PricingTier**
4. ✅ Implement **ComparisonTableRow**
5. ✅ Implement **EarningsCard**
6. ✅ Implement **GPUListItem**
7. ✅ Implement **UseCaseCard**
8. ✅ Write Storybook stories for each
9. ✅ Write unit tests

### Phase 6: Migration (Week 6-7)
**Priority:** Replace usage across codebase

1. ✅ Migrate main sections (hero, features, pricing)
2. ✅ Migrate developer pages
3. ✅ Migrate enterprise pages
4. ✅ Migrate feature pages
5. ✅ Migrate provider pages
6. ✅ Migrate pricing pages
7. ✅ Migrate navigation and footer
8. ✅ Remove old duplicated code
9. ✅ Update documentation

### Phase 7: Validation (Week 8)
**Priority:** Ensure quality

1. ✅ Visual regression testing
2. ✅ Accessibility audit
3. ✅ Performance benchmarks
4. ✅ Code review
5. ✅ Update component library docs

---

## File Structure

```
/frontend/bin/commercial/components/
├── primitives/                  # Renamed from "shared" to "primitives"
│   ├── layout/
│   │   ├── SectionContainer.tsx
│   │   └── SectionContainer.stories.tsx
│   ├── badges/
│   │   ├── PulseBadge.tsx
│   │   └── PulseBadge.stories.tsx
│   ├── cards/
│   │   ├── FeatureCard.tsx
│   │   ├── FeatureCard.stories.tsx
│   │   ├── TestimonialCard.tsx
│   │   ├── TestimonialCard.stories.tsx
│   │   ├── AudienceCard.tsx
│   │   ├── AudienceCard.stories.tsx
│   │   ├── SecurityCrateCard.tsx
│   │   ├── SecurityCrateCard.stories.tsx
│   │   ├── UseCaseCard.tsx
│   │   └── UseCaseCard.stories.tsx
│   ├── code/
│   │   ├── TerminalWindow.tsx
│   │   ├── TerminalWindow.stories.tsx
│   │   ├── CodeBlock.tsx
│   │   └── CodeBlock.stories.tsx
│   ├── icons/
│   │   ├── IconBox.tsx
│   │   └── IconBox.stories.tsx
│   ├── lists/
│   │   ├── CheckListItem.tsx
│   │   ├── CheckListItem.stories.tsx
│   │   ├── BulletListItem.tsx
│   │   └── BulletListItem.stories.tsx
│   ├── progress/
│   │   ├── ProgressBar.tsx
│   │   └── ProgressBar.stories.tsx
│   ├── stats/
│   │   ├── StatCard.tsx
│   │   └── StatCard.stories.tsx
│   ├── steps/
│   │   ├── StepNumber.tsx
│   │   └── StepNumber.stories.tsx
│   ├── callouts/
│   │   ├── BenefitCallout.tsx
│   │   └── BenefitCallout.stories.tsx
│   ├── indicators/
│   │   ├── TrustIndicator.tsx
│   │   └── TrustIndicator.stories.tsx
│   ├── tabs/
│   │   ├── TabButton.tsx
│   │   └── TabButton.stories.tsx
│   ├── diagrams/
│   │   ├── ArchitectureDiagram.tsx
│   │   └── ArchitectureDiagram.stories.tsx
│   ├── navigation/
│   │   ├── NavLink.tsx
│   │   └── NavLink.stories.tsx
│   ├── footer/
│   │   ├── FooterColumn.tsx
│   │   └── FooterColumn.stories.tsx
│   ├── pricing/
│   │   ├── PricingTier.tsx
│   │   ├── PricingTier.stories.tsx
│   │   ├── ComparisonTableRow.tsx
│   │   └── ComparisonTableRow.stories.tsx
│   ├── earnings/
│   │   ├── EarningsCard.tsx
│   │   ├── EarningsCard.stories.tsx
│   │   ├── GPUListItem.tsx
│   │   └── GPUListItem.stories.tsx
│   └── index.ts  # Barrel export
```

---

## Testing Requirements

Each component must have:

1. **Unit tests** (Vitest)
   - Props validation
   - Variant rendering
   - Event handlers
   - Accessibility

2. **Storybook stories**
   - Default state
   - All variants
   - Interactive controls
   - Documentation

3. **Visual regression tests**
   - Chromatic snapshots
   - Responsive breakpoints

---

## Success Metrics

- ✅ **Code reduction:** 50% fewer lines in component files (~4,800 lines)
- ✅ **Reusability:** Each primitive component used 3+ times
- ✅ **Test coverage:** 90%+ for primitive components
- ✅ **Performance:** No regression in bundle size or render time
- ✅ **Accessibility:** WCAG 2.1 AA compliance
- ✅ **Documentation:** 100% of components have Storybook stories

---

## Breaking Changes

None. This is purely a refactor. All existing pages will maintain identical visual appearance and behavior.

---

## Rollback Plan

If issues arise:
1. Git revert to pre-migration commit
2. Incremental rollback per component
3. Feature flag to toggle between old/new implementations

---

## Next Steps

1. ✅ Review and approve this plan
2. ✅ Create `/components/primitives/` directory structure
3. ✅ Start Phase 1 implementation
4. ✅ Set up Storybook workspace for primitive components
5. ✅ Configure Vitest for component testing

---

## Appendix: Component Usage Matrix

| Component | Usage Count | Priority | Estimated Savings |
|-----------|-------------|----------|-------------------|
| SectionContainer | 15+ | ⭐⭐⭐ | ~600 lines |
| FeatureCard | 12+ | ⭐⭐⭐ | ~480 lines |
| IconBox | 10+ | ⭐⭐⭐ | ~200 lines |
| CheckListItem | 8+ | ⭐⭐⭐ | ~160 lines |
| PulseBadge | 6+ | ⭐⭐ | ~180 lines |
| TerminalWindow | 5+ | ⭐⭐ | ~250 lines |
| CodeBlock | 6+ | ⭐⭐ | ~180 lines |
| ProgressBar | 4+ | ⭐ | ~80 lines |
| BulletListItem | 5+ | ⭐ | ~100 lines |
| TestimonialCard | 3+ | ⭐ | ~90 lines |
| StatCard | 4+ | ⭐ | ~80 lines |
| StepNumber | 4 | ⭐ | ~40 lines |
| TrustIndicator | 5+ | ⭐ | ~75 lines |
| BenefitCallout | 6+ | ⭐⭐ | ~120 lines |
| TabButton | 3+ | ⭐ | ~60 lines |
| AudienceCard | 3 | ⭐ | ~300 lines |
| SecurityCrateCard | 5 | ⭐ | ~250 lines |
| ArchitectureDiagram | 2+ | ⭐ | ~150 lines |
| **NavLink** | **20+** | **⭐⭐** | **~400 lines** |
| **FooterColumn** | **4** | **⭐** | **~80 lines** |
| **PricingTier** | **6** | **⭐⭐** | **~360 lines** |
| **ComparisonTableRow** | **16+** | **⭐⭐** | **~320 lines** |
| **EarningsCard** | **2** | **⭐** | **~120 lines** |
| **GPUListItem** | **8+** | **⭐⭐** | **~160 lines** |
| **UseCaseCard** | **4** | **⭐** | **~360 lines** |
| **TOTAL** | **150+** | | **~4,795 lines** |

---

**End of Migration Plan**

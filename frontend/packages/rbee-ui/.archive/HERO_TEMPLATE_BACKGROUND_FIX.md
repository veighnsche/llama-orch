# HeroTemplate Background Fix

**Issue:** HeroTemplate was rendering its own background elements, causing duplication/conflicts with TemplateContainer's background rendering.

**Root Cause:** Violation of separation of concerns—`HeroTemplate` should NOT handle backgrounds. `TemplateContainer` exclusively handles background rendering via the `TemplateBackground` organism.

---

## Fix Applied

### 1. Removed Background Rendering from HeroTemplate

**Files Modified:**
- `src/templates/HeroTemplate/HeroTemplate.tsx`
- `src/templates/HeroTemplate/HeroTemplateProps.tsx`

**Changes:**
- ❌ Removed `renderBackground()` function
- ❌ Removed `{renderBackground()}` call
- ❌ Removed `background` prop from `HeroTemplateProps`
- ❌ Removed unused `HeroBackground` and `BackgroundVariant` types
- ❌ Removed unused imports: `HoneycombPattern`, `cn`

---

## Migration Required

### Pages Using HeroTemplate (5 files)

These pages pass `background` directly to `HeroTemplate`. They need to be updated to pass `background` to `TemplateContainer` instead.

**Files to Update:**
1. `src/pages/HomePage/HomePageProps.tsx:174`
2. `src/pages/HomelabPage/HomelabPageProps.tsx:99`
3. `src/pages/PrivacyPage/PrivacyPageProps.tsx:45`
4. `src/pages/StartupsPage/StartupsPageProps.tsx:137`
5. `src/pages/TermsPage/TermsPageProps.tsx:63`

**Migration Pattern:**

```tsx
// ❌ BEFORE (passing background to HeroTemplate)
export const homePageProps: HomePageProps = {
  hero: {
    background: {
      variant: 'honeycomb',
      size: 'large',
      fadeDirection: 'radial',
    },
    // ... other hero props
  },
}

// ✅ AFTER (passing background to TemplateContainer)
export const homePageProps: HomePageProps = {
  hero: {
    // Remove background from here
    // ... other hero props
  },
  // Add background at page level for TemplateContainer
  background: {
    variant: 'pattern-honeycomb',
    patternSize: 'large',
    // Note: TemplateBackground uses different prop names
  },
}
```

---

### Hero Templates (7 files)

These templates instantiate `HeroTemplate` directly and pass `background` prop.

**Files to Update:**
1. `src/templates/DevelopersHero/DevelopersHeroTemplate.tsx:202`
2. `src/templates/EnterpriseHero/EnterpriseHero.tsx:196`
3. `src/templates/FeaturesHero/FeaturesHero.tsx:118`
4. `src/templates/HomeHero/HomeHero.tsx:190`
5. `src/templates/PricingHero/PricingHeroTemplate.tsx:101`
6. `src/templates/ProvidersHero/ProvidersHero.tsx:179`
7. `src/templates/UseCasesHero/UseCasesHeroTemplate.tsx:115`

**Migration Pattern:**

```tsx
// ❌ BEFORE (passing background to HeroTemplate)
<HeroTemplate
  background={{
    variant: 'honeycomb',
    size: 'large',
    fadeDirection: 'radial',
  }}
  // ... other props
/>

// ✅ AFTER (remove background prop)
<HeroTemplate
  // ... other props (no background)
/>
```

**Note:** These templates should be wrapped by `TemplateContainer` at the page level, which will handle the background.

---

## Background Prop Mapping

`HeroTemplate` used different prop names than `TemplateBackground`. Here's the mapping:

| HeroTemplate (OLD) | TemplateBackground (NEW) |
|--------------------|--------------------------|
| `variant: 'gradient'` | `variant: 'background'` |
| `variant: 'radial'` | `variant: 'gradient-radial'` |
| `variant: 'honeycomb'` | `variant: 'pattern-honeycomb'` + `decoration: <HoneycombPattern />` |
| `variant: 'custom'` | `decoration: <CustomElement />` |
| `size: 'small' \| 'large'` | `patternSize: 'small' \| 'medium' \| 'large'` |
| `fadeDirection: 'radial' \| 'bottom'` | (handled by HoneycombPattern props) |

---

## Verification

After migration, verify:

1. **Build passes:**
   ```bash
   pnpm --filter @rbee/ui build
   ```

2. **No background duplication:**
   - Open each page in Storybook/browser
   - Verify only ONE background is rendered
   - Verify background matches expected design

3. **TemplateContainer wraps HeroTemplate:**
   - All hero sections should be wrapped by `TemplateContainer`
   - Background config passed to `TemplateContainer`, not `HeroTemplate`

---

## Architecture Rule

**NEVER render backgrounds in templates. ALWAYS delegate to TemplateContainer.**

### Correct Pattern

```tsx
// Page level
<TemplateContainer
  background={{
    variant: 'pattern-honeycomb',
    decoration: <HoneycombPattern id="hero" size="large" />,
  }}
>
  <HeroTemplate
    // No background prop
    headline={...}
    subcopy={...}
    // ... other props
  />
</TemplateContainer>
```

### Incorrect Pattern

```tsx
// ❌ WRONG: Template rendering its own background
<HeroTemplate
  background={{ variant: 'honeycomb' }}  // ❌ NO!
  // ...
/>
```

---

## Why This Matters

1. **Separation of Concerns:** Templates handle content layout, containers handle presentation (backgrounds, spacing, etc.)
2. **No Duplication:** Prevents bugs like the one shown in the image where backgrounds conflict
3. **Consistency:** All templates use the same background system via `TemplateContainer`
4. **Maintainability:** Background logic lives in ONE place (`TemplateBackground`), not scattered across templates

---

## Status

- ✅ **HeroTemplate fixed** — No longer renders backgrounds
- ⏳ **Migration pending** — 12 files need to be updated (5 pages + 7 templates)
- ⏳ **Build failing** — TypeScript errors until migration complete

---

## Next Steps

1. Update 5 page files to move `background` from hero config to page-level `TemplateContainer`
2. Update 7 template files to remove `background` prop from `HeroTemplate` calls
3. Verify build passes
4. Visual test all affected pages
5. Document pattern in component library guidelines

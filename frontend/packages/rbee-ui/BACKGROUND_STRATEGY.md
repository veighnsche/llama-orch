# Background Strategy for Templates

## Principles

1. **Visual Hierarchy** - Alternate backgrounds to create rhythm and visual separation
2. **Content Focus** - Important content sections get subtle backgrounds
3. **Destructive Warnings** - Problem sections use destructive-gradient
4. **Neutral Defaults** - Most sections use 'background' (default theme color)
5. **Secondary for Alternation** - Use 'secondary' to break up long pages

## Background Patterns by Template Type

### Hero Sections
- **Background:** Built into HeroTemplate (honeycomb, radial, gradient)
- **Container:** NO - Heroes handle their own backgrounds
- **Examples:** HomeHero, DevelopersHero, EnterpriseHero, FeaturesHero

### EmailCapture
- **Background:** 'background' with bee glyph decorations
- **Container:** YES - Always wrap in TemplateContainer
- **Pattern:** Consistent across all pages

### FeaturesTabs
- **Background:** 'background' (neutral)
- **Container:** YES - Always wrap in TemplateContainer
- **Pattern:** Consistent across all pages

### Problem Templates
- **Background:** 'destructive-gradient' (warning/alert)
- **Container:** YES
- **Reason:** Highlight pain points and risks

### Solution Templates
- **Background:** 'background' or 'default' (neutral)
- **Container:** YES
- **Reason:** Present solutions clearly without distraction

### How It Works
- **Background:** 'secondary' (alternation)
- **Container:** YES
- **Reason:** Break up page flow, create visual separation

### Comparison Tables
- **Background:** 'secondary' or 'background'
- **Container:** YES
- **Reason:** Tables need clear boundaries

### FAQ Sections
- **Background:** 'background' (neutral)
- **Container:** YES
- **Reason:** Keep focus on content

### Testimonials
- **Background:** 'background' or 'default'
- **Container:** YES
- **Reason:** Social proof needs clean presentation

### CTA Sections
- **Background:** Usually none or subtle
- **Container:** Depends on CTA type
- **Reason:** CTAs often have their own styling

### Use Cases / Features Grids
- **Background:** Alternate 'background' and 'secondary'
- **Container:** YES
- **Reason:** Create rhythm on long pages

## Recommended Background Flow

### Typical Page Structure:
1. **Hero** - Built-in background (honeycomb/gradient)
2. **EmailCapture** - 'background' + decorations
3. **Problem** - 'destructive-gradient' ⚠️
4. **Solution** - 'background'
5. **How It Works** - 'secondary' (alternation)
6. **FeaturesTabs** - 'background'
7. **Use Cases** - 'secondary' (alternation)
8. **Comparison** - 'background'
9. **Pricing** - 'secondary' (alternation)
10. **Testimonials** - 'background'
11. **FAQ** - 'secondary' (alternation)
12. **CTA** - 'background' or none

## Background Variants

### 'background'
- Default theme background color
- Use for: Most content sections
- Frequency: 60% of sections

### 'secondary'
- Slightly different shade for alternation
- Use for: Breaking up long pages
- Frequency: 30% of sections

### 'destructive-gradient'
- Red/warning gradient
- Use for: Problem sections only
- Frequency: 5-10% of sections

### 'default'
- Alias for 'background'
- Legacy - prefer 'background'

### Pattern Backgrounds
- 'pattern-honeycomb' - Brand theme
- 'pattern-dots' - Subtle texture
- 'pattern-grid' - Technical feel
- Use sparingly for special sections

## Templates That DON'T Need Containers

1. **Hero Templates** - Handle their own backgrounds
2. **Standalone CTAs** - Often have built-in styling
3. **Navigation/Footer** - System-level components

## Next Steps

I'll now audit all pages and standardize their background usage following these principles.

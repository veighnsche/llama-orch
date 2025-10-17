# TemplateBackground Guide

## Overview

`TemplateBackground` provides comprehensive background control for templates with:

- **Solid colors** - Background, secondary, card, muted, accent, primary, destructive
- **Gradients** - Primary, secondary, destructive, radial, mesh, warm, cool
- **SVG Patterns** - Dots, grid, honeycomb, waves, circuit, diagonal
- **Custom decorations** - Pass any React element as decoration
- **Overlays** - Control opacity and color
- **Blur effects** - Apply blur to backgrounds

## Basic Usage

### Solid Colors

```tsx
<TemplateBackground variant="background">
  <YourContent />
</TemplateBackground>

<TemplateBackground variant="secondary">
  <YourContent />
</TemplateBackground>

<TemplateBackground variant="card">
  <YourContent />
</TemplateBackground>
```

### Gradients

```tsx
// Subtle primary gradient
<TemplateBackground variant="gradient-primary">
  <YourContent />
</TemplateBackground>

// Radial gradient from top
<TemplateBackground variant="gradient-radial">
  <YourContent />
</TemplateBackground>

// Mesh gradient (diagonal)
<TemplateBackground variant="gradient-mesh">
  <YourContent />
</TemplateBackground>

// Warm gradient (amber/orange)
<TemplateBackground variant="gradient-warm">
  <YourContent />
</TemplateBackground>

// Cool gradient (blue/cyan)
<TemplateBackground variant="gradient-cool">
  <YourContent />
</TemplateBackground>
```

### SVG Patterns

All patterns support three sizes (`small`, `medium`, `large`) and adjustable opacity:

```tsx
// Dots pattern
<TemplateBackground 
  variant="pattern-dots" 
  patternSize="medium" 
  patternOpacity={12}
>
  <YourContent />
</TemplateBackground>

// Grid pattern
<TemplateBackground 
  variant="pattern-grid" 
  patternSize="large" 
  patternOpacity={8}
>
  <YourContent />
</TemplateBackground>

// Honeycomb pattern (beehive theme!)
<TemplateBackground 
  variant="pattern-honeycomb" 
  patternSize="medium" 
  patternOpacity={10}
>
  <YourContent />
</TemplateBackground>

// Waves pattern
<TemplateBackground 
  variant="pattern-waves" 
  patternSize="small" 
  patternOpacity={12}
>
  <YourContent />
</TemplateBackground>

// Circuit pattern (tech theme)
<TemplateBackground 
  variant="pattern-circuit" 
  patternSize="medium" 
  patternOpacity={15}
>
  <YourContent />
</TemplateBackground>

// Diagonal lines
<TemplateBackground 
  variant="pattern-diagonal" 
  patternSize="medium" 
  patternOpacity={6}
>
  <YourContent />
</TemplateBackground>
```

## Advanced Usage

### Custom Decorations

Pass any React element as `decoration` for complete control:

```tsx
<TemplateBackground
  variant="background"
  decoration={
    <div className="absolute inset-0 opacity-10">
      <svg className="w-full h-full">
        <defs>
          <pattern id="custom" x="0" y="0" width="50" height="50" patternUnits="userSpaceOnUse">
            <circle cx="25" cy="25" r="3" fill="currentColor" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#custom)" className="text-primary" />
      </svg>
    </div>
  }
>
  <YourContent />
</TemplateBackground>
```

### Overlays

Add a semi-transparent overlay on top of any background:

```tsx
<TemplateBackground
  variant="gradient-primary"
  overlayOpacity={30}
  overlayColor="black"
>
  <YourContent />
</TemplateBackground>
```

### Blur Effects

Apply blur to the background (useful for decorations):

```tsx
<TemplateBackground
  variant="pattern-honeycomb"
  patternSize="large"
  patternOpacity={20}
  blur={true}
>
  <YourContent />
</TemplateBackground>
```

### Combined Effects

Combine multiple effects for rich backgrounds:

```tsx
<TemplateBackground
  variant="gradient-mesh"
  decoration={<CustomSVGPattern />}
  overlayOpacity={15}
  overlayColor="primary"
  blur={false}
>
  <YourContent />
</TemplateBackground>
```

## Using with TemplateContainer

`TemplateContainer` automatically uses `TemplateBackground`. Pass background configuration via the `background` prop:

```tsx
<TemplateContainer
  title="Features"
  description="Explore our capabilities"
  background={{
    variant: 'pattern-honeycomb',
    patternSize: 'large',
    patternOpacity: 8,
  }}
>
  <YourContent />
</TemplateContainer>
```

### Legacy bgVariant (Deprecated)

The old `bgVariant` prop is still supported but deprecated:

```tsx
// Old way (still works)
<TemplateContainer bgVariant="secondary">
  <YourContent />
</TemplateContainer>

// New way (preferred)
<TemplateContainer background={{ variant: 'secondary' }}>
  <YourContent />
</TemplateContainer>
```

## Pattern Recommendations

### By Use Case

**Hero Sections:**
- `pattern-honeycomb` (large, opacity 6-8) - Aligns with beehive brand
- `gradient-radial` - Draws focus to center
- `gradient-mesh` - Modern, dynamic feel

**Feature Sections:**
- `pattern-grid` (medium, opacity 8-10) - Structured, organized
- `pattern-dots` (small, opacity 12-15) - Subtle texture
- `gradient-primary` - Highlight important sections

**Technical Sections:**
- `pattern-circuit` (medium, opacity 12-15) - Tech/engineering theme
- `pattern-grid` (large, opacity 8) - Blueprint aesthetic
- `gradient-cool` - Professional, technical feel

**CTA Sections:**
- `gradient-warm` - Inviting, action-oriented
- `pattern-waves` (medium, opacity 10) - Dynamic movement
- `gradient-primary` - Brand emphasis

**Compliance/Security:**
- `pattern-diagonal` (small, opacity 6) - Subtle, professional
- `gradient-secondary` - Trustworthy, stable
- `muted` solid color - Clean, serious

### By Brand Theme

**Beehive/Nature Theme:**
- `pattern-honeycomb` ⭐ Primary choice
- `gradient-warm` (amber tones)
- Custom decoration with bee glyphs

**Enterprise/Professional:**
- `pattern-grid`
- `gradient-cool`
- `subtle-border`

**Developer/Technical:**
- `pattern-circuit`
- `pattern-grid`
- `gradient-mesh`

## Performance Notes

- SVG patterns are rendered inline and cached by the browser
- Each pattern instance generates a unique ID to avoid conflicts
- Patterns are lightweight (~1-2KB each)
- Use `patternOpacity` to control visual weight without affecting performance

## Accessibility

- All decorative elements have `aria-hidden="true"`
- Content layer has `z-10` to ensure proper stacking
- Patterns use `currentColor` to respect theme colors
- Blur effects respect `prefers-reduced-motion`

## Examples from Existing Templates

### HomeHero
Uses HeroTemplate with honeycomb background:
```tsx
background={{ variant: 'honeycomb', size: 'large', fadeDirection: 'radial' }}
```

### FeaturesHero
Uses HeroTemplate with honeycomb background:
```tsx
background={{ variant: 'honeycomb', size: 'small', fadeDirection: 'bottom' }}
```

### EmailCapture
Uses inline background:
```tsx
className="relative isolate py-28 bg-background"
```
**Recommendation:** Migrate to `TemplateContainer` with `background={{ variant: 'background' }}`

## Migration Checklist

When migrating templates to use `TemplateBackground`:

- [ ] Replace inline `bg-*` classes with `TemplateContainer` `background` prop
- [ ] Move inline SVG patterns to `decoration` prop
- [ ] Convert gradient classes to `variant` values
- [ ] Use `patternSize` and `patternOpacity` for fine control
- [ ] Test with both light and dark themes
- [ ] Verify accessibility (aria-hidden, z-index)
- [ ] Check performance (no layout shifts)

## Future Enhancements

Potential additions (not yet implemented):

- Animated patterns (subtle movement)
- Gradient animation (color shifts)
- Parallax effects for decorations
- Theme-aware pattern colors
- Pattern fade directions (like HoneycombPattern)

# Update TEAM-FE-002 Components to Design Tokens

**Task:** Replace all hardcoded colors in TEAM-FE-002 components

## Components to Update

### 1. Badge.story.vue
- Line 25: `bg-amber-500 text-white` → `bg-primary text-primary-foreground`

### 2. PricingCard.vue
- Multiple `slate-` colors → design tokens
- Multiple `amber-` colors → `primary` tokens
- `green-600` → appropriate token

### 3. PricingHero.vue  
- `slate-950`, `slate-900` → `background` tokens
- `amber-500` → `primary`
- `slate-300` → `muted-foreground`

### 4. PricingComparisonTable.vue
- Extensive `slate-` colors → design tokens
- `amber-50` → `primary/10` or `accent/10`
- `green-600` → success color token

### 5. PricingFAQ.vue
- `slate-` colors → design tokens

### 6. PricingTiers.vue
- Check for any hardcoded colors

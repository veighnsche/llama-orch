# Migration Guide: SVG Assets → React Components

Quick reference for migrating from static SVG imports to React icon components.

## Import Changes

### Before
```typescript
import { beeMarkSvg, githubIcon, discordIcon } from '@rbee/ui/assets'
```

### After
```typescript
import { BeeMark, GithubIcon, DiscordIcon } from '@rbee/ui/icons'
```

---

## Usage Patterns

### Pattern 1: Simple Icon Display

#### Before
```tsx
import Image from 'next/image'
import { beeMarkSvg } from '@rbee/ui/assets'

<Image src={beeMarkSvg} alt="Bee" width={24} height={24} />
```

#### After
```tsx
import { BeeMark } from '@rbee/ui/icons'

<BeeMark size={24} />
```

---

### Pattern 2: Styled Icons

#### Before
```tsx
<div className="icon-wrapper">
  <Image 
    src={beeMarkSvg} 
    alt="Bee" 
    width={32} 
    height={32}
    className="custom-icon"
  />
</div>
```

#### After
```tsx
<BeeMark size={32} className="custom-icon" />
```

---

### Pattern 3: Responsive Sizing

#### Before
```tsx
<Image 
  src={beeMarkSvg}
  alt="Bee"
  width={isMobile ? 24 : 48}
  height={isMobile ? 24 : 48}
/>
```

#### After
```tsx
<BeeMark size={isMobile ? 24 : 48} />
```

---

### Pattern 4: Colored Icons

#### Before
```tsx
<div style={{ color: '#f59e0b' }}>
  <Image src={beeMarkSvg} alt="Bee" width={24} height={24} />
</div>
```

#### After
```tsx
<BeeMark size={24} className="text-amber-500" />
```

---

### Pattern 5: Interactive Icons

#### Before
```tsx
<button onClick={handleClick}>
  <Image src={githubIcon} alt="GitHub" width={20} height={20} />
</button>
```

#### After
```tsx
<button onClick={handleClick}>
  <GithubIcon size={20} />
</button>

// Or with hover effects
<GithubIcon 
  size={20}
  className="text-foreground hover:text-primary transition-colors cursor-pointer"
  onClick={handleClick}
/>
```

---

### Pattern 6: Accessibility

#### Before
```tsx
<Image 
  src={beeMarkSvg}
  alt="Company logo"
  width={48}
  height={48}
  role="img"
/>
```

#### After
```tsx
<BeeMark 
  size={48}
  aria-label="Company logo"
  role="img"
/>
```

---

## Component Mapping

| Old Asset Import | New Icon Component |
|------------------|-------------------|
| `beeGlyph` | `<BeeGlyph />` |
| `beeMarkSvg` | `<BeeMark />` |
| `complianceShield` | `<ComplianceShield />` |
| `devGrid` | `<DevGrid />` |
| `discordIcon` | `<DiscordIcon />` |
| `formerCryptoMiner` | `<FormerCryptoMiner />` |
| `gamingPcOwner` | `<GamingPcOwner />` |
| `githubIcon` | `<GithubIcon />` |
| `gpuMarket` | `<GpuMarket />` |
| `homelabBee` | `<HomelabBee />` |
| `homelabEnthusiast` | `<HomelabEnthusiast />` |
| `honeycombPattern` | `<HoneycombPattern />` |
| `industriesHero` | `<IndustriesHero />` |
| `placeholderLogoSvg` | `<PlaceholderLogo />` |
| `placeholderSvg` | `<Placeholder />` |
| `pricingOrchestratorSvg` | `<PricingOrchestrator />` |
| `pricingScaleVisual` | `<PricingScaleVisual />` |
| `rbeeArch` | `<RbeeArch />` |
| `starRating` | `<StarRating />` |
| `useCasesHeroSvg` | `<UseCasesHero />` |
| `usecasesGridDark` | `<UsecasesGridDark />` |
| `vendorLockIn` | `<VendorLockIn />` |
| `workstationOwner` | `<WorkstationOwner />` |
| `xTwitterIcon` | `<XTwitterIcon />` |

---

## Advanced Usage

### With Framer Motion
```tsx
import { motion } from 'framer-motion'
import { BeeMark } from '@rbee/ui/icons'

<motion.div
  animate={{ rotate: 360 }}
  transition={{ duration: 2, repeat: Infinity }}
>
  <BeeMark size={64} />
</motion.div>
```

### With Radix UI
```tsx
import { BeeMark } from '@rbee/ui/icons'
import * as Dialog from '@radix-ui/react-dialog'

<Dialog.Trigger>
  <BeeMark size={24} className="cursor-pointer" />
</Dialog.Trigger>
```

### Conditional Rendering
```tsx
import { GithubIcon, DiscordIcon } from '@rbee/ui/icons'

const SocialIcon = ({ type }: { type: 'github' | 'discord' }) => {
  return type === 'github' 
    ? <GithubIcon size={24} />
    : <DiscordIcon size={24} />
}
```

---

## Benefits Summary

✅ **Cleaner code** - No need for Image component wrapper  
✅ **Better DX** - TypeScript autocomplete and type safety  
✅ **Smaller imports** - Tree-shakeable, only import what you use  
✅ **More flexible** - Easy sizing, styling, and composition  
✅ **Semantic HTML** - Renders as proper `<svg>` elements  
✅ **Accessibility** - Direct SVG props for ARIA attributes

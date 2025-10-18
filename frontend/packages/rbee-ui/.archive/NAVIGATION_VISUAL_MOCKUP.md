# Navigation Visual Mockup

**Date:** October 17, 2025  
**Purpose:** Visual representation of the new navigation design

---

## Desktop Navigation (≥768px)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                │
│  🐝 rbee     Product ▼    Solutions ▼    Resources ▼        Documentation  [GitHub] [🌙] [Join Waitlist]  │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
     ↑              ↑             ↑              ↑                      ↑          ↑      ↑         ↑
   Logo      Dropdown 1    Dropdown 2     Dropdown 3            Standalone!    Icons  Theme      CTA
```

---

## Product Dropdown (Simple List)

```
Product ▼
┌─────────────────────────┐
│  Features               │
│  Pricing                │
│  Use Cases              │
└─────────────────────────┘
```

**Interaction:**
- Hover: Opens dropdown
- Click trigger: Toggles dropdown
- Click outside: Closes dropdown
- Escape: Closes dropdown

---

## Solutions Dropdown (2-Column Layout)

```
Solutions ▼
┌────────────────────────────────────────────────────────────────────┐
│  PRIMARY AUDIENCES              │  INDUSTRIES                      │
├─────────────────────────────────┼──────────────────────────────────┤
│  💻 For Developers              │  🚀 Startups                     │
│     Build AI tools without      │     Scale your AI infrastructure │
│     vendor lock-in              │                                  │
│                                 │  🏠 Homelab                      │
│  🏢 For Enterprise              │     Self-hosted AI for           │
│     GDPR-compliant AI           │     enthusiasts                  │
│     infrastructure              │                                  │
│                                 │  🔬 Research                     │
│  🖥️  For Providers              │     Reproducible ML experiments  │
│     Earn with your idle GPUs    │                                  │
│                                 │  🛡️  Compliance                  │
│                                 │     EU-native, GDPR-ready        │
│                                 │                                  │
│                                 │  🎓 Education                    │
│                                 │     Learn distributed AI systems │
│                                 │                                  │
│                                 │  ⚙️  DevOps                      │
│                                 │     Production-ready             │
│                                 │     orchestration                │
└─────────────────────────────────┴──────────────────────────────────┘
```

**Layout:**
- Width: ~600px
- 2 columns: 50% / 50%
- Each item: Icon + Label + Description
- Hover state: Light background highlight
- Separator between columns

---

## Resources Dropdown (Simple List with Icons)

```
Resources ▼
┌─────────────────────────┐
│  👥 Community       ↗   │
│  🔒 Security            │
│  ⚖️  Legal              │
└─────────────────────────┘
```

**Note:** Community has external link icon (↗) if linking to Discord/GitHub Discussions

---

## Mobile Navigation (Accordion Style)

```
┌────────────────────────────────┐
│  🐝 rbee              ☰        │
└────────────────────────────────┘

[When menu opened:]

┌────────────────────────────────┐
│  🐝 rbee              ✕        │
├────────────────────────────────┤
│                                │
│  Product ▼                     │
│    Features                    │
│    Pricing                     │
│    Use Cases                   │
│                                │
│  Solutions ▼                   │
│    For Developers              │
│    For Enterprise              │
│    For Providers               │
│    ─────────────────           │
│    Startups                    │
│    Homelab                     │
│    Research                    │
│    Compliance                  │
│    Education                   │
│    DevOps                      │
│                                │
│  Resources ▼                   │
│    Community ↗                 │
│    Security                    │
│    Legal                       │
│                                │
│  ─────────────────             │
│                                │
│  Documentation ↗               │
│  GitHub                        │
│                                │
│  [Join Waitlist]               │
│                                │
└────────────────────────────────┘
```

---

## Hover States

### Desktop Dropdown Trigger (Closed)
```
Product ▼
  ↑
  Normal state: text-foreground/80
```

### Desktop Dropdown Trigger (Hovered)
```
Product ▼
  ↑
  Hover state: text-foreground, bg-accent/50
```

### Desktop Dropdown Trigger (Open)
```
Product ▲
  ↑
  Open state: text-foreground, bg-accent/50, chevron rotated
```

### Dropdown Item (Normal)
```
┌─────────────────────────┐
│  Features               │
└─────────────────────────┘
```

### Dropdown Item (Hovered)
```
┌─────────────────────────┐
│  Features               │  ← bg-accent, text-accent-foreground
└─────────────────────────┘
```

---

## Spacing & Alignment

### Desktop Layout Grid
```
┌─────────────────────────────────────────────────────────────────────────┐
│  [Zone A: 200px]  [Zone B: 1fr (centered)]  [Zone C: auto (right)]     │
│                                                                         │
│  🐝 rbee          Product  Solutions  Resources    Docs [Icons] [CTA]  │
│                   ↑                                                     │
│                   gap-6 (24px between each dropdown)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Zone C (Actions) Layout
```
[Documentation]  [GitHub]  [Theme]  [Join Waitlist]
       ↑            ↑         ↑           ↑
     gap-3        gap-1     gap-1       gap-2
    (12px)       (4px)     (4px)       (8px)
```

**Icon Box:**
```
┌──────────────────────┐
│ [GitHub] [Theme]     │  ← bg-muted/30, ring-1 ring-border/60, rounded-xl, p-0.5
└──────────────────────┘
```

---

## Dropdown Animation

### Opening (150ms ease-out)
```
Frame 1 (0ms):    opacity: 0, translateY: -8px
Frame 2 (75ms):   opacity: 0.5, translateY: -4px
Frame 3 (150ms):  opacity: 1, translateY: 0
```

### Closing (100ms ease-in)
```
Frame 1 (0ms):    opacity: 1, translateY: 0
Frame 2 (100ms):  opacity: 0, translateY: -4px
```

---

## Responsive Breakpoints

### Desktop (≥768px)
- Show: Dropdown menus
- Hide: Mobile hamburger menu
- Layout: 3-zone grid

### Tablet (768px - 1024px)
- Dropdown spacing: gap-4 (reduced from gap-6)
- Solutions dropdown: Slightly narrower columns

### Mobile (<768px)
- Show: Hamburger menu
- Hide: Dropdown menus
- Layout: Accordion-style menu

---

## Color Palette (Light Mode)

```
Background:       bg-background/95 (rgba(255, 255, 255, 0.95))
Border:           border-border/60 (rgba(0, 0, 0, 0.1))
Text:             text-foreground (rgba(0, 0, 0, 0.9))
Text Muted:       text-muted-foreground (rgba(0, 0, 0, 0.6))
Hover BG:         bg-accent (rgba(0, 0, 0, 0.05))
Hover Text:       text-accent-foreground (rgba(0, 0, 0, 1))
```

## Color Palette (Dark Mode)

```
Background:       bg-background/95 (rgba(0, 0, 0, 0.95))
Border:           border-border/60 (rgba(255, 255, 255, 0.1))
Text:             text-foreground (rgba(255, 255, 255, 0.9))
Text Muted:       text-muted-foreground (rgba(255, 255, 255, 0.6))
Hover BG:         bg-accent (rgba(255, 255, 255, 0.1))
Hover Text:       text-accent-foreground (rgba(255, 255, 255, 1))
```

---

## Icon Specifications

### Dropdown Icons (Lucide React)
- **Code** (For Developers): `<Code className="size-5" />`
- **Building** (For Enterprise): `<Building className="size-5" />`
- **Server** (For Providers): `<Server className="size-5" />`
- **Rocket** (Startups): `<Rocket className="size-5" />`
- **Home** (Homelab): `<Home className="size-5" />`
- **FlaskConical** (Research): `<FlaskConical className="size-5" />`
- **Shield** (Compliance): `<Shield className="size-5" />`
- **GraduationCap** (Education): `<GraduationCap className="size-5" />`
- **Settings** (DevOps): `<Settings className="size-5" />`
- **Users** (Community): `<Users className="size-5" />`
- **Lock** (Security): `<Lock className="size-5" />`
- **Scale** (Legal): `<Scale className="size-5" />`
- **BookOpen** (Documentation - standalone): `<BookOpen className="size-5" />`

### Chevron Icons
- **Closed**: `<ChevronDown className="size-4" />`
- **Open**: `<ChevronUp className="size-4" />`
- **Mobile Closed**: `<ChevronRight className="size-4" />`
- **Mobile Open**: `<ChevronDown className="size-4" />`

---

## Typography

### Dropdown Trigger
```
Font: font-sans (Geist Sans)
Size: text-sm (14px)
Weight: font-medium (500)
Line Height: leading-normal
Letter Spacing: tracking-normal
```

### Dropdown Item Label
```
Font: font-sans
Size: text-sm (14px)
Weight: font-medium (500)
```

### Dropdown Item Description
```
Font: font-sans
Size: text-xs (12px)
Weight: font-normal (400)
Color: text-muted-foreground
```

---

## Z-Index Layers

```
Navigation Bar:        z-50
Dropdown Content:      z-50 (same layer, positioned absolutely)
Mobile Menu Sheet:     z-50 (Sheet component handles overlay)
Skip Link (focused):   z-60
```

---

## Accessibility Focus Indicators

### Keyboard Focus (Desktop)
```
┌─────────────────────────┐
│  Features               │  ← ring-2 ring-primary ring-offset-2
└─────────────────────────┘
```

### Keyboard Focus (Mobile)
```
┌────────────────────────────────┐
│  Product ▼                     │  ← ring-2 ring-primary ring-offset-1
└────────────────────────────────┘
```

---

## Component Hierarchy

```
Navigation (organism)
├─ BrandLogo (molecule)
├─ NavigationDropdown (molecule) × 3
│  ├─ DropdownTrigger (atom/Button)
│  └─ NavigationDropdownContent (molecule)
│     └─ DropdownMenuItem (atom) × N
├─ NavLink (molecule) - "Documentation" ⭐ (standalone, DX-first)
├─ IconButton (atom) - GitHub
├─ ThemeToggle (molecule)
├─ Button (atom) - "Join Waitlist"
└─ MobileNavigationSheet (molecule)
   ├─ NavLink (molecule) - "Documentation" ⭐
   ├─ NavLink (molecule) - "GitHub"
   └─ MobileNavigationAccordion (molecule) × 3
      ├─ AccordionTrigger (atom)
      └─ AccordionContent (atom)
         └─ NavLink (molecule) × N
```

---

## Example Code Structure

### Desktop Dropdown
```tsx
<NavigationDropdown
  trigger="Product"
  items={[
    { label: 'Features', href: '/features' },
    { label: 'Pricing', href: '/pricing' },
    { label: 'Use Cases', href: '/use-cases' },
  ]}
/>
```

### Solutions Dropdown (2-column)
```tsx
<NavigationDropdown
  trigger="Solutions"
  columns={2}
  items={[
    {
      label: 'For Developers',
      href: '/developers',
      icon: <Code />,
      description: 'Build AI tools without vendor lock-in',
    },
    // ... more items
    { separator: true }, // Visual divider
    {
      label: 'Startups',
      href: '/industries/startups',
      icon: <Rocket />,
      description: 'Scale your AI infrastructure',
    },
    // ... more items
  ]}
/>
```

---

## Performance Metrics

### Target Metrics
- **First Paint:** No change (navigation is above fold)
- **Interaction Ready:** <100ms (dropdown opens)
- **Animation FPS:** 60fps (smooth transitions)
- **Bundle Size:** +3KB gzipped
- **Accessibility Score:** 100/100 (Lighthouse)

### Optimization Strategies
- Lazy load dropdown content on hover (desktop)
- Preload on focus (keyboard users)
- Use CSS transforms for animations (GPU-accelerated)
- Minimize JavaScript for interactions
- Use native HTML elements where possible

---

## Testing Scenarios

### Desktop
1. ✅ Hover over "Product" → Dropdown opens
2. ✅ Click "Product" → Dropdown toggles
3. ✅ Click outside → Dropdown closes
4. ✅ Press Escape → Dropdown closes
5. ✅ Tab through items → Focus visible
6. ✅ Click dropdown item → Navigates correctly
7. ✅ Hover over "Solutions" → 2-column layout displays
8. ✅ All icons render correctly

### Mobile
1. ✅ Tap hamburger → Menu opens
2. ✅ Tap "Product" → Accordion expands
3. ✅ Tap "Features" → Navigates and closes menu
4. ✅ Tap outside → Menu closes
5. ✅ Swipe down → Menu closes
6. ✅ All touch targets ≥44x44px

### Keyboard
1. ✅ Tab to "Product" → Focus visible
2. ✅ Press Enter → Dropdown opens
3. ✅ Arrow Down → Focus moves to first item
4. ✅ Arrow Up/Down → Navigate items
5. ✅ Press Escape → Dropdown closes
6. ✅ Tab (in dropdown) → Move to next item
7. ✅ Shift+Tab → Move backwards

---

**This mockup provides the visual foundation for implementing the new navigation design.**

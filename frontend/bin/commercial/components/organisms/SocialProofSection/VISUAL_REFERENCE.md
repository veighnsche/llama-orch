# SocialProofSection Visual Reference

Visual comparison and layout breakdown of the redesign.

---

## Before vs After

### BEFORE (Original)
```
┌─────────────────────────────────────────────────────┐
│                                                     │
│        Trusted by Developers Who Value             │
│                  Independence                       │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│   [1,200+]    [500+]    [8,000+]      [€0]        │
│  GitHub     Active      GPUs      Avg Monthly      │
│   Stars   Installations Orchestrated   Cost        │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐     │
│  │ Alex K.   │  │ Sarah M.  │  │ Dr. Thomas│     │
│  │ Solo Dev  │  │ CTO       │  │ Research  │     │
│  │           │  │           │  │ Director  │     │
│  │ "Quote"   │  │ "Quote"   │  │ "Quote"   │     │
│  └───────────┘  └───────────┘  └───────────┘     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### AFTER (Redesigned)
```
┌─────────────────────────────────────────────────────┐
│                                                     │
│        Trusted by Developers Who Value             │
│                  Independence                       │
│                                                     │
│    Local-first AI with zero monthly cost.          │
│      Loved by builders who keep control.           │
│                                                     │
│        [GitHub]  [HN]  [Reddit]  ← Trust strip     │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│   [1,200+]    [500+]    [8,000+]      [€0]        │
│  GitHub ↗   Active      GPUs ⓘ    Avg Monthly     │
│   Stars   Installations Orchestrated   Cost        │
│   (link)                (tooltip)   (success)      │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│         Real teams. Real savings. ← Kicker         │
│                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ 👤 Alex K.  │ │ 👤 Sarah M. │ │ 👤 Dr. Tho. │ │
│  │ Solo Dev    │ │ CTO ✓       │ │ Research    │ │
│  │             │ │ StartupCo   │ │ Director    │ │
│  │ " Quote     │ │ " Quote     │ │ " Quote     │ │
│  │   text...   │ │   text...   │ │   text...   │ │
│  │             │ │             │ │             │ │
│  │ [$80→$0]    │ │ [$500→$0]   │ │             │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │                                               │ │
│  │         [Editorial Image]                    │ │
│  │                                               │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│    Backed by an active community.                  │
│       Join us on GitHub and Discord.               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Layout Breakdown

### Desktop (≥1024px)

```
┌──────────────────────────────────────────────────────────────┐
│ SECTION HEADER (centered, max-w-4xl)                         │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ Title (text-4xl md:text-5xl)                             │ │
│ │ Subtitle (text-lg md:text-xl, muted)                     │ │
│ │ Trust Strip (flex gap-6, opacity-70 hover:100)           │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ METRICS ROW (grid-cols-4, max-w-5xl, gap-6)                 │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                        │
│ │ Stat │ │ Stat │ │ Stat │ │ Stat │                        │
│ │  1   │ │  2   │ │  3   │ │  4   │                        │
│ └──────┘ └──────┘ └──────┘ └──────┘                        │
│                                                              │
│ TESTIMONIALS (grid-cols-12, max-w-6xl)                      │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│ │ col-span-4      │ │ col-span-4      │ │ col-span-4      ││
│ │                 │ │                 │ │                 ││
│ │ TestimonialCard │ │ TestimonialCard │ │ TestimonialCard ││
│ │                 │ │                 │ │                 ││
│ └─────────────────┘ └─────────────────┘ └─────────────────┘│
│                                                              │
│ EDITORIAL IMAGE (hidden lg:block, max-w-6xl)                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 1200x560 WebP image with ring border                     │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ FOOTER (centered, text-sm)                                  │
│ Community CTA + Links                                       │
└──────────────────────────────────────────────────────────────┘
```

### Tablet (768px - 1023px)

```
┌────────────────────────────────────────────┐
│ SECTION HEADER (centered)                  │
│ Title + Subtitle + Trust Strip             │
│                                            │
│ METRICS ROW (grid-cols-4, gap-4)           │
│ [Stat 1] [Stat 2] [Stat 3] [Stat 4]       │
│                                            │
│ TESTIMONIALS (grid-cols-12)                │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│ │ col-4    │ │ col-4    │ │ col-4    │   │
│ │ Card 1   │ │ Card 2   │ │ Card 3   │   │
│ └──────────┘ └──────────┘ └──────────┘   │
│                                            │
│ (Image hidden)                             │
│                                            │
│ FOOTER                                     │
└────────────────────────────────────────────┘
```

### Mobile (<768px)

```
┌──────────────────────┐
│ SECTION HEADER       │
│ Title + Subtitle     │
│ (Trust strip hidden) │
│                      │
│ METRICS (grid-cols-2)│
│ ┌────────┬────────┐  │
│ │ Stat 1 │ Stat 2 │  │
│ ├────────┼────────┤  │
│ │ Stat 3 │ Stat 4 │  │
│ └────────┴────────┘  │
│                      │
│ TESTIMONIALS (stack) │
│ ┌──────────────────┐ │
│ │ Card 1           │ │
│ │ (col-span-12)    │ │
│ └──────────────────┘ │
│ ┌──────────────────┐ │
│ │ Card 2           │ │
│ └──────────────────┘ │
│ ┌──────────────────┐ │
│ │ Card 3           │ │
│ └──────────────────┘ │
│                      │
│ (Image hidden)       │
│                      │
│ FOOTER               │
└──────────────────────┘
```

---

## TestimonialCard Anatomy

### Basic Card
```
┌─────────────────────────────────────────┐
│ ┌─────┐ Name                            │
│ │ 👤  │ Role                            │
│ └─────┘                                 │
│                                         │
│ " Quote text goes here and can span    │
│   multiple lines with proper leading.  │
│                                         │
└─────────────────────────────────────────┘
```

### Enhanced Card (All Features)
```
┌─────────────────────────────────────────┐
│ ┌─────┐ Name [Logo]            [✓ Verified]
│ │ 👤  │ Role                            │
│ └─────┘ Company                         │
│                                         │
│ ★★★★★ (optional rating)                │
│                                         │
│ " Quote text goes here and can span    │
│   multiple lines with proper leading.  │
│                                         │
│ [$500/mo → $0]        2025-10-13 Source │
│ (highlight pill)      (date)   (link)  │
└─────────────────────────────────────────┘
```

### Card States

**Default:**
- Border: `border-border`
- Background: `bg-card/90`
- Shadow: none

**Hover:**
- Border: `border-primary/40`
- Transform: `translate-y-[-2px]`
- Shadow: `shadow-lg`

**Focus (keyboard):**
- Ring: `ring-2 ring-primary`
- Outline: visible

---

## Color Palette

### Text Colors
```
foreground           → Titles, names (highest contrast)
muted-foreground     → Subtitles, quotes, labels
muted-foreground/80  → Footer, meta info
muted-foreground/70  → Trust badges (default)
```

### Accent Colors
```
primary              → Links, decorative quote, verified badge
chart-3 (green)      → Success metric, highlight pills
amber-500            → Star ratings
```

### Background Colors
```
secondary            → Section background
card/90              → Testimonial cards (90% opacity)
primary/10           → Verified badge background
chart-3/10           → Highlight pill background
```

### Border Colors
```
border               → Default card border
primary/40           → Hover card border
border/60            → Image ring
```

---

## Typography Scale

```
Section Title:    text-4xl md:text-5xl (36px → 48px)
Subtitle:         text-lg md:text-xl (18px → 20px)
Stat Value:       text-4xl (36px)
Stat Label:       text-sm (14px)
Kicker:           text-sm uppercase (14px)
Card Name:        font-bold text-base (16px)
Card Role:        text-sm (14px)
Card Quote:       text-sm leading-6 (14px / 24px)
Highlight Pill:   text-xs (12px)
Verified Badge:   text-[11px] (11px)
Footer:           text-sm (14px)
```

---

## Spacing System

```
Section Padding:      py-24 (96px)
Header Margin:        mb-16 (64px)
Metrics Margin:       mb-12 (48px)
Testimonials Margin:  mb-6 (24px)
Footer Margin:        mt-12 (48px)

Card Padding:         p-6 (24px)
Card Gap:             gap-4 (16px)
Grid Gap (desktop):   gap-6 (24px)
Grid Gap (mobile):    gap-4 (16px)
```

---

## Animation Timeline

```
0ms    │ Page loads
       │
100ms  │ ▶ Header fades in (500ms duration)
       │
200ms  │ ▶ Stat 1 slides up (500ms)
       │
300ms  │ ▶ Stat 2 slides up (500ms)
       │
400ms  │ ▶ Stat 3 slides up (500ms)
       │
500ms  │ ▶ Stat 4 slides up (500ms)
       │
600ms  │ ▶ Card 1 zooms in (400ms)
       │ (Header animation completes)
       │
700ms  │ ▶ Card 2 zooms in (400ms)
       │ (Stat 1 animation completes)
       │
800ms  │ ▶ Card 3 zooms in (400ms)
       │ (Stat 2 animation completes)
       │
900ms  │ (Stat 3 animation completes)
       │
1000ms │ (Stat 4 animation completes)
       │ (Card 1 animation completes)
       │
1100ms │ (Card 2 animation completes)
       │
1200ms │ (Card 3 animation completes)
       │ ✓ All animations complete
```

---

## Interactive Elements

### Clickable Areas

```
Trust Badges:     Full badge area (min 44x44px)
GitHub Stars:     Full stat card area
Testimonial Link: "Source" text link
Footer Links:     "GitHub" and "Discord" text links
```

### Hover States

```
Trust Badges:     opacity-70 → opacity-100
GitHub Stars:     opacity-100 → opacity-80
Testimonial Card: border-border → border-primary/40
                  translate-y-0 → translate-y-[-2px]
                  shadow-none → shadow-lg
Footer Links:     no-underline → underline
```

### Focus States

```
All Links:        ring-2 ring-primary ring-offset-2
All Cards:        ring-2 ring-primary (when focused within)
```

---

## Accessibility Features

### Semantic Structure
```html
<section>                    ← SectionContainer
  <h2>                       ← Section title
  <p>                        ← Subtitle
  <div role="group">         ← Stat groups
  <article itemScope>        ← Testimonial cards
    <blockquote>             ← Quote
    <time dateTime>          ← Date
```

### ARIA Labels
```
role="group"                 → Stat containers
aria-label="Stat: ..."       → Stat descriptions
aria-label="Rating: X/5"     → Star ratings
itemProp="author"            → Testimonial author
itemProp="reviewBody"        → Testimonial quote
```

### Keyboard Navigation
```
Tab Order:
1. Trust badges (GitHub, HN, Reddit)
2. GitHub Stars stat (link)
3. Testimonial 1 source link (if present)
4. Testimonial 2 source link (if present)
5. Testimonial 3 source link (if present)
6. Footer GitHub link
7. Footer Discord link
```

---

## Performance Metrics

### Target Metrics
- First Contentful Paint: <1.5s
- Largest Contentful Paint: <2.5s
- Cumulative Layout Shift: <0.1
- Time to Interactive: <3.5s

### Optimization Strategies
1. **Images:** WebP format, lazy loading, responsive sizes
2. **Animations:** CSS transforms only (GPU-accelerated)
3. **Fonts:** Preload critical fonts, font-display: swap
4. **CSS:** Purge unused Tailwind classes
5. **JS:** Minimal JavaScript (Next.js Image only)

---

## Browser Support

### Fully Supported
- Chrome 120+
- Firefox 121+
- Safari 17+
- Edge 120+

### Graceful Degradation
- **No CSS animations:** Content visible, no motion
- **No CSS Grid:** Falls back to flexbox (Tailwind)
- **No JavaScript:** All content visible, images load
- **No WebP:** Falls back to PNG/JPG (Next.js)

---

## Print Styles

When printed:
- Animations disabled
- Colors adjusted for grayscale
- Links show URL in parentheses
- Trust strip hidden
- Editorial image optional (user preference)

---

## Dark Mode Support

All colors use CSS variables:
- `--foreground` → Adapts to theme
- `--muted-foreground` → Adapts to theme
- `--primary` → Adapts to theme
- `--card` → Adapts to theme
- `--border` → Adapts to theme

No hardcoded colors (except gradient avatars).

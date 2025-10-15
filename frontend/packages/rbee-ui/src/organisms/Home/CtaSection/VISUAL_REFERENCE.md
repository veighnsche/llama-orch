# CTASection Visual Reference

## Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [Optional Gradient Background Flourish]                    │
│                                                             │
│              ┌─────────────────────┐                        │
│              │ 🔵 Eyebrow Badge   │  (optional)             │
│              └─────────────────────┘                        │
│                                                             │
│         Stop depending on AI providers.                     │
│           Start building today.                             │
│                                                             │
│    Join 500+ developers who've taken control of             │
│           their AI infrastructure.                          │
│                                                             │
│    ┌──────────────────┐  ┌──────────────────┐              │
│    │ Get started free │  │ View documentation│              │
│    │        →         │  │  📖               │              │
│    └──────────────────┘  └──────────────────┘              │
│                                                             │
│         100% open source. No credit card required.          │
│                  Install in 15 minutes.                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Spacing & Typography

### Section
- **Padding:** `py-24` (96px top/bottom)
- **Border:** `border-b border-border`
- **Background:** `bg-background` + optional gradient overlay

### Container
- **Max width:** `max-w-7xl` (1280px)
- **Padding:** `px-6 lg:px-8` (24px → 32px)

### Content Block
- **Max width:** `max-w-3xl` (768px)
- **Alignment:** `text-center` or `text-left`

### Eyebrow (optional)
- **Spacing:** `mb-3` (12px below)
- **Padding:** `px-3 py-1` (12px × 4px)
- **Style:** `rounded-full border border-border`
- **Typography:** `text-xs font-medium text-muted-foreground`

### Title
- **Typography:** `text-4xl sm:text-5xl font-bold tracking-tight`
- **Color:** `text-foreground`
- **Line height:** Natural (tight tracking compensates)

### Subtitle
- **Spacing:** `mt-3` (12px above)
- **Typography:** `text-lg text-muted-foreground`

### Actions Row
- **Spacing:** `mt-8` (32px above), `gap-3` (12px between)
- **Layout:** `flex-col sm:flex-row` (stack → horizontal)
- **Alignment:** `items-center` + `justify-center` or `justify-start`

### Buttons
- **Size:** `size="lg"` (h-11, px-8)
- **Gap:** Internal gap-2 for icon spacing
- **Primary:** `bg-primary text-primary-foreground hover:bg-primary/90`
- **Secondary:** `bg-transparent border-border text-foreground hover:bg-secondary`

### Trust Note
- **Spacing:** `mt-6` (24px above)
- **Typography:** `text-sm text-muted-foreground`

## Color Palette

| Element | Light Mode | Dark Mode |
|---------|-----------|-----------|
| Background | `hsl(0 0% 100%)` | `hsl(222.2 84% 4.9%)` |
| Foreground | `hsl(222.2 84% 4.9%)` | `hsl(210 40% 98%)` |
| Muted foreground | `hsl(215.4 16.3% 46.9%)` | `hsl(217.9 10.6% 64.9%)` |
| Primary | `hsl(221.2 83.2% 53.3%)` | `hsl(217.2 91.2% 59.8%)` |
| Border | `hsl(214.3 31.8% 91.4%)` | `hsl(217.2 32.6% 17.5%)` |

## Animation Timeline

```
0ms    ─────────────────────────────────────────────
       │
       │ [Eyebrow fades in]
       │
100ms  ─────────────────────────────────────────────
       │
       │ [Title fades + slides in from bottom]
       │
150ms  ─────────────────────────────────────────────
       │
       │ [Subtitle fades in]
       │
200ms  ─────────────────────────────────────────────
       │
       │ [Buttons zoom in]
       │
300ms  ─────────────────────────────────────────────
       │
       │ [Trust note fades in]
       │
600ms  ─────────────────────────────────────────────
       All animations complete
```

## Responsive Behavior

### Mobile (< 640px)
```
┌─────────────────────┐
│                     │
│   [Eyebrow Badge]   │
│                     │
│   Stop depending    │
│   on AI providers.  │
│   Start building    │
│   today.            │
│                     │
│   Join 500+         │
│   developers...     │
│                     │
│ ┌─────────────────┐ │
│ │ Get started free│ │
│ │        →        │ │
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ View docs       │ │
│ │  📖            │ │
│ └─────────────────┘ │
│                     │
│   100% open source  │
│                     │
└─────────────────────┘
```

### Desktop (≥ 640px)
```
┌───────────────────────────────────────────────┐
│                                               │
│            [Eyebrow Badge]                    │
│                                               │
│    Stop depending on AI providers.            │
│         Start building today.                 │
│                                               │
│  Join 500+ developers who've taken control    │
│       of their AI infrastructure.             │
│                                               │
│  ┌──────────────┐  ┌──────────────┐          │
│  │ Get started  │  │ View docs    │          │
│  │      →       │  │  📖         │          │
│  └──────────────┘  └──────────────┘          │
│                                               │
│  100% open source. No credit card required.   │
│           Install in 15 minutes.              │
│                                               │
└───────────────────────────────────────────────┘
```

## Gradient Emphasis Variant

When `emphasis="gradient"`:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ╭─────────────────────────────────────────────────╮        │
│  │         [Radial gradient overlay]              │        │
│  │         primary/10 → transparent                │        │
│  │         70% width, 50% height                   │        │
│  │         centered at top                         │        │
│  ╰─────────────────────────────────────────────────╯        │
│                                                             │
│              [Content as normal]                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

CSS:
```css
.emphasis-gradient::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(
    70% 50% at 50% 0%,
    theme(colors.primary / 10%),
    transparent 60%
  );
  pointer-events: none;
}
```

## Left-Aligned Variant

When `align="left"`:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────┐                                    │
│  │ 🔵 Eyebrow Badge   │                                    │
│  └─────────────────────┘                                    │
│                                                             │
│  Stop depending on AI providers.                            │
│  Start building today.                                      │
│                                                             │
│  Join 500+ developers who've taken control of               │
│  their AI infrastructure.                                   │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Get started free │  │ View documentation│                │
│  │        →         │  │  📖               │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  100% open source. No credit card required.                 │
│  Install in 15 minutes.                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Icon Positioning

### Left Icon
```tsx
iconLeft: GitHubIcon

Renders as:
┌─────────────────────┐
│ 📖 View documentation│
│ ↑                   │
│ mr-2 (8px gap)      │
└─────────────────────┘
```

### Right Icon
```tsx
iconRight: ArrowRight

Renders as:
┌─────────────────────┐
│ Get started free → │
│                  ↑  │
│         ml-2 (8px)  │
└─────────────────────┘

With hover animation:
group-hover:translate-x-1 (4px shift)
```

## Accessibility Features

1. **Semantic HTML:**
   - `<section>` for the container
   - `<h2>` for the title
   - `<Link>` for actions (via Button asChild)

2. **ARIA:**
   - Icons marked `aria-hidden="true"`
   - No redundant labels (button text is clear)

3. **Keyboard:**
   - All buttons are focusable
   - Focus rings visible via Button component
   - Tab order: eyebrow → title → buttons → note

4. **Motion:**
   - All animations use `motion-safe:` prefix
   - Respects `prefers-reduced-motion`

## Usage Examples

### Minimal
```tsx
<CTASection
  title="Get started today"
  primary={{ label: 'Sign up', href: '/signup' }}
/>
```

### Full-featured
```tsx
<CTASection
  eyebrow="Limited time offer"
  title="Stop depending on AI providers. Start building today."
  subtitle="Join 500+ developers who've taken control of their AI infrastructure."
  primary={{ label: 'Get started free', href: '/getting-started', iconRight: ArrowRight }}
  secondary={{ label: 'View documentation', href: '/docs', iconLeft: BookOpen, variant: 'outline' }}
  note="100% open source. No credit card required. Install in 15 minutes."
  emphasis="gradient"
  align="center"
  id="cta"
/>
```

### Left-aligned (product pages)
```tsx
<CTASection
  title="Ready to deploy?"
  subtitle="Get your infrastructure running in minutes."
  primary={{ label: 'Deploy now', href: '/deploy', iconRight: Rocket }}
  secondary={{ label: 'Talk to sales', href: '/contact', variant: 'outline' }}
  align="left"
/>
```

---

**Design Philosophy:** Clean, premium, and flexible. The component adapts to different contexts while maintaining consistent spacing and typography. The optional gradient adds visual interest without overwhelming the content.

# Hero Section Visual QA Checklist

## Quick Visual Verification

### Desktop (lg: 1024px+)

**Layout:**
- [ ] 12-column grid visible with 6-6 split
- [ ] Messaging stack on left (cols 1–6)
- [ ] Visual stack on right (cols 7–12)
- [ ] Min-height ~88vh (not full screen)

**Messaging Stack:**
- [ ] Badge + Docs/GitHub links in top row with separators
- [ ] Kicker text: "Self-Hosted • OpenAI-Compatible • Multi-Backend"
- [ ] Headline: "AI Infrastructure." / "On Your Terms." (text-6xl)
- [ ] Subcopy wraps at ~58 characters
- [ ] Three checkmark bullets visible
- [ ] Two buttons side-by-side (primary + outline)
- [ ] GitHub link below buttons with arrow
- [ ] Four trust badges at bottom

**Visual Stack:**
- [ ] Terminal window with shortened script
- [ ] GPU labels in title case (Workstation, Mac Studio, Gaming PC)
- [ ] Floating KPI card at bottom-left of terminal
- [ ] KPI card shows: GPU Pool, Cost, Latency
- [ ] Network diagram image below terminal (or broken image icon if not created)

**Motion (if not reduced-motion):**
- [ ] Headline fades in (opacity)
- [ ] KPI card slides up + fades in (150ms delay)
- [ ] GitHub arrow translates right on hover

### Tablet (md: 768px–1023px)

**Layout:**
- [ ] Single column stack
- [ ] Messaging first, then visual
- [ ] Terminal centered with max-width

**Messaging:**
- [ ] Headline text-6xl
- [ ] Buttons stack vertically or side-by-side (depends on width)
- [ ] All content readable

**Visual:**
- [ ] Terminal visible
- [ ] KPI card visible
- [ ] Network diagram hidden

### Mobile (sm: <768px)

**Layout:**
- [ ] Full vertical stack
- [ ] Terminal constrained to max-w-[520px]

**Messaging:**
- [ ] Utility row links hidden
- [ ] Headline text-5xl
- [ ] Buttons stack vertically
- [ ] Trust badges wrap to multiple rows

**Visual:**
- [ ] Terminal visible
- [ ] KPI card hidden
- [ ] Network diagram hidden

**Touch Targets:**
- [ ] All buttons ≥44px height
- [ ] Links have adequate spacing

### Accessibility

**Keyboard Navigation:**
- [ ] Tab order: Badge → Docs → GitHub → Headline → Subcopy → Primary CTA → Secondary CTA → GitHub link → (terminal content)
- [ ] All interactive elements have visible focus ring (2px, primary/40, offset-2)
- [ ] Focus ring color contrasts with background

**Screen Reader:**
- [ ] Section has landmark: `aria-labelledby="hero-title"`
- [ ] H1 has `id="hero-title"`
- [ ] Decorative icons have `aria-hidden="true"`
- [ ] "Generating code..." has `aria-live="polite"`
- [ ] Trust badges are in semantic `<ul>` list

**Color Contrast:**
- [ ] Headline text meets 4.5:1 contrast
- [ ] Subcopy text meets 4.5:1 contrast
- [ ] Button text meets 4.5:1 contrast
- [ ] Trust badge text meets 4.5:1 contrast
- [ ] KPI card text meets 4.5:1 contrast on backdrop-blur background

### Dark Mode

**Toggle dark mode and verify:**
- [ ] All semantic tokens update correctly
- [ ] Background: dark navy (#0f172a)
- [ ] Foreground: light (#f1f5f9)
- [ ] Primary: amber (#f59e0b)
- [ ] Secondary background: darker (#1e293b)
- [ ] Border: lighter (#334155)
- [ ] KPI card backdrop: `bg-secondary/30` (more transparent)
- [ ] Terminal colors remain consistent
- [ ] All text remains readable

### Responsive Breakpoints

Test at these exact widths:
- [ ] 320px (minimum mobile)
- [ ] 375px (iPhone SE)
- [ ] 768px (tablet portrait)
- [ ] 1024px (desktop)
- [ ] 1440px (large desktop)

**At 320px:**
- [ ] No horizontal scroll
- [ ] Badge doesn't overflow
- [ ] Buttons don't overflow
- [ ] Text wraps cleanly

### Performance

**Lighthouse Audit (Hero Section):**
- [ ] Performance ≥90
- [ ] Accessibility ≥95
- [ ] Best Practices ≥95
- [ ] SEO ≥90

**Network Tab:**
- [ ] Network diagram image loads with `priority` (if exists)
- [ ] No layout shift when image loads
- [ ] Terminal renders immediately (no flash)

**Reduced Motion:**
- [ ] Set OS to prefer reduced motion
- [ ] Headline appears immediately (no fade)
- [ ] KPI card appears immediately (no slide)
- [ ] GitHub arrow doesn't animate on hover

### Browser Testing

**Chrome/Edge:**
- [ ] All features work
- [ ] Grid layout correct
- [ ] Backdrop-blur renders

**Firefox:**
- [ ] All features work
- [ ] Grid layout correct
- [ ] Backdrop-blur renders

**Safari:**
- [ ] All features work
- [ ] Grid layout correct
- [ ] Backdrop-blur renders
- [ ] SVH units work (or fallback to vh)

### Content Verification

**Copy Accuracy:**
- [ ] Kicker: "Self-Hosted • OpenAI-Compatible • Multi-Backend"
- [ ] Headline: "AI Infrastructure." / "On Your Terms."
- [ ] Subcopy: "Run LLMs on your hardware—across any GPUs and machines. Build with AI, keep control, and avoid vendor lock-in."
- [ ] Bullet 1: "Your GPUs, your network"
- [ ] Bullet 2: "Zero API fees"
- [ ] Bullet 3: "Drop-in OpenAI API"
- [ ] Primary CTA: "Get Started Free"
- [ ] Secondary CTA: "View Docs"
- [ ] Tertiary: "Star on GitHub"
- [ ] Trust 1: "Open Source (GPL-3.0)"
- [ ] Trust 2: "On GitHub"
- [ ] Trust 3: "OpenAI-Compatible"
- [ ] Trust 4: "$0 • No Cloud Required"

**Terminal Content:**
- [ ] Command: `rbee-keeper infer --model llama-3.1-70b`
- [ ] GPU 1: "Workstation" (85%)
- [ ] GPU 2: "Mac Studio" (72%)
- [ ] GPU 3: "Gaming PC" (91%)
- [ ] Cost label: "Local Inference"
- [ ] Cost value: "$0.00"

**KPI Card:**
- [ ] GPU Pool: "3 nodes / 7 GPUs"
- [ ] Cost: "$0.00 / hr" (green/chart-3 color)
- [ ] Latency: "~34 ms"

### Analytics

**Umami Event Tracking:**
- [ ] Primary CTA has `data-umami-event="cta:get-started"`
- [ ] Click primary CTA and verify event fires in Umami dashboard

**Link Attributes:**
- [ ] External links have `rel="noopener noreferrer"`
- [ ] External links have `target="_blank"`
- [ ] Internal links use proper Next.js routing

### Edge Cases

**Long Content:**
- [ ] Headline doesn't break layout if text wraps
- [ ] Subcopy wraps at 58ch max-width
- [ ] Trust badges wrap gracefully on narrow screens

**No JavaScript:**
- [ ] Content still visible and readable
- [ ] No motion (static display)
- [ ] Links still work

**Slow Network:**
- [ ] Terminal renders immediately (no dependency on image)
- [ ] KPI card renders (no external dependencies)
- [ ] Network diagram shows loading state or broken image icon

---

## Quick Test Commands

```bash
# Start dev server
pnpm --filter @rbee/commercial dev

# Build production
pnpm --filter @rbee/commercial build

# Lighthouse audit
pnpm --filter @rbee/commercial build && pnpm --filter @rbee/commercial start
# Then run Lighthouse in Chrome DevTools on http://localhost:3000
```

## Visual Regression

If you have visual regression testing set up:

```bash
# Capture baseline
npm run test:visual -- --update-snapshots

# Compare against baseline
npm run test:visual
```

---

**Status:** ✅ Build successful, ready for visual QA

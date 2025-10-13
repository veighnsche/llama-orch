# Enterprise Hero — Testing & QA Checklist

**Component:** `enterprise-hero.tsx`  
**Status:** ✅ Implementation Complete  
**Testing Required:** Manual QA + Accessibility Audit

---

## Visual Regression Testing

### Desktop (≥1024px)

- [ ] **Layout**
  - [ ] Two-column grid displays correctly
  - [ ] Right audit panel is sticky (`lg:sticky lg:top-24`)
  - [ ] Decorative background image displays (or gracefully hidden if missing)
  - [ ] No layout shift from background image
  - [ ] Radial gradient visible under H1

- [ ] **Typography**
  - [ ] H1 uses `text-balance` and scales to `lg:text-6xl`
  - [ ] Support copy is readable at `text-xl`
  - [ ] All text meets ≥4.5:1 contrast ratio

- [ ] **Proof Tiles**
  - [ ] Three tiles display in equal-height grid
  - [ ] Hover states work (border/background change)
  - [ ] Values and labels are clearly readable

- [ ] **Compliance Proof Bar**
  - [ ] Three chips display inline with icons
  - [ ] Hover states work
  - [ ] Chips wrap gracefully if needed

- [ ] **Audit Console**
  - [ ] Card displays with proper shadow and border
  - [ ] Filter strip shows all four options (All, Auth, Data, Exports)
  - [ ] Four audit events display in list format
  - [ ] Time elements are properly formatted
  - [ ] Footer displays retention and tamper-evident text
  - [ ] Floating badges (EU Only, 32 Types) positioned correctly
  - [ ] Floating badges have proper drop shadows

### Tablet (768px - 1023px)

- [ ] **Layout**
  - [ ] Single-column layout (grid collapses)
  - [ ] Audit console no longer sticky
  - [ ] Decorative background still visible (md:block)
  - [ ] Content remains centered and readable

### Mobile (<768px)

- [ ] **Layout**
  - [ ] Single-column layout
  - [ ] Decorative background hidden
  - [ ] Proof tiles stack vertically or in smaller grid
  - [ ] CTAs stack vertically with proper spacing
  - [ ] Compliance chips wrap to multiple lines
  - [ ] Audit console scales down appropriately
  - [ ] Floating badges don't overlap content

---

## Motion & Animation Testing

- [ ] **Section header animation**
  - [ ] Fades in from 50% opacity
  - [ ] Slides up from bottom (2 units)
  - [ ] Duration: 500ms
  - [ ] No jank or stutter

- [ ] **Proof tiles animation**
  - [ ] Fades in with 120ms delay
  - [ ] Smooth transition

- [ ] **Audit console animation**
  - [ ] Fades in and slides from right
  - [ ] 150ms delay
  - [ ] Smooth transition

- [ ] **Reduced motion**
  - [ ] All animations disabled when `prefers-reduced-motion: reduce`

---

## Accessibility Testing

### Keyboard Navigation

- [ ] **Tab order is logical:**
  1. Eyebrow badge (if focusable)
  2. "Schedule Demo" button
  3. "View Compliance Details" link
  4. Compliance chips (if focusable)
  5. Filter buttons (All, Auth, Data, Exports)
  6. Audit event list items (if focusable)

- [ ] **Focus indicators**
  - [ ] All interactive elements have visible focus rings
  - [ ] Focus rings meet contrast requirements
  - [ ] Focus rings are not clipped or hidden

### Screen Reader Testing

Test with NVDA (Windows), JAWS (Windows), or VoiceOver (macOS):

- [ ] **Section landmark**
  - [ ] Announced as "region" with label "AI Infrastructure That Meets Your Compliance Requirements"

- [ ] **H1**
  - [ ] Properly announced as heading level 1
  - [ ] Text is clear and complete

- [ ] **Proof tiles**
  - [ ] Each tile announced with value and label
  - [ ] Help text is available (sr-only)

- [ ] **CTAs**
  - [ ] "Schedule Demo" announced with aria-label: "Schedule a compliance demo"
  - [ ] "View Compliance Details" announced as link
  - [ ] Both reference "compliance-proof-bar" via aria-describedby

- [ ] **Compliance chips**
  - [ ] Each chip has descriptive aria-label
  - [ ] Icons are properly hidden from screen readers (aria-hidden)

- [ ] **Audit console**
  - [ ] Filter buttons have proper labels ("Filter: All events", etc.)
  - [ ] Event list announced as "list" with label "Recent audit events"
  - [ ] Each event has full context in aria-label
  - [ ] Time elements are properly announced with ISO 8601 format

- [ ] **Floating badges**
  - [ ] Announced as "status" with aria-live="polite"
  - [ ] Descriptive aria-labels are read

### ARIA Validation

- [ ] **No ARIA errors in browser console**
- [ ] **All ARIA attributes are valid:**
  - [ ] `aria-labelledby` on section
  - [ ] `aria-label` on buttons and badges
  - [ ] `aria-describedby` on CTAs
  - [ ] `aria-live` on floating badges
  - [ ] `aria-hidden` on decorative icons

---

## Semantic HTML Validation

- [ ] **Section element**
  - [ ] Has `role="region"`
  - [ ] Has `aria-labelledby="enterprise-hero-h1"`

- [ ] **H1 element**
  - [ ] Has `id="enterprise-hero-h1"`
  - [ ] Only one H1 in the section

- [ ] **Time elements**
  - [ ] All use `<time>` tag
  - [ ] All have valid `dateTime` attribute (ISO 8601)

- [ ] **List elements**
  - [ ] Audit events use `<ul>` and `<li>`
  - [ ] Proper `role="list"` attribute

- [ ] **Button elements**
  - [ ] Filter buttons use `<button type="button">`
  - [ ] CTAs use proper Button component

---

## Color Contrast Testing

Use browser DevTools or axe DevTools:

- [ ] **Text on background**
  - [ ] H1 (foreground on gradient): ≥4.5:1
  - [ ] Support copy (foreground/85 on gradient): ≥4.5:1
  - [ ] Helper text (muted-foreground): ≥4.5:1

- [ ] **Proof tiles**
  - [ ] Value (primary on card/50): ≥4.5:1
  - [ ] Label (muted-foreground on card/50): ≥4.5:1

- [ ] **Compliance chips**
  - [ ] Text (foreground on card/40): ≥4.5:1

- [ ] **Audit console**
  - [ ] Event names (primary on background): ≥4.5:1
  - [ ] User emails (muted-foreground on background): ≥4.5:1
  - [ ] Timestamps (muted-foreground/70 on background): ≥4.5:1
  - [ ] Footer text (foreground/85 on card): ≥4.5:1

- [ ] **Floating badges**
  - [ ] Text (primary on card): ≥4.5:1

---

## Performance Testing

- [ ] **Image loading**
  - [ ] Decorative image uses Next.js Image component
  - [ ] Image has proper width/height to prevent CLS
  - [ ] Image is lazy-loaded (default Next.js behavior)
  - [ ] Image gracefully handles 404 (if missing)

- [ ] **Animation performance**
  - [ ] No layout thrashing during animations
  - [ ] Animations use GPU-accelerated properties (opacity, transform)
  - [ ] No janky scrolling

- [ ] **Lighthouse scores**
  - [ ] Performance: ≥90
  - [ ] Accessibility: 100
  - [ ] Best Practices: ≥90
  - [ ] SEO: 100

---

## Cross-Browser Testing

- [ ] **Chrome/Edge (Chromium)**
  - [ ] All features work
  - [ ] Animations smooth
  - [ ] Layout correct

- [ ] **Firefox**
  - [ ] All features work
  - [ ] Animations smooth
  - [ ] Layout correct

- [ ] **Safari (macOS/iOS)**
  - [ ] All features work
  - [ ] Animations smooth
  - [ ] Layout correct
  - [ ] Sticky positioning works

---

## Responsive Breakpoints

Test at these specific widths:

- [ ] **320px** (iPhone SE)
- [ ] **375px** (iPhone 12/13 Pro)
- [ ] **768px** (iPad portrait)
- [ ] **1024px** (iPad landscape / small desktop)
- [ ] **1280px** (standard desktop)
- [ ] **1920px** (large desktop)

---

## Content Testing

- [ ] **Copy accuracy**
  - [ ] Eyebrow: "EU-Native AI Infrastructure"
  - [ ] H1: "AI Infrastructure That Meets Your Compliance Requirements"
  - [ ] Support: Correct GDPR/SOC2/ISO text
  - [ ] Helper: "EU data residency guaranteed. Audited event types updated quarterly."

- [ ] **Stat values**
  - [ ] 100% — GDPR Compliant
  - [ ] 7 Years — Audit Retention
  - [ ] Zero — US Cloud Deps

- [ ] **Compliance chips**
  - [ ] GDPR Compliant (FileCheck icon)
  - [ ] SOC2 Ready (Shield icon)
  - [ ] ISO 27001 Aligned (Lock icon)

- [ ] **Audit events**
  - [ ] Four events with correct timestamps
  - [ ] All events show "success" status
  - [ ] All emails end in ".eu"

---

## Integration Testing

- [ ] **CTA functionality**
  - [ ] "Schedule Demo" button triggers correct action (when wired)
  - [ ] "View Compliance Details" link navigates to #compliance

- [ ] **Theme switching**
  - [ ] Component looks correct in light mode
  - [ ] Component looks correct in dark mode
  - [ ] All colors use semantic tokens

---

## Edge Cases

- [ ] **Long content**
  - [ ] H1 wraps gracefully if translated to longer language
  - [ ] Support copy wraps properly
  - [ ] Audit events handle long email addresses

- [ ] **Missing image**
  - [ ] Component renders without decorative image
  - [ ] No console errors
  - [ ] No layout shift

- [ ] **Slow network**
  - [ ] Component renders progressively
  - [ ] No FOUC (flash of unstyled content)

---

## Acceptance Criteria

All items must pass before marking as production-ready:

- ✅ **Visual design matches spec**
- ✅ **All animations work smoothly**
- ✅ **Keyboard navigation is complete**
- ✅ **Screen reader announces all content correctly**
- ✅ **Color contrast meets WCAG AA**
- ✅ **Responsive on all breakpoints**
- ✅ **No console errors or warnings**
- ✅ **Lighthouse accessibility score: 100**

---

## Known Issues / Future Improvements

1. **Decorative image**: Asset needs to be created (see `/public/illustrations/README-audit-ledger.md`)
2. **CTA wiring**: "Schedule Demo" button needs to be connected to booking flow
3. **Filter functionality**: Filter buttons are non-functional UI affordances (could be made interactive)
4. **Real-time updates**: Audit events are static (could be connected to live data)

---

## Testing Tools

- **Accessibility**: axe DevTools, WAVE, Lighthouse
- **Screen readers**: NVDA, JAWS, VoiceOver
- **Contrast**: Colour Contrast Analyser, axe DevTools
- **Performance**: Lighthouse, WebPageTest
- **Responsive**: Browser DevTools, BrowserStack

---

**Last Updated:** 2025-10-13  
**Tested By:** [Pending QA]  
**Status:** Ready for QA

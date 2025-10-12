# Navigation QA Checklist

## Visual Verification

### Desktop (≥768px)
- [ ] Navigation height is 56px (h-14)
- [ ] 3-zone grid layout visible: Logo | Links | Actions
- [ ] Links centered in middle zone
- [ ] GitHub + ThemeToggle grouped in ghost toolbar
- [ ] Primary CTA visible and properly styled
- [ ] Premium gradient highlight visible at top edge
- [ ] Logo SVG renders correctly
- [ ] Brand wordmark uses neutral foreground color

### Tablet (640px - 767px)
- [ ] Links collapse to mobile menu
- [ ] Mobile menu toggle visible
- [ ] ThemeToggle remains visible
- [ ] Logo + wordmark remain visible

### Mobile (320px - 639px)
- [ ] All elements remain tappable
- [ ] No horizontal overflow
- [ ] Mobile menu toggle properly sized
- [ ] Safe area padding applied to menu

## Accessibility Testing

### Keyboard Navigation
- [ ] Tab order: Logo → Links (L→R) → GitHub → Theme → CTA → Mobile Menu
- [ ] Skip link appears on first Tab press
- [ ] Skip link navigates to #main on Enter
- [ ] All interactive elements have visible focus rings
- [ ] Focus rings visible in both light and dark themes
- [ ] Mobile menu toggle has proper aria-expanded state

### Screen Reader
- [ ] Navigation has "Primary" label
- [ ] Logo link announces "rbee home"
- [ ] GitHub link announces "Open rbee on GitHub"
- [ ] CTA announces "Join the rbee waitlist"
- [ ] Mobile menu announces as dialog
- [ ] Icons marked aria-hidden (not announced)

### ARIA Attributes
- [ ] `<nav role="navigation" aria-label="Primary">`
- [ ] Mobile toggle: `aria-expanded` toggles correctly
- [ ] Mobile toggle: `aria-controls="mobile-nav"`
- [ ] Mobile menu: `role="dialog" aria-modal="true"`
- [ ] All buttons have `aria-label` attributes

## Interactive Behavior

### Desktop Links
- [ ] Hover changes color to foreground
- [ ] Active page shows underline indicator
- [ ] Underline appears 2px below text
- [ ] Underline smooth opacity transition (200ms)
- [ ] External Docs link opens in new tab
- [ ] External Docs link has rel="noopener"

### Mobile Menu
- [ ] Toggle button opens/closes menu
- [ ] Backdrop click closes menu
- [ ] Menu positioned directly below nav (top-14)
- [ ] Links close menu on click
- [ ] GitHub link visible in menu
- [ ] CTA full-width at bottom
- [ ] Safe area padding visible on notched devices

### GitHub Link
- [ ] Hover shows background (muted/40)
- [ ] Focus ring visible
- [ ] Opens in new tab
- [ ] Icon properly sized (size-5)
- [ ] Tooltip shows "GitHub" on hover

### ThemeToggle
- [ ] Click toggles theme
- [ ] Icon changes (Sun ↔ Moon)
- [ ] Hover shows background
- [ ] Focus ring visible
- [ ] Tooltip shows "Toggle theme" on hover
- [ ] Smooth icon transition (300ms)

### Primary CTA
- [ ] Hover reduces opacity to 85%
- [ ] Focus ring visible
- [ ] Analytics event fires on click (`cta:join-waitlist`)
- [ ] Accessible label present

## Responsive Breakpoints

### 320px (min mobile)
- [ ] No horizontal scroll
- [ ] All tap targets ≥44px
- [ ] Text remains readable
- [ ] Mobile menu usable

### 640px (sm)
- [ ] Padding increases to sm:px-6
- [ ] Layout remains stable

### 768px (md)
- [ ] Links appear in center zone
- [ ] Mobile menu hidden
- [ ] Desktop actions visible
- [ ] 3-zone grid active

### 1024px (lg)
- [ ] Padding increases to lg:px-8
- [ ] Layout remains stable

### 1280px (xl)
- [ ] Link gap increases to gap-8
- [ ] Layout remains balanced

## Color Themes

### Light Theme
- [ ] Background: light with subtle blur
- [ ] Border: visible but subtle
- [ ] Links: muted foreground → foreground on hover
- [ ] Active underline: primary color visible
- [ ] Gradient highlight visible
- [ ] Focus rings visible

### Dark Theme
- [ ] Background: dark with subtle blur
- [ ] Border: visible but subtle
- [ ] Links: muted foreground → foreground on hover
- [ ] Active underline: primary color visible
- [ ] Gradient highlight visible
- [ ] Focus rings visible

## Performance

### Loading
- [ ] Logo image loads immediately (priority flag)
- [ ] No layout shift during hydration
- [ ] ThemeToggle renders without flash

### Interactions
- [ ] Hover states smooth (no jank)
- [ ] Mobile menu opens/closes smoothly
- [ ] Theme toggle transitions smoothly
- [ ] Active state transitions smooth

### Network
- [ ] Only one brand asset loaded (bee-mark.svg)
- [ ] No unnecessary requests
- [ ] No console errors

## Browser Compatibility

### Chrome/Edge
- [ ] All features work
- [ ] Grid layout correct
- [ ] Pseudo-elements render

### Firefox
- [ ] All features work
- [ ] Grid layout correct
- [ ] Pseudo-elements render

### Safari
- [ ] All features work
- [ ] Grid layout correct
- [ ] Pseudo-elements render
- [ ] Safe area padding works on iOS

## Edge Cases

### Long Page Titles
- [ ] Active underline doesn't break layout
- [ ] Links don't wrap awkwardly

### Rapid Theme Switching
- [ ] No visual glitches
- [ ] Icon transitions smooth

### Mobile Menu Spam
- [ ] Open/close remains smooth
- [ ] No state corruption

### Focus Trap
- [ ] Focus remains in mobile menu when open
- [ ] Escape key closes menu (if implemented)

## Analytics

### Event Tracking
- [ ] CTA click fires `cta:join-waitlist` event
- [ ] Event visible in analytics dashboard
- [ ] Event includes proper metadata

## Copy Verification

### Navigation Links
- [ ] "Features" (not "For Features")
- [ ] "Use Cases"
- [ ] "Pricing"
- [ ] "Developers" (not "For Developers")
- [ ] "Providers" (not "For Providers")
- [ ] "Enterprise" (not "For Enterprise")
- [ ] "Docs"

### Tooltips
- [ ] GitHub: "GitHub"
- [ ] ThemeToggle: "Toggle theme"

### ARIA Labels
- [ ] Logo: "rbee home"
- [ ] GitHub: "Open rbee on GitHub"
- [ ] CTA: "Join the rbee waitlist"
- [ ] Mobile toggle: "Toggle menu"

## Final Checks

- [ ] No console errors
- [ ] No console warnings
- [ ] No TypeScript errors
- [ ] No accessibility violations (axe DevTools)
- [ ] Lighthouse accessibility score ≥95
- [ ] All interactive elements ≥44px hit area
- [ ] All text meets WCAG AA contrast ratios

---

**Testing Tools:**
- Chrome DevTools (Responsive mode)
- axe DevTools (Accessibility)
- Lighthouse (Performance + A11y)
- VoiceOver/NVDA (Screen reader)
- Keyboard only (no mouse)

**Test Browsers:**
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile Safari (iOS)
- Chrome Mobile (Android)

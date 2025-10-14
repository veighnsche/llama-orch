# PricingSection QA Checklist

## Visual Testing

### Desktop (≥1024px)
- [ ] Three pricing cards display in equal-height columns
- [ ] Team card has visible glow ring and top accent bar
- [ ] "Most Popular" badge is prominent and readable
- [ ] Billing toggle (Monthly/Yearly) is centered above cards
- [ ] "Save 2 months" badge appears on Yearly button
- [ ] Team price changes from €99/month to €990/year when toggled
- [ ] "2 months free" badge appears next to yearly price
- [ ] Trust badges (Open source, OpenAI-compatible, etc.) display in header
- [ ] Editorial image appears below pricing grid
- [ ] Footer compliance note is visible and readable
- [ ] All animations play smoothly (fade-in, slide-in-from-bottom)
- [ ] Hover states work on cards (slight lift + shadow)
- [ ] Hover states work on buttons

### Tablet (768px - 1023px)
- [ ] Three-column grid maintained
- [ ] Editorial image hidden
- [ ] All content remains readable
- [ ] Touch targets are adequate (≥44px)

### Mobile (<768px)
- [ ] Cards stack vertically
- [ ] Team card appears first
- [ ] Trust badges wrap naturally
- [ ] Billing toggle remains usable
- [ ] All text remains readable
- [ ] No horizontal scroll
- [ ] Touch targets are adequate

## Functional Testing

### Billing Toggle
- [ ] Clicking "Monthly" shows monthly prices
- [ ] Clicking "Yearly" shows yearly prices
- [ ] Toggle state persists during interaction
- [ ] `aria-pressed` attribute updates correctly
- [ ] Visual active state matches logical state

### CTAs & Routing
- [ ] "Download rbee" links to `/download`
- [ ] "Start 30-Day Trial" links to `/signup?plan=team`
- [ ] "Contact Sales" links to `/contact?type=enterprise`
- [ ] All links are clickable
- [ ] Links work with keyboard (Enter key)
- [ ] Links work with assistive tech

### Content Display
- [ ] All feature bullets render correctly
- [ ] Check icons display consistently
- [ ] Footnotes appear under each CTA
- [ ] Price formatting is correct (€ symbol, spacing)
- [ ] Period labels display correctly (/month, /year, forever)

## Accessibility Testing

### Keyboard Navigation
- [ ] Tab order is logical (header → toggle → cards → footer)
- [ ] All interactive elements are focusable
- [ ] Focus indicators are visible
- [ ] Enter/Space activates buttons
- [ ] No keyboard traps

### Screen Reader
- [ ] Section title is announced
- [ ] Subtitle is announced
- [ ] Trust badges are announced (or skipped if decorative)
- [ ] Toggle buttons announce state (pressed/not pressed)
- [ ] Each card is announced as a section
- [ ] Card titles are announced as headings
- [ ] Feature lists are announced as lists
- [ ] CTAs have descriptive labels
- [ ] Footnotes are announced

### ARIA
- [ ] `aria-labelledby` links cards to their titles
- [ ] `aria-label` on feature lists describes content
- [ ] `aria-label` on buttons is descriptive
- [ ] `aria-pressed` on toggle buttons reflects state
- [ ] `aria-hidden` on decorative icons
- [ ] `role="list"` on feature lists

### Color Contrast
- [ ] Title text meets WCAG AA (4.5:1)
- [ ] Body text meets WCAG AA (4.5:1)
- [ ] Muted text meets WCAG AA (4.5:1)
- [ ] Button text meets WCAG AA (4.5:1)
- [ ] Badge text meets WCAG AA (4.5:1)
- [ ] Test in both light and dark modes

### Motion
- [ ] Animations respect `prefers-reduced-motion`
- [ ] No animations when motion is reduced
- [ ] Content is still usable without animations

## Cross-Browser Testing

### Chrome/Edge
- [ ] Layout renders correctly
- [ ] Animations work
- [ ] Backdrop blur works
- [ ] All interactions work

### Firefox
- [ ] Layout renders correctly
- [ ] Animations work
- [ ] Backdrop blur works
- [ ] All interactions work

### Safari
- [ ] Layout renders correctly
- [ ] Animations work
- [ ] Backdrop blur works (with prefix)
- [ ] All interactions work

## Performance

- [ ] Editorial image loads with priority
- [ ] No layout shift when image loads
- [ ] Toggle state updates instantly
- [ ] No janky animations
- [ ] Page loads in <3s on 3G

## Content Accuracy

### Home/Lab Plan
- [ ] Price: €0
- [ ] Period: forever
- [ ] CTA: "Download rbee"
- [ ] Footnote: "Local use. No feature gates."
- [ ] 5 feature bullets (correct copy)

### Team Plan
- [ ] Price (Monthly): €99/month
- [ ] Price (Yearly): €990/year
- [ ] Badge: "Most Popular"
- [ ] Save badge: "2 months free" (yearly only)
- [ ] CTA: "Start 30-Day Trial"
- [ ] Footnote: "Cancel anytime during trial."
- [ ] 5 feature bullets (correct copy)

### Enterprise Plan
- [ ] Price: Custom
- [ ] CTA: "Contact Sales"
- [ ] Footnote: "We'll reply within 1 business day."
- [ ] 5 feature bullets (correct copy)

## Edge Cases

- [ ] Very long feature text doesn't break layout
- [ ] Missing editorial image doesn't break layout
- [ ] JavaScript disabled: content still accessible
- [ ] High zoom (200%): content remains usable
- [ ] Small viewport (320px): no horizontal scroll

## Regression Testing

- [ ] Other sections on page still render correctly
- [ ] Global styles not affected
- [ ] Other pricing components (if any) still work
- [ ] Build succeeds without errors
- [ ] No console errors in browser

---

## Test Environments

- [ ] Local development (`npm run dev`)
- [ ] Production build (`npm run build && npm start`)
- [ ] Staging environment (if available)

## Sign-Off

- [ ] Visual design approved
- [ ] Accessibility audit passed
- [ ] Performance metrics acceptable
- [ ] Content accuracy verified
- [ ] Cross-browser testing complete

**Tested by:** _________________  
**Date:** _________________  
**Notes:** _________________

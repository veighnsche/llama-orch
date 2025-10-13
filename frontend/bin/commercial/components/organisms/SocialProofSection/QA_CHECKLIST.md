# SocialProofSection QA Checklist

**Version:** 1.0  
**Date:** 2025-10-13  
**Tester:** ___________

---

## Visual Regression

### Desktop (1920x1080)
- [ ] Section header displays title + subtitle + trust strip
- [ ] Trust strip shows 3 badges (GitHub, HN, Reddit) with hover effects
- [ ] Metrics row shows 4 stats in single row with proper spacing
- [ ] GitHub Stars stat is clickable and opens in new tab
- [ ] Testimonials display in 3-column grid
- [ ] All 3 testimonial cards have equal height
- [ ] Sarah M. card shows "Verified" badge
- [ ] Highlight pills display correctly ("$80/mo → $0", "$500/mo → $0")
- [ ] Editorial image displays below testimonials
- [ ] Footer reassurance shows with GitHub/Discord links

### Tablet (768x1024)
- [ ] Trust strip remains visible
- [ ] Metrics remain in 4-column layout
- [ ] Testimonials remain in 3-column layout
- [ ] Editorial image hidden
- [ ] All text remains readable

### Mobile (375x667)
- [ ] Trust strip hidden
- [ ] Metrics collapse to 2-column grid (2 rows)
- [ ] Testimonials stack vertically (1 column)
- [ ] Highlight pills don't overflow
- [ ] Footer links remain accessible
- [ ] All touch targets ≥44x44px

---

## Animations

### On Page Load
- [ ] Header fades in (`fade-in-50 duration-500`)
- [ ] Metrics slide in from bottom with stagger (100/200/300/400ms delays)
- [ ] Testimonials zoom in with stagger (100/200/300ms delays)

### On Hover
- [ ] Trust badges increase opacity (70% → 100%)
- [ ] GitHub Stars stat reduces opacity (100% → 80%)
- [ ] Testimonial cards translate up 2px and show shadow
- [ ] Testimonial cards border changes to `primary/40`
- [ ] Footer links underline on hover

### Reduced Motion
- [ ] All animations disabled when `prefers-reduced-motion: reduce`
- [ ] Layout remains functional without animations

---

## Accessibility

### Keyboard Navigation
- [ ] All links focusable with Tab key
- [ ] Focus indicators visible (ring-2 ring-primary)
- [ ] GitHub Stars link opens with Enter/Space
- [ ] Trust badges accessible via keyboard
- [ ] Footer links accessible via keyboard

### Screen Reader
- [ ] Section title announced correctly
- [ ] Subtitle announced after title
- [ ] Each stat group has proper aria-label
- [ ] Testimonial cards announced as "Review" (Schema.org)
- [ ] Author names announced with itemProp
- [ ] Quote text announced with itemProp
- [ ] Verified badge announced
- [ ] Source links announced with "opens in new window"

### ARIA & Semantics
- [ ] Stat groups have `role="group"` and `aria-label`
- [ ] Testimonial cards use `<article>` tag
- [ ] Quotes use `<blockquote>` tag
- [ ] Dates use `<time>` tag with `dateTime` attribute
- [ ] Links have `rel="noopener noreferrer"` for external URLs

### Color Contrast
- [ ] Title text (foreground) passes WCAG AA (4.5:1)
- [ ] Subtitle text (muted-foreground) passes WCAG AA
- [ ] Stat values (primary/chart-3) pass WCAG AA
- [ ] Stat labels (muted-foreground) pass WCAG AA
- [ ] Quote text (muted-foreground) passes WCAG AA
- [ ] Highlight pills (chart-3 on chart-3/10) pass WCAG AA
- [ ] Verified badge (primary on primary/10) passes WCAG AA

---

## Functionality

### Links
- [ ] GitHub Stars link → `https://github.com/veighnsche/llama-orch`
- [ ] GitHub trust badge → `https://github.com/veighnsche/llama-orch`
- [ ] HN trust badge → `#` (placeholder)
- [ ] Reddit trust badge → `#` (placeholder)
- [ ] Footer GitHub link → `https://github.com/veighnsche/llama-orch`
- [ ] Footer Discord link → `#` (placeholder)
- [ ] All external links open in new tab
- [ ] All external links have `rel="noopener noreferrer"`

### Tooltips
- [ ] GitHub badge shows "Star us on GitHub" on hover
- [ ] HN badge shows "Discussed on Hacker News" on hover
- [ ] Reddit badge shows "Join our community on Reddit" on hover
- [ ] GPUs Orchestrated stat shows "Cumulative across clusters" on hover

### TestimonialCard Props
- [ ] Alex K. card renders without optional props (backward-compatible)
- [ ] Sarah M. card renders with `company`, `verified`, `highlight`
- [ ] Dr. Thomas R. card renders without optional props
- [ ] Avatar gradients render correctly (blue, amber, green)
- [ ] Decorative quote character (`"`) displays before quote text
- [ ] Quote text clamps to 6 lines on mobile
- [ ] Quote text expands on desktop (no clamp)

---

## Content Verification

### Copy Accuracy
- [ ] Title: "Trusted by Developers Who Value Independence"
- [ ] Subtitle: "Local-first AI with zero monthly cost. Loved by builders who keep control."
- [ ] Kicker: "Real teams. Real savings."
- [ ] Footer: "Backed by an active community. Join us on GitHub and Discord."
- [ ] Alex K. quote matches spec
- [ ] Sarah M. quote matches spec
- [ ] Dr. Thomas R. quote matches spec

### Metrics
- [ ] GitHub Stars: "1,200+"
- [ ] Active Installations: "500+"
- [ ] GPUs Orchestrated: "8,000+"
- [ ] Avg Monthly Cost: "€0" (success variant)

### Highlights
- [ ] Alex K.: "$80/mo → $0"
- [ ] Sarah M.: "$500/mo → $0"
- [ ] Dr. Thomas R.: (none)

---

## Performance

### Load Time
- [ ] Section renders within 1 second on 3G
- [ ] Animations don't cause jank (60fps)
- [ ] Image lazy-loads (not blocking initial render)

### Bundle Size
- [ ] No new external dependencies added
- [ ] Tailwind classes purged correctly
- [ ] No unused CSS in production build

---

## Browser Compatibility

### Desktop
- [ ] Chrome 120+ ✓
- [ ] Firefox 121+ ✓
- [ ] Safari 17+ ✓
- [ ] Edge 120+ ✓

### Mobile
- [ ] iOS Safari 17+ ✓
- [ ] Chrome Android 120+ ✓
- [ ] Samsung Internet 23+ ✓

---

## Edge Cases

### Missing Image
- [ ] Section renders correctly if `social-proof-collage.webp` missing
- [ ] No broken image icon shown (hidden on lg+ only)

### Long Content
- [ ] Quote text >200 chars clamps correctly on mobile
- [ ] Highlight pills with long text don't break layout
- [ ] Company names >30 chars wrap correctly

### No JavaScript
- [ ] All content visible without JS
- [ ] Links functional without JS
- [ ] Layout remains intact without JS

---

## Sign-off

- [ ] All critical issues resolved
- [ ] All accessibility issues resolved
- [ ] All responsive breakpoints tested
- [ ] All animations tested (with and without motion)
- [ ] All links verified
- [ ] Content proofread

**Tester Signature:** ___________  
**Date:** ___________  
**Status:** [ ] PASS [ ] FAIL [ ] NEEDS REVISION

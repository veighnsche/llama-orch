# SocialProofSection Developer Guide

Quick reference for using the redesigned SocialProofSection and enhanced TestimonialCard.

---

## TestimonialCard API

### Minimal Usage (Backward-Compatible)
```tsx
<TestimonialCard
  name="John Doe"
  role="Developer"
  quote="This product changed my workflow completely."
  avatar={{ from: 'blue-400', to: 'blue-600' }}
/>
```

### Full-Featured Usage
```tsx
<TestimonialCard
  // Required
  name="Jane Smith"
  role="CTO"
  quote="We saved $500/month by switching to rbee."
  
  // Optional - Avatar
  avatar={{ from: 'amber-400', to: 'amber-600' }}
  // OR: avatar="https://example.com/avatar.jpg"
  
  // Optional - Company
  company={{ name: 'StartupCo', logo: '/logos/startup.png' }}
  
  // Optional - Verification
  verified={true}
  
  // Optional - Source
  link="https://twitter.com/user/status/123"
  date="2025-10-13"
  
  // Optional - Rating
  rating={5}
  
  // Optional - Highlight
  highlight="$500/mo → $0"
/>
```

### Props Reference

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `name` | `string` | ✅ | Person's name |
| `role` | `string` | ✅ | Person's role/title |
| `quote` | `string` | ✅ | Testimonial text |
| `avatar` | `string \| { from: string; to: string }` | ❌ | Image URL or gradient colors |
| `company` | `{ name: string; logo?: string }` | ❌ | Company info with optional logo |
| `verified` | `boolean` | ❌ | Show verified badge |
| `link` | `string` | ❌ | Source link (opens in new tab) |
| `date` | `string` | ❌ | ISO date or human-readable string |
| `rating` | `1 \| 2 \| 3 \| 4 \| 5` | ❌ | Star rating (1-5) |
| `highlight` | `string` | ❌ | Payoff badge (e.g., "$500/mo → $0") |
| `className` | `string` | ❌ | Additional CSS classes |

### Avatar Options

**Gradient (recommended for consistency):**
```tsx
avatar={{ from: 'blue-400', to: 'blue-600' }}
avatar={{ from: 'amber-400', to: 'amber-600' }}
avatar={{ from: 'green-400', to: 'green-600' }}
```

**Image URL:**
```tsx
avatar="https://example.com/avatar.jpg"
avatar="/images/team/john-doe.jpg"
```

**Default (no avatar prop):**
- Falls back to `from-primary to-chart-2` gradient

---

## SocialProofSection Customization

### Updating Metrics

Edit the metrics in `SocialProofSection.tsx`:

```tsx
<StatCard value="1,200+" label="GitHub Stars" />
<StatCard value="500+" label="Active Installations" />
<StatCard value="8,000+" label="GPUs Orchestrated" />
<StatCard value="€0" label="Avg Monthly Cost" variant="success" />
```

**Adding a link to a metric:**
```tsx
<a href="https://example.com" target="_blank" rel="noopener noreferrer">
  <StatCard value="1,200+" label="GitHub Stars" />
</a>
```

### Updating Trust Badges

Edit the `trustBadges` array:

```tsx
const trustBadges = [
  { name: 'GitHub', url: 'https://github.com/...', tooltip: 'Star us on GitHub' },
  { name: 'HN', url: 'https://news.ycombinator.com/...', tooltip: 'Discussed on HN' },
  { name: 'Reddit', url: 'https://reddit.com/r/...', tooltip: 'Join our subreddit' },
]
```

### Adding/Removing Testimonials

```tsx
<div className="grid grid-cols-12 gap-6">
  <div className="col-span-12 md:col-span-4 motion-safe:animate-in ...">
    <TestimonialCard {...props1} />
  </div>
  <div className="col-span-12 md:col-span-4 motion-safe:animate-in ...">
    <TestimonialCard {...props2} />
  </div>
  <div className="col-span-12 md:col-span-4 motion-safe:animate-in ...">
    <TestimonialCard {...props3} />
  </div>
  {/* Add more cards here - adjust md:col-span-X for layout */}
</div>
```

**For 2 testimonials:** Use `md:col-span-6`  
**For 4 testimonials:** Use `md:col-span-3`  
**For 5+ testimonials:** Consider carousel or pagination

---

## Animation Customization

### Disabling Animations

Remove `motion-safe:` prefixes:

```tsx
// Before
className="motion-safe:animate-in motion-safe:fade-in-50"

// After (no animation)
className=""
```

### Adjusting Animation Timing

```tsx
// Faster
motion-safe:duration-300

// Slower
motion-safe:duration-700

// Custom delay
motion-safe:delay-[500ms]
```

### Stagger Pattern

Current stagger: 100ms increments
```tsx
motion-safe:delay-100  // 1st item
motion-safe:delay-200  // 2nd item
motion-safe:delay-300  // 3rd item
```

To change increment:
```tsx
motion-safe:delay-150  // 1st item
motion-safe:delay-300  // 2nd item
motion-safe:delay-450  // 3rd item
```

---

## Styling Customization

### Card Hover Effects

Current hover state:
```tsx
hover:border-primary/40
motion-safe:hover:translate-y-[-2px]
motion-safe:hover:shadow-lg
```

To disable hover lift:
```tsx
// Remove translate-y
hover:border-primary/40
motion-safe:hover:shadow-lg
```

### Highlight Pill Colors

Current: `bg-chart-3/10 text-chart-3` (green success color)

Alternatives:
```tsx
// Blue
bg-primary/10 text-primary

// Amber
bg-amber-500/10 text-amber-500

// Red
bg-red-500/10 text-red-500
```

### Quote Clamping

Current: 6 lines on mobile, full on desktop
```tsx
line-clamp-6 md:line-clamp-none
```

Alternatives:
```tsx
// Always clamp to 4 lines
line-clamp-4

// Never clamp
// (remove line-clamp classes)

// Clamp to 8 lines on mobile
line-clamp-8 md:line-clamp-none
```

---

## Responsive Breakpoints

### Tailwind Breakpoints Used

| Prefix | Min Width | Usage |
|--------|-----------|-------|
| `md:` | 768px | Trust strip, 4-col metrics, 3-col testimonials |
| `lg:` | 1024px | Editorial image, tighter gaps |

### Adjusting Breakpoints

To show trust strip on mobile:
```tsx
// Before
className="hidden md:flex"

// After
className="flex"
```

To change testimonial grid breakpoint:
```tsx
// Before
col-span-12 md:col-span-4

// After (3-col at lg instead of md)
col-span-12 lg:col-span-4
```

---

## Accessibility Best Practices

### Adding ARIA Labels

For custom stats:
```tsx
<div role="group" aria-label="Stat: Custom Metric">
  <StatCard value="123" label="Custom Metric" />
</div>
```

### Adding Tooltips

```tsx
<div title="Additional context shown on hover">
  <StatCard value="123" label="Metric" />
</div>
```

### Link Accessibility

Always include for external links:
```tsx
target="_blank"
rel="noopener noreferrer"
```

For screen readers:
```tsx
aria-label="Opens in new window"
```

---

## Common Patterns

### Verified Testimonial with Source
```tsx
<TestimonialCard
  name="Jane Doe"
  role="CEO"
  quote="..."
  verified
  link="https://twitter.com/jane/status/123"
  date="2025-10-13"
  highlight="$1000/mo → $0"
/>
```

### Company Testimonial with Logo
```tsx
<TestimonialCard
  name="John Smith"
  role="CTO"
  company={{ name: 'TechCorp', logo: '/logos/techcorp.png' }}
  quote="..."
  avatar="https://example.com/john.jpg"
/>
```

### Simple Testimonial (No Extras)
```tsx
<TestimonialCard
  name="Alex K."
  role="Developer"
  quote="..."
  avatar={{ from: 'blue-400', to: 'blue-600' }}
/>
```

---

## Troubleshooting

### Image Not Showing

**Issue:** Editorial image not displaying

**Solution:**
1. Ensure image exists at `public/images/social-proof-collage.webp`
2. Check Next.js Image optimization is enabled
3. Verify image dimensions (1200x560)

### Animations Not Working

**Issue:** Cards not animating on load

**Solution:**
1. Check browser supports CSS animations
2. Verify `prefers-reduced-motion` is not set
3. Ensure Tailwind `animate-in` plugin is installed

### Layout Breaking on Mobile

**Issue:** Cards overflowing or misaligned

**Solution:**
1. Check `col-span-12` is set for mobile
2. Verify `gap-6` is not too large
3. Test with long content (quotes >200 chars)

### TypeScript Errors

**Issue:** Props not recognized

**Solution:**
1. Ensure `TestimonialCardProps` interface is imported
2. Check all required props are provided (`name`, `role`, `quote`)
3. Verify optional props match type definitions

---

## Performance Tips

1. **Lazy load images:** Next.js Image component handles this automatically
2. **Reduce animation complexity:** Use `transform` and `opacity` only (GPU-accelerated)
3. **Minimize re-renders:** Memoize testimonial data if fetched from API
4. **Optimize images:** Use WebP format, compress to <100KB
5. **Defer non-critical content:** Load editorial image after testimonials

---

## Migration from Old Version

### Breaking Changes
None! All existing `TestimonialCard` usage remains compatible.

### New Features (Opt-In)
- `company` prop for company info
- `verified` prop for verified badge
- `link` prop for source attribution
- `date` prop for timestamp
- `rating` prop for star ratings
- `highlight` prop for payoff badges

### Recommended Updates
1. Add `highlight` to testimonials with cost savings
2. Add `verified` to testimonials with source links
3. Add `link` to testimonials from social media
4. Update copy to match new tighter style

---

## Support

For questions or issues:
- Check `QA_CHECKLIST.md` for testing guidance
- Review `REDESIGN_SUMMARY.md` for implementation details
- Open an issue on GitHub for bugs

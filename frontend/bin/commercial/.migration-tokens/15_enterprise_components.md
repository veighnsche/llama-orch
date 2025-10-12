# Work Unit 15: Enterprise Page Components

**Priority:** LOW  
**Directory:** `/components/enterprise/`

**Files:**
- `enterprise-hero.tsx`
- `enterprise-problem.tsx`
- `enterprise-solution.tsx`
- `enterprise-features.tsx`
- `enterprise-compliance.tsx`
- `enterprise-security.tsx`
- `enterprise-how-it-works.tsx`
- `enterprise-use-cases.tsx`
- `enterprise-comparison.tsx`
- `enterprise-testimonials.tsx`
- `enterprise-cta.tsx`

---

## Migration Strategy

The enterprise page components follow the same patterns as the main landing page components. Apply the same token replacements with enterprise-specific considerations.

### Common Patterns

| Current Pattern | Token Replacement |
|----------------|-------------------|
| `bg-slate-50` | `bg-secondary` |
| `bg-white` | `bg-background` or `bg-card` |
| `bg-slate-900` | `bg-background` (dark mode) or `bg-card` |
| `text-slate-900` | `text-foreground` or `text-card-foreground` |
| `text-slate-600` | `text-muted-foreground` |
| `text-slate-300` | `text-muted-foreground` |
| `text-amber-500/600` | `text-primary` |
| `bg-amber-500/600` | `bg-primary` |
| `text-blue-400/500/600` | `text-chart-2` (info) |
| `bg-blue-100` | `bg-chart-2/10` |
| `text-green-400/500/600` | `text-chart-3` (success) |
| `bg-green-100` | `bg-chart-3/10` |
| `text-red-400/500/600` | `text-destructive` |
| `bg-red-100` | `bg-destructive/10` |
| `border-slate-200` | `border-border` |

### Component-Specific Notes

#### enterprise-hero.tsx
- Similar to main `hero-section.tsx`
- Professional, serious tone → May use darker backgrounds
- Trust indicators → Use `text-chart-3` for checkmarks
- Badge → `bg-primary/10 border-primary/20 text-primary`

#### enterprise-problem.tsx
- Similar to main `problem-section.tsx`
- Use `destructive` token for risk/problem indicators
- Compliance risks → `text-destructive`
- Security concerns → `text-destructive`

#### enterprise-solution.tsx
- Similar to main `solution-section.tsx`
- Architecture diagram → Use `bg-primary` for orchestrator
- Security features → Use `text-chart-2` (blue for security)
- Compliance features → Use `text-chart-3` (green for compliance)

#### enterprise-features.tsx
- Similar to main `features-section.tsx`
- Feature cards → `bg-card border-border`
- Icon backgrounds → Use semantic colors
- Enterprise-grade indicators → `text-primary`

#### enterprise-compliance.tsx
- Compliance badges → `bg-chart-3/10 border-chart-3/20` (green for compliant)
- Audit trail features → `text-chart-2` (blue for info)
- Retention policies → `text-foreground`
- Callout boxes → Use `chart-3/10` for compliance success

#### enterprise-security.tsx
- Security features → `text-chart-2` (blue)
- Threat indicators → `text-destructive` (red)
- Secure states → `text-chart-3` (green)
- Code blocks → `bg-muted` with syntax highlighting
- Security badges → `bg-chart-2/10 border-chart-2/20`

#### enterprise-how-it-works.tsx
- Similar to main `how-it-works-section.tsx`
- Step numbers → `bg-primary text-primary-foreground`
- Deployment diagrams → Use hierarchy with tokens
- Infrastructure visuals → `bg-card` with `border-border`

#### enterprise-use-cases.tsx
- Similar to main `use-cases-section.tsx`
- Industry-specific cards → `bg-card border-border`
- Compliance indicators → `text-chart-3`
- ROI metrics → `text-primary`

#### enterprise-comparison.tsx
- Similar to main `comparison-section.tsx`
- Table → `bg-card border-border`
- rbee column highlight → `bg-primary/5`
- Check icons → `text-chart-3`
- X icons → `text-destructive`

#### enterprise-testimonials.tsx
- Similar to main `social-proof-section.tsx`
- Testimonial cards → `bg-card border-border`
- Company logos → Keep as-is
- Quote text → `text-muted-foreground`

#### enterprise-cta.tsx
- Similar to main `cta-section.tsx`
- Primary button → `bg-primary hover:bg-primary/90 text-primary-foreground`
- Secondary button → `border-border hover:bg-secondary`
- Professional tone → May use solid backgrounds instead of gradients

---

## Implementation Approach

### Step 1: Read Each File
Read each component file to identify hardcoded colors.

### Step 2: Apply Pattern Matching
Use the common patterns table above to replace colors systematically.

### Step 3: Preserve Enterprise Semantics
Ensure semantic colors are preserved:
- **Blue:** Security, trust, information
- **Green:** Compliance, success, approved
- **Amber:** Primary brand, CTAs, enterprise features
- **Red:** Risks, threats, non-compliant

### Step 4: Professional Aesthetic
Enterprise pages should feel:
- **Trustworthy:** Clear contrast, readable
- **Professional:** Clean, minimal decorative elements
- **Secure:** Blue tones for security features
- **Compliant:** Green indicators for compliance

### Step 5: Test Each Component
Verify each component in both light and dark modes.

---

## Verification Checklist

For each component:
- [ ] Renders correctly in light mode
- [ ] Renders correctly in dark mode
- [ ] Semantic colors preserved (security=blue, compliance=green)
- [ ] Professional aesthetic maintained
- [ ] Trust indicators are visible
- [ ] Compliance badges are clear
- [ ] Security features stand out
- [ ] Buttons have good contrast
- [ ] Cards are readable
- [ ] Tables are scannable
- [ ] No hardcoded `slate-*`, `amber-*`, `blue-*`, `green-*`, `red-*` classes remain

---

## Estimated Complexity

**Medium-High** - 11 components with enterprise-specific semantics. More complex than developers page due to compliance/security requirements.

---

## Notes

### Enterprise Color Semantics

- **Security features:** Blue (`chart-2`) - trust, protection
- **Compliance features:** Green (`chart-3`) - approved, compliant
- **Risks/threats:** Red (`destructive`) - danger, non-compliant
- **Enterprise features:** Amber (`primary`) - premium, brand

### Compliance & Security Callouts

Use distinct colors for different types of callouts:
- **Compliance:** `bg-chart-3/10 border-chart-3/20` (green)
- **Security:** `bg-chart-2/10 border-chart-2/20` (blue)
- **Risk:** `bg-destructive/10 border-destructive/20` (red)
- **Feature:** `bg-primary/10 border-primary/20` (amber)

### Professional Tone

Enterprise pages should avoid:
- Overly playful colors
- Excessive gradients
- Decorative elements that reduce trust

Maintain:
- High contrast for readability
- Clear visual hierarchy
- Professional color palette
- Consistent spacing

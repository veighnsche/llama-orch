# Deployment Flow Decorative Image

## Required Asset: `deployment-flow.webp`

**Dimensions:** 1200×640px  
**Format:** WebP (optimized for web)  
**Purpose:** Decorative background illustration for Enterprise Deployment Process section

### Design Brief

Create an abstract, premium-quality illustration that conveys:

- **EU-blue flow diagram** with four distinct checkpoints
- **Connecting lines** suggesting sequential deployment stages
- **Compliance handoffs** visual metaphor through validated transitions
- **Enterprise deployment stages** (assessment → deployment → validation → launch)
- **Premium dark UI aesthetic** matching the enterprise theme
- **Subtle amber accents** at validation/checkpoint nodes

### Visual Elements

- Four distinct checkpoint nodes representing the deployment stages
- Flowing, connected paths between checkpoints (left-to-right or top-to-bottom)
- EU color palette: blues (#3b82f6, #2563eb) with amber highlights (#f59e0b)
- Abstract representation of:
  - Compliance assessment (initial analysis)
  - Infrastructure deployment (server/cloud setup)
  - Validation checkpoints (audit verification)
  - Production launch (go-live)
- Minimalist, modern design language
- Should work well at low opacity (15%) with slight blur

### Technical Requirements

- Dimensions: 1200×640px
- Format: WebP
- File size: < 200KB (optimized)
- Color space: sRGB
- Transparent or dark background (#0f172a or similar)

### Placement

The image is positioned:
- `left: 50%`
- `top: 2rem` (8px)
- `transform: translateX(-50%)` (centered)
- `width: 48rem`
- `opacity: 0.15`
- `blur: 0.5px`
- `z-index: -10`
- Hidden on mobile, visible on `md:` breakpoint and up

### Visual Inspiration

Think of:
- Deployment pipelines with stages
- Compliance validation workflows
- Enterprise onboarding journeys
- Sequential process flows with checkpoints
- Audit trail handoffs between stages

### Key Differences from Other Assets

- **audit-ledger.webp** (EnterpriseHero): Single audit trail flow
- **eu-ledger-grid.webp** (EnterpriseSolution): Structured grid with checkpoints
- **compliance-ledger.webp** (EnterpriseCompliance): Multi-path compliance convergence
- **security-mesh.webp** (EnterpriseSecurity): Interconnected mesh with five node clusters
- **deployment-flow.webp** (THIS): Sequential four-stage flow with connecting lines, deployment journey

### Alternative

Until the asset is created, the component will gracefully handle the missing image (Next.js Image component will show a placeholder or fail silently with the `hidden md:block` classes preventing layout shift).

---

**Related Assets:**
- `/illustrations/audit-ledger.webp` (EnterpriseHero)
- `/decor/eu-ledger-grid.webp` (EnterpriseSolution)
- `/decor/compliance-ledger.webp` (EnterpriseCompliance)
- `/decor/security-mesh.webp` (EnterpriseSecurity)

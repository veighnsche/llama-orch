# Sector Grid Decorative Image

## Required Asset: `sector-grid.webp`

**Dimensions:** 1200×640px  
**Format:** WebP (optimized for web)  
**Purpose:** Decorative background illustration for Enterprise Use Cases / Regulated Industries section

### Design Brief

Create an abstract, premium-quality illustration that conveys:

- **EU-blue grid** of industry tiles representing four sectors
- **Industry tiles** for finance, healthcare, legal, and government
- **Soft amber accents** at tile intersections or validation points
- **Premium dark UI aesthetic** matching the enterprise theme
- **Compliance theme** suggesting regulated industries and audit-ready infrastructure

### Visual Elements

- Four distinct tile/section areas representing the industries
- Grid structure with clean, geometric divisions
- EU color palette: blues (#3b82f6, #2563eb) with amber highlights (#f59e0b)
- Abstract representation of:
  - Financial services (banking, insurance)
  - Healthcare (hospitals, medical)
  - Legal services (law firms, documents)
  - Government (public sector, sovereignty)
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
- `top: 1.5rem` (6px)
- `transform: translateX(-50%)` (centered)
- `width: 50rem`
- `opacity: 0.15`
- `blur: 0.5px`
- `z-index: -10`
- Hidden on mobile, visible on `md:` breakpoint and up

### Visual Inspiration

Think of:
- Industry sector grids or matrices
- Compliance framework mappings
- Regulated industry landscapes
- Four-quadrant strategic frameworks
- Enterprise industry segmentation

### Key Differences from Other Assets

- **audit-ledger.webp** (EnterpriseHero): Single audit trail flow
- **eu-ledger-grid.webp** (EnterpriseSolution): Structured grid with checkpoints
- **compliance-ledger.webp** (EnterpriseCompliance): Multi-path compliance convergence
- **security-mesh.webp** (EnterpriseSecurity): Interconnected mesh with five node clusters
- **deployment-flow.webp** (EnterpriseHowItWorks): Sequential four-stage flow
- **sector-grid.webp** (THIS): Four-tile industry grid with sector divisions, compliance theme

### Alternative

Until the asset is created, the component will gracefully handle the missing image (Next.js Image component will show a placeholder or fail silently with the `hidden md:block` classes preventing layout shift).

---

**Related Assets:**
- `/illustrations/audit-ledger.webp` (EnterpriseHero)
- `/decor/eu-ledger-grid.webp` (EnterpriseSolution)
- `/decor/compliance-ledger.webp` (EnterpriseCompliance)
- `/decor/security-mesh.webp` (EnterpriseSecurity)
- `/decor/deployment-flow.webp` (EnterpriseHowItWorks)
